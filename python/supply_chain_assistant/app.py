"""Streamlit dashboard for inventory, forecasts, sourcing, and approvals."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd
import pydeck as pdk
import streamlit as st

if __package__ in (None, ""):
    package_root = str(Path(__file__).resolve().parent.parent)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)

from supply_chain_assistant.agents import (
    AgentToolRegistry,
    CostOptimizationReport,
    CostOptimizationRiskAgent,
)
from supply_chain_assistant.auto_replenishment import build_auto_purchase_drafts
from supply_chain_assistant.chat_agents import (
    MultiAgentChatDispatcher,
    SupplyChainChatContext,
)
from supply_chain_assistant.database import Database
from supply_chain_assistant.diversification import (
    DiversificationReport,
    ProductDiversificationAgent,
)
from supply_chain_assistant.email_dispatcher import EmailDispatcher, SMTPSettings
from supply_chain_assistant.geospatial import build_supply_chain_map_data
from supply_chain_assistant.models import (
    Document,
    InventoryItem,
    LandedCostRecord,
    PurchaseDraft,
    StockForecast,
    SupplierRecommendation,
    WorkflowResult,
)
from supply_chain_assistant.orchestrator import SupplyChainAssistant
from supply_chain_assistant.rag import LocalDocumentStore
from supply_chain_assistant.risk_radar import (
    SupplyChainRisk,
    detect_supply_chain_risks,
)
from supply_chain_assistant.sample_data import seed_demo_data
from supply_chain_assistant.scenario import simulate_inventory
from supply_chain_assistant.supplier_scorecard import (
    SupplierScore,
    calculate_supplier_scores,
)
from supply_chain_assistant.upload_data import (
    ProductUpload,
    merge_inventory_uploads,
    merge_supplier_documents,
    parse_inventory_table,
    parse_product_table,
    parse_supplier_table,
    read_uploaded_table,
)

ParsedUpload = TypeVar("ParsedUpload")

def _money(value: float, currency: str = "USD") -> str:
    return f"{currency} {value:,.2f}"


def _parse_upload(
    uploaded_file: Any,
    dataset_name: str,
    parser: Callable[[pd.DataFrame], ParsedUpload],
) -> ParsedUpload | None:
    if uploaded_file is None:
        return None
    try:
        return parser(read_uploaded_table(uploaded_file))
    except (ImportError, OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError) as error:
        st.error(f"{dataset_name} upload was not applied: {error}")
        return None


def _inventory_unit_costs(
    recommendations: tuple[SupplierRecommendation, ...],
    landed_cost_records: tuple[LandedCostRecord, ...],
) -> dict[str, tuple[float, str]]:
    unit_costs = {}
    latest_landed_costs = {}
    for record in landed_cost_records:
        current = latest_landed_costs.get(record.sku)
        if current is None or record.received_at > current.received_at:
            latest_landed_costs[record.sku] = record
    for sku, record in latest_landed_costs.items():
        unit_costs[sku] = (
            record.unit_price
            + (record.freight_cost + record.duties + record.handling_cost)
            / record.quantity,
            record.currency,
        )
    for recommendation in recommendations:
        sku = recommendation.need.item.sku
        if sku not in unit_costs:
            unit_costs[sku] = (
                recommendation.total_cost / recommendation.order_quantity
                if recommendation.order_quantity
                else recommendation.offer.unit_price,
                recommendation.offer.currency,
            )
    return unit_costs


def _record_date(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(value[:10])


def _supplier_location(document: Document) -> tuple[str, float, float] | None:
    metadata = document.metadata
    supplier = metadata.get("supplier")
    latitude = metadata.get("latitude")
    longitude = metadata.get("longitude")
    if document.category != "price_list" or supplier is None:
        return None
    if latitude is None or longitude is None:
        return None
    try:
        latitude_value = float(latitude)
        longitude_value = float(longitude)
    except (TypeError, ValueError):
        return None
    if not (-90 <= latitude_value <= 90 and -180 <= longitude_value <= 180):
        return None
    return str(supplier), latitude_value, longitude_value


def _location_label(location: tuple[str, float, float]) -> str:
    supplier, latitude, longitude = location
    return f"{supplier} · {latitude:.6f}, {longitude:.6f}"


def _filter_inventory_and_documents(
    all_inventory: tuple[InventoryItem, ...],
    all_documents: tuple[Document, ...],
    selected_categories: set[str],
    selected_locations: set[str],
    location_by_label: Mapping[str, tuple[str, float, float]],
) -> tuple[tuple[InventoryItem, ...], tuple[Document, ...], set[str]]:
    category_filtered = tuple(
        item
        for item in all_inventory
        if ((item.category or "").strip() or "Unclassified") in selected_categories
    )
    location_filter_active = selected_locations != set(location_by_label)
    selected_location_keys = {
        location_by_label[label]
        for label in selected_locations
        if label in location_by_label
    }
    if location_filter_active:
        location_skus = {
            str(document.metadata["sku"])
            for document in all_documents
            if _supplier_location(document) in selected_location_keys
            and document.metadata.get("sku") is not None
        }
        inventory = tuple(
            item for item in category_filtered if item.sku in location_skus
        )
    else:
        inventory = category_filtered

    inventory_skus = {item.sku for item in inventory}
    relevant_suppliers = {
        str(document.metadata["supplier"])
        for document in all_documents
        if (
            document.category == "price_list"
            and document.metadata.get("supplier") is not None
            and document.metadata.get("sku") is not None
            and str(document.metadata["sku"]) in inventory_skus
            and (
                not location_filter_active
                or _supplier_location(document) in selected_location_keys
            )
        )
    }

    filtered_documents = []
    for document in all_documents:
        location = _supplier_location(document)
        if (
            location_filter_active
            and document.category == "price_list"
            and location not in selected_location_keys
        ):
            continue
        sku = document.metadata.get("sku")
        if sku is not None and str(sku) not in inventory_skus:
            continue
        supplier = document.metadata.get("supplier")
        if (
            document.category == "supplier_contract"
            and supplier is not None
            and str(supplier) not in relevant_suppliers
        ):
            continue
        filtered_documents.append(document)

    return inventory, tuple(filtered_documents), relevant_suppliers


def _render_executive_kpis(
    inventory: tuple[InventoryItem, ...],
    recommendations: tuple[SupplierRecommendation, ...],
    landed_cost_records: tuple[LandedCostRecord, ...],
    supplier_otif: float | None,
    order_fill_rate: float | None,
    currency: str,
) -> None:
    unit_costs = _inventory_unit_costs(recommendations, landed_cost_records)
    valued_items = [
        (item, unit_costs[item.sku])
        for item in inventory
        if item.sku in unit_costs and item.on_hand != 0
    ]
    missing_cost_items = [
        item for item in inventory if item.on_hand > 0 and item.sku not in unit_costs
    ]
    currencies = {item_cost[1] for _, item_cost in valued_items}

    if len(currencies) > 1:
        inventory_value = "Mixed currencies"
        inventory_detail = "Cannot combine inventory values across currencies"
        inventory_status = "warning"
    elif currencies:
        inventory_currency = next(iter(currencies))
        total_value = sum(
            item.on_hand * unit_cost[0] for item, unit_cost in valued_items
        )
        inventory_value = (
            f"~{_money(total_value, inventory_currency)}"
            if missing_cost_items
            else _money(total_value, inventory_currency)
        )
        inventory_detail = (
            f"Partial value; {len(missing_cost_items)} stocked SKU(s) lack cost data"
            if missing_cost_items
            else "All stocked SKUs valued"
        )
        inventory_status = "warning" if missing_cost_items else "good"
    elif not missing_cost_items:
        inventory_value = _money(0, currency)
        inventory_detail = "No unpriced stock on hand"
        inventory_status = "good"
    else:
        inventory_value = "Unavailable"
        inventory_detail = "No unit-cost data is available"
        inventory_status = "unavailable"

    critical_stockouts = sum(item.on_hand <= 0 for item in inventory)
    stockout_detail = (
        "No current stockouts"
        if critical_stockouts == 0
        else f"{critical_stockouts} SKU(s) currently out of stock"
    )

    if supplier_otif is None:
        supplier_risk_value = "Unavailable"
        supplier_risk_detail = "No due supplier-delivery records"
        supplier_risk_status = "unavailable"
    else:
        supplier_risk = (1 - supplier_otif) * 100
        supplier_risk_value = f"{supplier_risk:.0f}/100"
        if supplier_risk <= 20:
            supplier_risk_detail = "Low risk · based on supplier OTIF"
            supplier_risk_status = "good"
        elif supplier_risk <= 40:
            supplier_risk_detail = "Moderate risk · based on supplier OTIF"
            supplier_risk_status = "warning"
        else:
            supplier_risk_detail = "High risk · based on supplier OTIF"
            supplier_risk_status = "critical"

    if order_fill_rate is None:
        fill_rate_value = "Unavailable"
        fill_rate_detail = "No fulfillment records available"
        fill_rate_status = "unavailable"
    else:
        fill_rate_value = f"{order_fill_rate:.1%}"
        fill_rate_detail = (
            "Meets 95% target"
            if order_fill_rate >= 0.95
            else "Below 95% target"
        )
        fill_rate_status = "good" if order_fill_rate >= 0.95 else "critical"

    cards = (
        (
            "inventory_value",
            "💰 Total inventory value",
            inventory_value,
            inventory_detail,
            inventory_status,
        ),
        (
            "critical_stockouts",
            "📦 Critical stockouts",
            str(critical_stockouts),
            stockout_detail,
            "good" if critical_stockouts == 0 else "critical",
        ),
        (
            "supplier_risk",
            "🏭 Supplier risk score",
            supplier_risk_value,
            supplier_risk_detail,
            supplier_risk_status,
        ),
        (
            "order_fill_rate",
            "✅ Overall order fill rate",
            fill_rate_value,
            fill_rate_detail,
            fill_rate_status,
        ),
    )
    status_colors = {
        "good": "#138a55",
        "warning": "#d99b20",
        "critical": "#d64545",
        "unavailable": "#83918a",
    }
    card_styles = "\n".join(
        f".st-key-executive_kpi_{key} [data-testid='stMetric'] "
        f"{{ border-top: 3px solid {status_colors[status]}; }}"
        for key, _, _, _, status in cards
    )
    st.markdown(f"<style>{card_styles}</style>", unsafe_allow_html=True)

    columns = st.columns(4)
    for column, (key, label, value, detail, status) in zip(columns, cards):
        with column:
            with st.container(key=f"executive_kpi_{key}"):
                st.metric(label, value)
                status_icon = {
                    "good": "🟢",
                    "warning": "🟠",
                    "critical": "🔴",
                    "unavailable": "⚪",
                }[status]
                st.caption(f"{status_icon} {detail}")


def _render_risk_radar(risks: tuple[SupplyChainRisk, ...]) -> None:
    st.subheader("AI Risk Radar & Automated Anomaly Detector")
    st.caption(
        "Rule-based anomaly checks on current dashboard data. The reorder point is "
        "used as the configured stock threshold; cost variances require comparable "
        "supplier, SKU, and currency data."
    )
    critical_count = sum(risk.severity == "Critical" for risk in risks)
    warning_count = sum(risk.severity == "Warning" for risk in risks)
    late_delivery_count = sum(
        risk.category == "Supplier delay" for risk in risks
    )
    cost_variance_count = sum(
        risk.category in {"Cost variance", "Landed-cost variance"}
        for risk in risks
    )

    metrics = st.columns(4)
    metrics[0].metric("Active risks", len(risks))
    metrics[1].metric("Critical", critical_count)
    metrics[2].metric("Late deliveries", late_delivery_count)
    metrics[3].metric("Cost variances", cost_variance_count)

    if not risks:
        st.success("No active anomalies were detected in the selected data.")
        return

    for risk in risks:
        render_alert = st.error if risk.severity == "Critical" else st.warning
        render_alert(
            f"**{risk.category} · {risk.title}**\n\n"
            f"{risk.description}\n\n"
            f"**Recommended action:** {risk.action}"
        )


def _render_supplier_scorecard(scores: tuple[SupplierScore, ...]) -> None:
    st.subheader("Supplier Scorecard & Ranking")
    st.caption(
        "Composite score: OTIF 50%, lead-time consistency 25%, and comparable "
        "unit-cost competitiveness 25%. Available component weights are "
        "renormalized when evidence is missing. Cost is compared only for the "
        "same SKU and currency; lead-time consistency is based on the spread "
        "of actual-versus-promised delivery days."
    )
    if not scores:
        st.info(
            "No supplier delivery history or comparable price data is available "
            "for the selected filters."
        )
        return

    tier_a_count = sum(score.tier == "Tier A" for score in scores)
    tier_c_count = sum(score.tier == "Tier C" for score in scores)
    average_score = sum(score.score for score in scores) / len(scores)
    summary = st.columns(3)
    summary[0].metric("Ranked suppliers", len(scores))
    summary[1].metric("Average score", f"{average_score:.1f}/100")
    summary[2].metric("Tier A / Tier C", f"{tier_a_count} / {tier_c_count}")

    chart_data = pd.DataFrame(
        [{"Supplier": score.supplier, "Score": round(score.score, 1)}
         for score in scores]
    ).set_index("Supplier")
    st.markdown("#### Overall performance")
    st.bar_chart(chart_data, y="Score", y_label="Score (0–100)")

    st.markdown("#### Ranked scorecard")
    rows = [
        {
            "Rank": rank,
            "Supplier": score.supplier,
            "Score": round(score.score, 1),
            "Tier": score.tier,
            "OTIF": (
                f"{score.otif_rate:.1%}"
                if score.otif_rate is not None else "Unavailable"
            ),
            "Lead-time consistency": (
                f"{score.lead_time_consistency:.1f}/100"
                if score.lead_time_consistency is not None else "Unavailable"
            ),
            "Cost competitiveness": (
                f"{score.cost_competitiveness:.1f}/100"
                if score.cost_competitiveness is not None else "Unavailable"
            ),
            "Deliveries assessed": score.delivery_count,
            "Comparable SKU/currency prices": score.price_comparison_count,
            "Components observed": f"{score.observed_components}/3",
        }
        for rank, score in enumerate(scores, start=1)
    ]
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score",
                min_value=0,
                max_value=100,
                format="%.1f",
            ),
        },
    )

    lowest_score = scores[-1]
    if lowest_score.tier == "Tier C":
        st.warning(
            f"**Supplier to review:** {lowest_score.supplier} is ranked lowest "
            f"at {lowest_score.score:.1f}/100 ({lowest_score.tier}). Review the "
            "component scores and qualify a recovery or alternate-source plan."
        )
    st.caption(
        "Tiers: A ≥ 80, B ≥ 60, C < 60. A cost score is available only when "
        "at least two suppliers have a comparable price for the same SKU and "
        "currency. A supplier with limited observations may still have a score; "
        "check the coverage columns before making sourcing decisions."
    )


def _render_scenario_simulation(
    inventory: tuple[InventoryItem, ...],
    forecasts: tuple[StockForecast, ...],
    unit_costs: Mapping[str, tuple[float, str]],
    lead_time_multiplier_pct: float,
    demand_surge_pct: float,
    holding_cost_adjustment_pct: float,
    baseline_carrying_rate_pct: float,
) -> None:
    st.subheader("What-if scenario")
    st.caption(
        "A deterministic demand-trend projection through the adjusted supplier lead time. "
        "Stockout exposure is valued at estimated unit cost; annual holding cost compares "
        "current inventory with projected inventory at delivery."
    )
    projections = simulate_inventory(
        inventory,
        forecasts,
        unit_costs,
        lead_time_multiplier_pct,
        demand_surge_pct,
        holding_cost_adjustment_pct,
        baseline_carrying_rate_pct,
    )
    stockout_count = sum(projection.stockout_risk for projection in projections)
    reorder_risk_count = sum(
        projection.stockout_risk
        or projection.risk_level == "At/below reorder point"
        for projection in projections
    )
    metrics = st.columns(3)
    metrics[0].metric("SKUs at stockout risk", stockout_count)
    metrics[1].metric("SKUs at/below reorder point", reorder_risk_count)
    metrics[2].metric(
        "Units short before delivery",
        f"{sum(projection.stockout_units for projection in projections):,.2f}",
    )

    st.subheader("Projected inventory and stockout risk")
    rows = [
        {
            "SKU": projection.sku,
            "Product": projection.product_name,
            "Adjusted lead time (days)": projection.adjusted_lead_time_days,
            "Projected inventory at delivery": projection.projected_inventory,
            "Demand before delivery": projection.demand_during_lead_time,
            "Risk": projection.risk_level,
            "Days until stockout": (
                f"{projection.days_until_stockout:.2f}"
                if projection.days_until_stockout is not None
                else "Not during lead time"
            ),
            "Units short": projection.stockout_units,
            "Stockout exposure": (
                _money(projection.stockout_cost_exposure, projection.currency)
                if projection.stockout_cost_exposure is not None
                and projection.currency is not None
                else "Unavailable"
            ),
            "Annual holding-cost change": (
                _money(projection.annual_holding_cost_change, projection.currency)
                if projection.annual_holding_cost_change is not None
                and projection.currency is not None
                else "Unavailable"
            ),
        }
        for projection in projections
    ]
    st.dataframe(rows, width="stretch", hide_index=True)

    st.subheader("Estimated financial impact")
    valued_projections = [
        projection
        for projection in projections
        if projection.currency is not None
    ]
    if not valued_projections:
        st.info("No unit-cost records are available to estimate financial impacts.")
    else:
        currencies = sorted(
            {projection.currency for projection in valued_projections if projection.currency}
        )
        for currency in currencies:
            currency_projections = [
                projection
                for projection in valued_projections
                if projection.currency == currency
            ]
            stockout_exposure = sum(
                projection.stockout_cost_exposure or 0.0
                for projection in currency_projections
            )
            holding_cost_change = sum(
                projection.annual_holding_cost_change or 0.0
                for projection in currency_projections
            )
            columns = st.columns(2)
            columns[0].metric(
                f"Stockout exposure ({currency})",
                _money(stockout_exposure, currency),
            )
            holding_cost_label = (
                f"+{_money(holding_cost_change, currency)}"
                if holding_cost_change > 0
                else _money(holding_cost_change, currency)
            )
            columns[1].metric(
                f"Annual holding-cost change ({currency})",
                holding_cost_label,
            )

    st.caption(
        f"Baseline carrying rate: {baseline_carrying_rate_pct:g}% per year; "
        f"scenario adjustment: {holding_cost_adjustment_pct:+g}% of that rate. "
        "Unit values use the latest landed cost where available, otherwise a current "
        "supplier recommendation. SKUs without a unit cost are excluded from dollar totals."
    )


def _render_inventory_tab(
    database: Database,
    inventory: tuple[InventoryItem, ...],
    *,
    read_only: bool = False,
) -> None:
    st.subheader("Inventory")
    if read_only:
        st.caption(
            "Inventory is read-only while an uploaded inventory or product dataset is active. "
            "Clear the upload to return to saved inventory editing."
        )
        st.dataframe(
            [
                {
                    "SKU": item.sku,
                    "Product": item.name,
                    "On hand": item.on_hand,
                    "Reorder point": item.reorder_point,
                    "Average daily usage": item.average_daily_usage,
                    "Target stock": item.target_stock,
                    "Lead time (days)": item.supplier_lead_time_days,
                    "Category": item.category or "Unclassified",
                    "Warehouse latitude": item.warehouse_latitude,
                    "Warehouse longitude": item.warehouse_longitude,
                }
                for item in inventory
            ],
            width="stretch",
            hide_index=True,
        )
        return

    for item in inventory:
        with st.expander(f"{item.sku}  ·  {item.name}", expanded=False):
            with st.form(f"inventory_{item.sku}"):
                columns = st.columns(4)
                on_hand = columns[0].number_input(
                    "On hand", min_value=0, value=item.on_hand, key=f"on_hand_{item.sku}"
                )
                reorder_point = columns[1].number_input(
                    "Reorder point", min_value=0, value=item.reorder_point,
                    key=f"reorder_{item.sku}",
                )
                target_stock = columns[2].number_input(
                    "Target stock", min_value=0, value=item.target_stock,
                    key=f"target_{item.sku}",
                )
                lead_days = columns[3].number_input(
                    "Lead time (days)", min_value=0,
                    value=item.supplier_lead_time_days, key=f"lead_{item.sku}",
                )
                category = st.text_input(
                    "Product category",
                    value=item.category or "",
                    key=f"category_{item.sku}",
                )
                if st.form_submit_button("Save inventory"):
                    database.save_inventory_item(
                        InventoryItem(
                            item.sku,
                            item.name,
                            int(on_hand),
                            int(reorder_point),
                            item.average_daily_usage,
                            int(target_stock),
                            int(lead_days),
                            item.warehouse_latitude,
                            item.warehouse_longitude,
                            category.strip() or None,
                        ),
                        int(lead_days),
                    )
                    st.rerun()

    with st.expander("Record daily usage"):
        if inventory:
            with st.form("daily_usage"):
                selected_item = st.selectbox(
                    "Product", inventory, format_func=lambda item: f"{item.sku} · {item.name}"
                )
                usage_date = st.date_input("Date", value=date.today())
                units_used = st.number_input("Units used", min_value=0.0, step=1.0)
                if st.form_submit_button("Save usage"):
                    database.save_usage(selected_item.sku, usage_date, units_used)
                    st.rerun()


def _render_contacts(
    database: Database,
    suppliers: set[str] | None = None,
) -> None:
    st.subheader("Supplier contacts")
    contacts = database.list_supplier_contacts()
    for supplier, email in contacts.items():
        if suppliers is not None and supplier not in suppliers:
            continue
        with st.form(f"contact_{supplier}"):
            new_email = st.text_input("Recipient email", value=email, key=f"email_{supplier}")
            if st.form_submit_button(f"Save {supplier}"):
                database.save_supplier_contact(supplier, new_email)
                st.rerun()
    st.caption("Sample .example addresses are intentionally blocked from SMTP delivery.")


def _render_purchase_review(
    database: Database,
    recommendations: tuple,
    drafts: tuple[PurchaseDraft, ...],
    approved: bool,
    reason: str,
    approver: str,
) -> None:
    st.subheader("Supplier recommendations")
    if not recommendations:
        st.info("No supplier recommendations for the current inventory.")
        return
    draft_by_sku = {draft.sku: draft for draft in drafts}
    if not approved:
        st.warning(f"Financial review blocked purchase drafts. {reason}")

    for recommendation in recommendations:
        offer = recommendation.offer
        item = recommendation.need.item
        with st.container(border=True):
            columns = st.columns([2, 1, 1])
            columns[0].markdown(f"**{item.name}**  \n`{item.sku}` · {offer.supplier}")
            columns[1].metric("Order quantity", recommendation.order_quantity)
            columns[2].metric("Estimated total", _money(recommendation.total_cost, offer.currency))
            columns[0].caption(
                f"{_money(offer.unit_price, offer.currency)} per unit · "
                f"{offer.lead_time_days} day lead · trigger: {recommendation.need.trigger}"
            )
            for evidence in recommendation.evidence:
                st.caption(f"Source: {evidence.document.source} · {evidence.excerpt}")

            draft = draft_by_sku.get(item.sku)
            if draft is None:
                st.caption("A purchase draft is unavailable until financial approval passes.")
                continue

            with st.expander("Review email draft", expanded=False):
                st.text_input("Subject", value=draft.subject, disabled=True, key=f"subject_{item.sku}")
                st.text_area("Message", value=draft.body, height=180, disabled=True, key=f"body_{item.sku}")
                contact = database.get_supplier_contact(draft.supplier) or ""
                with st.form(f"send_{item.sku}"):
                    recipient = st.text_input(
                        "Send to", value=contact, key=f"send_to_{item.sku}"
                    )
                    approver_name = st.text_input(
                        "Approved by", value=approver, key=f"approved_by_{item.sku}"
                    )
                    can_send = approved and bool(approver_name.strip()) and not recipient.lower().endswith(".example")
                    submitted = st.form_submit_button(
                        "Approve & send email", type="primary", disabled=not can_send
                    )
                    if submitted:
                        if "@" not in recipient:
                            st.error("Enter a valid supplier email address.")
                        else:
                            try:
                                settings = SMTPSettings.from_environment()
                                database.save_supplier_contact(draft.supplier, recipient)
                                order_id = database.create_approved_order(
                                    draft,
                                    recipient=recipient,
                                    approved_by=approver_name,
                                )
                                EmailDispatcher(database, settings).send_approved_order(order_id)
                                st.success(f"Email sent and order #{order_id} logged.")
                                st.rerun()
                            except Exception as error:
                                st.error(f"Email was not sent: {error}")


def _render_manual_purchase_order_form(
    database: Database,
    inventory: tuple[InventoryItem, ...],
    documents: tuple[Document, ...],
) -> None:
    st.subheader("Create a manual pending purchase order")
    items_by_sku = {item.sku: item for item in inventory}
    offers = []
    for document in documents:
        if document.category != "price_list":
            continue
        sku = str(document.metadata.get("sku", ""))
        item = items_by_sku.get(sku)
        supplier = document.metadata.get("supplier")
        if item is None or supplier is None:
            continue
        try:
            unit_price = float(document.metadata["unit_price"])
            lead_time_days = int(document.metadata.get("lead_time_days", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if unit_price < 0 or lead_time_days < 0:
            continue
        currency = str(document.metadata.get("currency", "USD")).upper()
        label = (
            f"{item.sku} · {item.name} — {supplier} "
            f"({currency} {unit_price:,.2f}/unit, {lead_time_days}-day lead)"
        )
        offers.append((label, item, str(supplier), unit_price, currency, lead_time_days))

    if not offers:
        st.info(
            "A manual pending order requires an inventory product and a matching "
            "supplier price-list offer."
        )
        return

    with st.form("manual_pending_purchase_order", clear_on_submit=True):
        selected_offer = st.selectbox(
            "Product and supplier offer",
            offers,
            format_func=lambda offer: offer[0],
        )
        _, item, supplier, unit_price, currency, lead_time_days = selected_offer
        quantity = st.number_input(
            "Quantity",
            min_value=1,
            value=max(1, item.target_stock - item.on_hand),
            step=1,
        )
        expected_date = st.date_input(
            "Expected delivery date",
            value=date.today() + timedelta(days=lead_time_days),
            min_value=date.today(),
        )
        submitted = st.form_submit_button(
            "Save pending order", type="primary"
        )
        if submitted:
            try:
                pending_id, _ = database.create_pending_purchase_order(
                    sku=item.sku,
                    supplier=supplier,
                    quantity=int(quantity),
                    unit_price=unit_price,
                    currency=currency,
                    expected_date=expected_date,
                    source="manual",
                )
            except (TypeError, ValueError, RuntimeError) as error:
                st.error(f"Pending order was not saved: {error}")
            else:
                st.success(
                    f"Pending order #{pending_id} saved for review. "
                    "It has not been approved or sent."
                )
                st.rerun()


def _create_auto_purchase_orders(
    database: Database,
    inventory: tuple[InventoryItem, ...],
    documents: tuple[Document, ...],
) -> int:
    existing_pending_skus = {
        str(order["sku"])
        for order in database.list_pending_purchase_orders()
    }
    created_count = 0
    for draft in build_auto_purchase_drafts(inventory, documents):
        if draft.sku in existing_pending_skus:
            continue
        _, created = database.create_pending_purchase_order(
            sku=draft.sku,
            supplier=draft.supplier,
            quantity=draft.quantity,
            unit_price=draft.unit_price,
            currency=draft.currency,
            expected_date=draft.expected_date,
            source="auto",
            idempotency_key=f"auto-replenishment:{draft.sku}",
        )
        created_count += int(created)
        existing_pending_skus.add(draft.sku)
    return created_count


def _build_chat_tool_registry(
    database: Database,
    inventory: tuple[InventoryItem, ...],
    documents: tuple[Document, ...],
) -> AgentToolRegistry:
    registry = AgentToolRegistry()

    def filter_inventory(arguments: Mapping[str, object]) -> str:
        sku = str(arguments.get("sku", "")).strip().casefold()
        low_stock = arguments.get("low_stock") is True
        selected = tuple(
            item
            for item in inventory
            if (not sku or sku in item.sku.casefold() or sku in item.name.casefold())
            and (not low_stock or item.on_hand <= item.reorder_point)
        )
        if not selected:
            return "No inventory items match the requested filters."
        return "\n".join(
            f"{item.sku} ({item.name}): {item.on_hand} on hand; "
            f"reorder point {item.reorder_point}; target {item.target_stock}."
            for item in selected
        )

    def create_auto_purchase_orders(_: Mapping[str, object]) -> str:
        created_count = _create_auto_purchase_orders(database, inventory, documents)
        pending_auto_orders = tuple(
            order
            for order in database.list_pending_purchase_orders()
            if order["source"] == "auto"
        )
        summaries = "\n".join(
            f"- {order['sku']}: {order['quantity']} units from {order['supplier']} "
            f"({order['currency']} {order['total_cost']:,.2f}), expected "
            f"{order['expected_date']}."
            for order in pending_auto_orders
        )
        status = (
            f"Created {created_count} new pending Auto-PO draft(s)."
            if created_count
            else "No new pending Auto-PO drafts were created; eligible products may "
            "already have pending orders or lack a valid supplier quote."
        )
        return (
            f"{status} Drafts require human review; none were approved or sent."
            + (f"\nCurrent pending Auto-PO drafts:\n{summaries}" if summaries else "")
        )

    def refresh_risk_radar(_: Mapping[str, object]) -> str:
        risks = detect_supply_chain_risks(
            inventory,
            database.list_orders(),
            database.list_supplier_deliveries(),
            database.list_landed_cost_records(),
            documents,
        )
        if not risks:
            return "Risk radar refreshed: no current risks were detected."
        return "Risk radar refreshed:\n" + "\n".join(
            f"- {risk.severity} — {risk.title}: {risk.description} "
            f"Action: {risk.action}"
            for risk in risks
        )

    registry.register(
        "filter_inventory",
        agent="InventoryTrackingAgent",
        description="Filter current inventory by SKU/name and/or reorder status.",
        handler=filter_inventory,
    )
    registry.register(
        "create_auto_purchase_orders",
        agent="NegotiationAndProposalAgent",
        description="Create idempotent pending Auto-PO drafts for eligible stock.",
        handler=create_auto_purchase_orders,
    )
    registry.register(
        "refresh_risk_radar",
        agent="CostOptimizationRiskAgent",
        description="Recompute risks from the latest database records.",
        handler=refresh_risk_radar,
    )
    return registry


def _render_pending_purchase_orders(
    pending_orders: tuple[dict[str, Any], ...],
) -> None:
    st.subheader("Pending purchase orders")
    if not pending_orders:
        st.info("There are no pending purchase orders.")
        return
    st.caption(
        "Pending orders are drafts for review only. They are not approved or sent."
    )
    st.dataframe(
        [
            {
                "Draft": order["pending_id"],
                "Source": "Automated" if order["source"] == "auto" else "Manual",
                "SKU": order["sku"],
                "Supplier": order["supplier"],
                "Quantity": order["quantity"],
                "Unit price": _money(order["unit_price"], order["currency"]),
                "Total": _money(order["total_cost"], order["currency"]),
                "Expected delivery": order["expected_date"],
                "Created": order["created_at"],
                "Status": order["status"].title(),
            }
            for order in pending_orders
        ],
        width="stretch",
        hide_index=True,
    )


def _render_cost_optimization(report: CostOptimizationReport) -> None:
    st.subheader("Supply chain performance indicators")
    metric_columns = st.columns(len(report.metrics))
    for column, metric in zip(metric_columns, report.metrics):
        column.metric(metric.name, metric.value)
        column.caption(metric.detail)

    st.subheader("Cost and risk recommendations")
    if report.findings:
        for finding in report.findings:
            with st.container(border=True):
                st.markdown(
                    f"**{finding.priority} · {finding.area}: {finding.title}**"
                )
                st.write(finding.evidence)
                st.caption(f"Recommended action: {finding.action}")
    else:
        st.info("No cost or risk findings for the current data.")

    with st.expander("Data needed for complete KPI and risk analysis"):
        for gap in report.data_gaps:
            st.markdown(f"- {gap}")


def _render_diversification_analysis(report: DiversificationReport) -> None:
    st.subheader("Product portfolio performance")
    st.caption(
        "Fulfillment event counts represent recorded demand transactions, not customer "
        "frequency or revenue. Category turns are an annualized usage/on-hand proxy; "
        "the overall turns metric uses reported COGS and average inventory value."
    )
    metrics = st.columns(3)
    metrics[0].metric(
        "Portfolio inventory turns",
        f"{report.overall_inventory_turns:.2f}x"
        if report.overall_inventory_turns is not None
        else "Unavailable",
    )
    metrics[1].metric(
        "Recorded demand fill rate",
        f"{report.overall_fill_rate:.1%}"
        if report.overall_fill_rate is not None
        else "Unavailable",
    )
    metrics[2].metric(
        "Supplier OTIF",
        f"{report.overall_supplier_otif:.1%}"
        if report.overall_supplier_otif is not None
        else "Unavailable",
    )

    st.subheader("Category comparison")
    if report.categories:
        st.dataframe(
            [
                {
                    "Category": category.category,
                    "SKUs": category.sku_count,
                    "Fulfillment events": category.fulfillment_events,
                    "Requested units": category.requested_units,
                    "Fulfilled units": category.fulfilled_units,
                    "Fill rate": (
                        f"{category.fill_rate:.1%}"
                        if category.fill_rate is not None else "Unavailable"
                    ),
                    "Estimated annual turns": (
                        f"{category.estimated_annual_turns:.2f}x"
                        if category.estimated_annual_turns is not None
                        else "Unavailable"
                    ),
                    "Supplier OTIF": (
                        f"{category.supplier_otif:.1%}"
                        if category.supplier_otif is not None else "Unavailable"
                    ),
                    "Deliveries due": category.supplier_deliveries_due,
                }
                for category in report.categories
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No inventory categories are available for analysis.")

    st.subheader("Diversification advisor")
    if report.llm_commentary:
        st.markdown(report.llm_commentary)
    else:
        st.caption(
            "Recommendations below are evidence-driven rules, not generative AI output. "
            "An optional configured text-generation client can add commentary."
        )
    for suggestion in report.suggestions:
        with st.container(border=True):
            st.markdown(f"**{suggestion.priority} · {suggestion.title}**")
            st.write(suggestion.rationale)
            st.caption(f"Risk control: {suggestion.risk_guardrail}")


def _render_supply_chain_map(
    inventory: tuple[InventoryItem, ...],
    forecasts: tuple[StockForecast, ...],
    recommendations: tuple[SupplierRecommendation, ...],
    documents: tuple[Document, ...],
) -> None:
    st.subheader("Supplier, warehouse, and route map")
    supplier_points, warehouse_points, routes = build_supply_chain_map_data(
        inventory,
        forecasts,
        recommendations,
        documents,
    )
    if not supplier_points and not warehouse_points:
        st.info(
            "No mapped locations are available yet. Add supplier_latitude and "
            "supplier_longitude to a supplier upload, and warehouse_latitude and "
            "warehouse_longitude to an inventory or product upload. Coordinates must "
            "use decimal degrees."
        )
        return

    layers = []
    if routes:
        layers.append(
            pdk.Layer(
                "ArcLayer",
                data=routes,
                get_source_position="source",
                get_target_position="target",
                get_source_color=[18, 107, 82, 190],
                get_target_color=[245, 158, 11, 190],
                get_width=5,
                pickable=True,
                auto_highlight=True,
            )
        )
    if supplier_points:
        supplier_rows = [
            {
                **point,
                "risk": "Supplier",
                "on_hand": "-",
                "projected_stock": "-",
                "reorder_point": "-",
                "supplier": point["name"],
                "quantity": "-",
            }
            for point in supplier_points
        ]
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=supplier_rows,
                get_position="[longitude, latitude]",
                get_fill_color=[37, 99, 235, 210],
                get_line_color=[255, 255, 255, 230],
                get_line_width=2,
                line_width_min_pixels=1,
                get_radius=9000,
                radius_min_pixels=7,
                radius_max_pixels=18,
                stroked=True,
                pickable=True,
                auto_highlight=True,
            )
        )
    if warehouse_points:
        risk_colors = {
            "High": [220, 38, 38, 220],
            "Elevated": [245, 158, 11, 220],
            "Normal": [22, 163, 74, 220],
        }
        warehouse_rows = [
            {
                **point,
                "color": risk_colors[str(point["risk"])],
                "skus": point["name"],
                "supplier": "-",
                "quantity": "-",
            }
            for point in warehouse_points
        ]
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=warehouse_rows,
                get_position="[longitude, latitude]",
                get_fill_color="color",
                get_line_color=[255, 255, 255, 230],
                get_line_width=2,
                line_width_min_pixels=1,
                get_radius=12000,
                radius_min_pixels=8,
                radius_max_pixels=20,
                stroked=True,
                pickable=True,
                auto_highlight=True,
            )
        )

    coordinates = [
        [float(point["longitude"]), float(point["latitude"])]
        for point in supplier_points + warehouse_points
    ]
    center_longitude = sum(point[0] for point in coordinates) / len(coordinates)
    center_latitude = sum(point[1] for point in coordinates) / len(coordinates)
    view_state = pdk.ViewState(
        latitude=center_latitude,
        longitude=center_longitude,
        zoom=2 if len(coordinates) > 1 else 6,
        pitch=35,
    )
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={
            "html": (
                "<b>{name}</b><br/>SKU(s): {skus}<br/>Risk: {risk}<br/>"
                "On hand: {on_hand}<br/>Projected: {projected_stock}<br/>"
                "Supplier: {supplier}<br/>Order quantity: {quantity}"
            ),
            "style": {"backgroundColor": "#173d31", "color": "white"},
        },
    )
    st.pydeck_chart(deck, width="stretch", height=560)
    st.caption(
        "Blue markers are suppliers; warehouse markers show forecast stock risk "
        "(red: high, amber: elevated, green: normal). Arcs show recommended supplier "
        "shipments where both endpoint coordinates are provided."
    )


def _render_floating_chat(
    dispatcher: MultiAgentChatDispatcher,
    inventory: tuple[InventoryItem, ...],
    result: WorkflowResult,
    cost_report: CostOptimizationReport,
    database: Database,
    documents: tuple[Document, ...],
) -> None:
    with st.container(key="floating_chat_widget"):
        with st.popover("💬", help="Open chat", key="supply_chain_chat_popover"):
            st.markdown("### Supply chain assistant")
            st.caption(
                "Specialists cover inventory, forecasts, costs, suppliers, and risk. "
                "You can also filter inventory, generate pending Auto-PO drafts, "
                "or refresh the risk radar."
            )
            if st.button("Clear conversation", key="clear_supply_chain_chat"):
                st.session_state["supply_chain_chat_messages"] = []
                st.rerun()

            messages = st.session_state.setdefault("supply_chain_chat_messages", [])
            with st.container(key="floating_chat_history", height=300):
                for message in messages:
                    with st.chat_message(message["role"]):
                        routed_agents = message.get("agents", ())
                        if message["role"] == "assistant" and routed_agents:
                            st.caption(f"Routed to: {', '.join(routed_agents)}")
                        st.markdown(message["content"])

            prompt = st.chat_input(
                "Ask a supply chain question",
                key="supply_chain_chat_input",
            )
            if not prompt:
                return

            history = tuple(
                (message["role"], message["content"]) for message in messages
            )
            messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            context = SupplyChainChatContext(
                inventory=inventory,
                forecasts=result.forecasts,
                recommendations=result.recommendations,
                financial_assessment=result.financial_assessment,
                cost_report=cost_report,
                tool_registry=_build_chat_tool_registry(
                    database,
                    inventory,
                    documents,
                ),
            )
            with st.chat_message("assistant"):
                with st.spinner("Reviewing the latest supply chain data..."):
                    dispatch_result = dispatcher.dispatch(
                        prompt,
                        context=context,
                        history=history,
                    )
                st.caption(f"Routed to: {', '.join(dispatch_result.agents)}")
                st.markdown(dispatch_result.response)
            messages.append(
                {
                    "role": "assistant",
                    "content": dispatch_result.response,
                    "agents": dispatch_result.agents,
                }
            )
            st.rerun()


def _render_theme_styles(dark_mode: bool) -> None:
    palette = (
        {
            "background": "#111a17",
            "surface": "#1c2924",
            "sidebar": "#16211c",
            "text": "#e5eee9",
            "muted": "#a2b4aa",
            "border": "#34483e",
            "header": "rgba(17, 26, 23, 0.92)",
        }
        if dark_mode
        else {
            "background": "#f3f6f3",
            "surface": "#ffffff",
            "sidebar": "#edf2ee",
            "text": "#1e2d27",
            "muted": "#53675e",
            "border": "#dce5df",
            "header": "rgba(243, 246, 243, 0.92)",
        }
    )
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
        .stApp {{ background: {palette["background"]}; color: {palette["text"]}; }}
        [data-testid="stHeader"] {{ background: {palette["header"]}; }}
        [data-testid="stSidebar"] {{ background: {palette["sidebar"]}; }}
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"],
        [data-testid="stWidgetLabel"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] label,
        label {{ color: {palette["text"]}; }}
        [data-testid="stCaptionContainer"] {{ color: {palette["muted"]}; }}
        [data-testid="stMetric"] {{
            background: {palette["surface"]};
            border: 1px solid {palette["border"]};
            padding: 14px 16px;
            border-radius: 6px;
        }}
        [data-testid="stMetricLabel"] {{ color: {palette["muted"]}; }}
        [data-testid="stMetricValue"] {{ color: {palette["text"]}; }}
        h1, h2, h3 {{ color: {"#d6eee2" if dark_mode else "#173d31"}; letter-spacing: 0; }}
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] > div,
        [data-testid="stDateInput"] input {{
            background-color: {palette["surface"]};
            border-color: {palette["border"]};
            color: {palette["text"]};
        }}
        div.stButton > button[kind="primary"],
        button[kind="primaryFormSubmit"] {{
            background: #126b52; border-color: #126b52;
        }}
        div.st-key-floating_chat_widget {{
            position: fixed; right: 1.5rem; bottom: 1.5rem; z-index: 1000;
            width: auto;
        }}
        div.st-key-floating_chat_widget [data-testid="stPopoverButton"] {{
            width: 3.5rem; height: 3.5rem; padding: 0;
            border-radius: 999px; background: #126b52; color: #fff;
            border-color: #126b52; font-size: 1.35rem;
            box-shadow: 0 4px 16px rgba(23, 61, 49, 0.25);
        }}
        div.st-key-floating_chat_widget [data-testid="stPopoverBody"] {{
            width: min(24rem, calc(100vw - 2rem));
            max-height: min(35rem, calc(100vh - 7rem));
            box-sizing: border-box; overflow-y: auto;
            background: {palette["surface"]};
            color: {palette["text"]};
            border: 1px solid {palette["border"]}; border-radius: 12px;
            box-shadow: 0 8px 32px rgba(23, 61, 49, 0.22);
        }}
        div.st-key-floating_chat_history {{
            height: min(240px, 26vh) !important;
            min-height: 5rem;
            flex: 1 1 auto;
            overflow-y: auto;
            border: 0;
        }}
        @media (max-width: 480px) {{
            div.st-key-floating_chat_widget {{
                right: 0.75rem; bottom: 0.75rem;
            }}
            div.st-key-floating_chat_widget [data-testid="stPopoverBody"] {{
                width: calc(100vw - 1rem);
                max-height: min(35rem, calc(100vh - 6rem));
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Supply Chain Assistant", page_icon="S", layout="wide")
    st.session_state.setdefault("dark_mode", False)

    database = Database()
    seed_demo_data(database)
    st.title("Supply Chain Assistant")

    with st.sidebar:
        st.toggle(
            "🌙 Dark mode",
            key="dark_mode",
            help="Switch between the light and dark dashboard themes.",
        )
        _render_theme_styles(st.session_state["dark_mode"])
        st.header("Upload datasets")
        st.caption(
            "CSV, XLSX, and XLS are supported. Uploaded data applies only to this "
            "session and is not saved to the database."
        )
        inventory_file = st.file_uploader(
            "Inventory dataset",
            type=["csv", "xlsx", "xls"],
            key="inventory_dataset_upload",
            help=(
                "Required columns: SKU, on_hand, reorder_point. Optional: name, "
                "category, average_daily_usage, target_stock, supplier_lead_time_days, "
                "warehouse_latitude, warehouse_longitude."
            ),
        )
        product_file = st.file_uploader(
            "Product dataset",
            type=["csv", "xlsx", "xls"],
            key="product_dataset_upload",
            help=(
                "Required columns: SKU and name. Optional inventory columns enrich "
                "matching SKUs; new products default missing stock values to zero. "
                "Optional category labels products; warehouse_latitude and "
                "warehouse_longitude map stock locations."
            ),
        )
        supplier_file = st.file_uploader(
            "Supplier dataset",
            type=["csv", "xlsx", "xls"],
            key="supplier_dataset_upload",
            help=(
                "Required columns: SKU, supplier, unit_price. Optional: currency, "
                "lead_time_days, minimum_order_quantity, discount threshold and percentage, "
                "supplier_latitude, supplier_longitude."
            ),
        )
        inventory_upload = _parse_upload(
            inventory_file, "Inventory", parse_inventory_table
        )
        product_upload = _parse_upload(product_file, "Product", parse_product_table)
        supplier_upload = _parse_upload(
            supplier_file, "Supplier", parse_supplier_table
        )
        if inventory_file is not None and inventory_upload is not None:
            st.success(f"Using {len(inventory_upload)} uploaded inventory rows.")
        if product_file is not None and product_upload is not None:
            st.success(f"Using {len(product_upload)} uploaded product rows.")
        if supplier_file is not None and supplier_upload is not None:
            st.success(f"Using {len(supplier_upload)} uploaded supplier offers.")

        st.header("Business limits")
        with st.form("business_limits"):
            cash_balance = st.number_input(
                "Cash balance", min_value=0.0,
                value=float(database.get_setting("cash_balance", "1500")), step=100.0,
            )
            cash_reserve = st.number_input(
                "Protected reserve", min_value=0.0,
                value=float(database.get_setting("minimum_cash_reserve", "500")), step=100.0,
            )
            approver = st.text_input("Approver", value="Business owner")
            if st.form_submit_button("Save limits"):
                database.set_setting("cash_balance", cash_balance)
                database.set_setting("minimum_cash_reserve", cash_reserve)
                st.rerun()
        st.header("Scenario controls")
        lead_time_multiplier_pct = st.slider(
            "Lead time multiplier (%)",
            min_value=50,
            max_value=200,
            value=100,
            step=5,
            help="100% uses each supplier's current lead time.",
        )
        demand_surge_pct = st.slider(
            "Demand surge (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="Increases forecast demand uniformly across the simulated lead time.",
        )
        holding_cost_adjustment_pct = st.slider(
            "Holding cost adjustment (%)",
            min_value=-100,
            max_value=100,
            value=0,
            step=5,
            help="Adjusts the baseline annual carrying-cost rate proportionally.",
        )
        baseline_carrying_rate_pct = st.slider(
            "Baseline annual carrying rate (%)",
            min_value=0,
            max_value=50,
            value=20,
            step=1,
            help="Starting annual carrying cost as a percentage of inventory value.",
        )

    all_inventory = merge_inventory_uploads(
        database.list_inventory(),
        inventory_upload,
        product_upload,
    )
    all_documents = merge_supplier_documents(
        database.list_documents(),
        supplier_upload,
        supplier_file.name if supplier_file is not None else "uploaded-suppliers",
    )
    all_orders = database.list_orders()
    all_pending_orders = database.list_pending_purchase_orders()
    all_financial_kpis = database.list_financial_kpis()
    all_fulfillment_records = database.list_fulfillment_records()
    all_supplier_deliveries = database.list_supplier_deliveries()
    all_landed_cost_records = database.list_landed_cost_records()
    all_usage_history = {
        item.sku: database.get_usage_history(item.sku) for item in all_inventory
    }
    supplier_locations = {
        location
        for document in all_documents
        if (location := _supplier_location(document)) is not None
    }
    location_by_label = {
        _location_label(location): location for location in sorted(supplier_locations)
    }
    location_options = list(location_by_label)
    category_options = sorted(
        {(item.category or "").strip() or "Unclassified" for item in all_inventory}
    )

    order_dates = [
        record_date
        for order in all_orders
        if (record_date := _record_date(order.get("created_at"))) is not None
    ] + [
        record_date
        for order in all_pending_orders
        if (record_date := _record_date(order.get("created_at"))) is not None
    ]
    available_dates = order_dates + [
        record_date
        for value in (
            [record.period_end for record in all_financial_kpis]
            + [record.recorded_at for record in all_fulfillment_records]
            + [record.promised_date for record in all_supplier_deliveries]
            + [record.received_at for record in all_landed_cost_records]
            + [
                usage_date
                for history in all_usage_history.values()
                for usage_date, _ in history
            ]
        )
        if (record_date := _record_date(value)) is not None
    ]
    if not available_dates:
        available_dates = [date.today()]
    date_bounds = (min(available_dates), max(available_dates))
    with st.sidebar:
        st.divider()
        st.header("Global filters")
        selected_date_range = st.slider(
            "Order and reporting date range",
            min_value=date_bounds[0],
            max_value=date_bounds[1],
            value=date_bounds,
            format="YYYY-MM-DD",
            help=(
                "Filters orders and date-stamped fulfillment, delivery, usage, "
                "financial, and landed-cost records. Available dates include all "
                "records so the unfiltered dashboard shows the complete data range."
            ),
        )
        selected_categories = st.multiselect(
            "Product categories",
            options=category_options,
            default=category_options,
            help="Choose which product categories appear across the dashboard.",
        )
        selected_locations = st.multiselect(
            "Supplier locations",
            options=location_options,
            default=location_options,
            help=(
                "Locations are supplier price-list coordinates. Products without an "
                "offer at a selected location are excluded when this filter is narrowed."
            ),
        )

    range_start, range_end = selected_date_range
    inventory, documents, selected_supplier_names = (
        _filter_inventory_and_documents(
            all_inventory,
            all_documents,
            set(selected_categories),
            set(selected_locations),
            location_by_label,
        )
    )
    location_filter_active = set(selected_locations) != set(location_options)
    inventory_skus = {item.sku for item in inventory}
    if not inventory:
        st.info("No products match the selected category and supplier-location filters.")

    fulfillment_records = tuple(
        record
        for record in all_fulfillment_records
        if record.sku in inventory_skus
        and range_start <= record.recorded_at <= range_end
    )
    supplier_deliveries = tuple(
        record
        for record in all_supplier_deliveries
        if record.sku in inventory_skus
        and range_start <= record.promised_date <= range_end
    )
    financial_kpis = tuple(
        record
        for record in all_financial_kpis
        if range_start <= record.period_end <= range_end
    )
    landed_cost_records = tuple(
        record
        for record in all_landed_cost_records
        if record.sku in inventory_skus
        and range_start <= record.received_at <= range_end
    )
    orders = tuple(
        order
        for order in all_orders
        if (
            (order_date := _record_date(order.get("created_at"))) is not None
            and range_start <= order_date <= range_end
            and str(order["sku"]) in inventory_skus
            and (
                not location_filter_active
                or order["supplier"] in selected_supplier_names
            )
        )
    )
    auto_purchase_order_count = _create_auto_purchase_orders(
        database,
        inventory,
        documents,
    )
    pending_orders = tuple(
        order
        for order in database.list_pending_purchase_orders()
        if (
            (created_date := _record_date(order.get("created_at"))) is not None
            and range_start <= created_date <= range_end
            and str(order["sku"]) in inventory_skus
            and (
                not location_filter_active
                or str(order["supplier"]) in selected_supplier_names
            )
        )
    )
    usage_history = {
        item.sku: tuple(
            (usage_date, units)
            for usage_date, units in all_usage_history[item.sku]
            if range_start <= date.fromisoformat(usage_date[:10]) <= range_end
        )
        for item in inventory
    }

    document_store = LocalDocumentStore()
    for document in documents:
        document_store.add(document)
    assistant = SupplyChainAssistant(document_store)
    result = assistant.run(
        inventory,
        cash_balance=float(database.get_setting("cash_balance", "1500")),
        minimum_cash_reserve=float(database.get_setting("minimum_cash_reserve", "500")),
        usage_history=usage_history,
    )
    cost_report = CostOptimizationRiskAgent().analyze(
        inventory,
        result.forecasts,
        result.recommendations,
        documents,
        usage_history,
        financial_kpis,
        fulfillment_records,
        supplier_deliveries,
        landed_cost_records,
    )
    diversification_report = ProductDiversificationAgent().analyze(
        inventory,
        fulfillment_records,
        supplier_deliveries,
        financial_kpis,
    )
    _render_executive_kpis(
        inventory,
        result.recommendations,
        landed_cost_records,
        diversification_report.overall_supplier_otif,
        diversification_report.overall_fill_rate,
        result.financial_assessment.currency,
    )
    active_risks = detect_supply_chain_risks(
        inventory,
        orders,
        supplier_deliveries,
        landed_cost_records,
        documents,
    )
    _render_risk_radar(active_risks)

    (
        overview_tab,
        purchasing_tab,
        cost_optimization_tab,
        scenario_tab,
        map_tab,
        diversification_tab,
        inventory_tab,
        supplier_scorecard_tab,
        contacts_tab,
        order_log_tab,
    ) = st.tabs(
        [
            "Overview",
            "Purchase review",
            "Cost & risk analysis",
            "What-if simulation",
            "Geographic map",
            "Product diversification",
            "Inventory & usage",
            "Supplier scorecard",
            "Suppliers",
            "Order log",
        ]
    )
    with overview_tab:
        metrics = st.columns(4)
        metrics[0].metric("Products", len(inventory))
        metrics[1].metric("Low stock", len(result.low_stock))
        metrics[2].metric("Forecasts", sum(f.proactive_order for f in result.forecasts))
        metrics[3].metric(
            "Est. spend",
            _money(result.financial_assessment.requested_amount, result.financial_assessment.currency),
        )
        st.subheader("Stock position")
        stock_rows = [
            {
                "SKU": item.sku,
                "Product": item.name,
                "On hand": item.on_hand,
                "Reorder point": item.reorder_point,
                "Forecast at lead time": next(
                    forecast.projected_stock_at_lead_time
                    for forecast in result.forecasts if forecast.sku == item.sku
                ),
                "Status": "Reorder" if item.on_hand <= item.reorder_point else "Monitoring",
            }
            for item in inventory
        ]
        st.dataframe(stock_rows, width="stretch", hide_index=True)
        st.subheader("Usage trend forecast")
        forecast_rows = [
            {
                "SKU": forecast.sku,
                "Daily trend": forecast.daily_usage_slope,
                "Forecast daily use": forecast.forecast_daily_usage,
                "Stock at lead time": forecast.projected_stock_at_lead_time,
                "Days to reorder": forecast.days_until_reorder_point,
                "Proactive order": "Yes" if forecast.proactive_order else "No",
            }
            for forecast in result.forecasts
        ]
        st.dataframe(forecast_rows, width="stretch", hide_index=True)

    with purchasing_tab:
        st.subheader("Automated replenishment")
        if auto_purchase_order_count:
            st.success(
                f"Generated {auto_purchase_order_count} pending Auto-PO draft(s) "
                "for products at or below their configured reorder point. "
                "No orders were approved or sent."
            )
        else:
            st.caption(
                "Auto-PO creates an idempotent pending draft when stock is at or "
                "below the configured reorder point and a supplier quote is available."
            )
        _render_pending_purchase_orders(pending_orders)
        _render_manual_purchase_order_form(database, inventory, documents)
        st.metric(
            "Available after reserve",
            _money(result.financial_assessment.spendable_cash, result.financial_assessment.currency),
        )
        _render_purchase_review(
            database,
            result.recommendations,
            result.purchase_drafts,
            result.financial_assessment.approved,
            result.financial_assessment.reason,
            approver,
        )

    with cost_optimization_tab:
        _render_cost_optimization(cost_report)

    with scenario_tab:
        _render_scenario_simulation(
            inventory,
            result.forecasts,
            _inventory_unit_costs(result.recommendations, landed_cost_records),
            lead_time_multiplier_pct,
            demand_surge_pct,
            holding_cost_adjustment_pct,
            baseline_carrying_rate_pct,
        )

    with map_tab:
        _render_supply_chain_map(
            inventory,
            result.forecasts,
            result.recommendations,
            documents,
        )

    with diversification_tab:
        _render_diversification_analysis(diversification_report)

    with inventory_tab:
        _render_inventory_tab(
            database,
            inventory,
            read_only=inventory_upload is not None or product_upload is not None,
        )

    with supplier_scorecard_tab:
        _render_supplier_scorecard(
            calculate_supplier_scores(
                supplier_deliveries,
                documents,
                landed_cost_records,
            )
        )

    with contacts_tab:
        _render_contacts(database, selected_supplier_names)

    with order_log_tab:
        _render_pending_purchase_orders(pending_orders)
        if orders:
            st.dataframe(
                [
                    {
                        "Order": order["order_id"],
                        "Date": order["created_at"],
                        "SKU": order["sku"],
                        "Supplier": order["supplier"],
                        "Recipient": order["recipient"],
                        "Total": _money(order["total_cost"], order["currency"]),
                        "Record": "Sample / demo" if order["is_demo"] else "Operational",
                        "Status": (
                            f"{order['status'].title()} (sample)"
                            if order["is_demo"] else order["status"].title()
                        ),
                        "Approved by": order["approved_by"],
                        "Error": order["error"] or "",
                    }
                    for order in orders
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No approved or sent orders have been logged.")

    if "supply_chain_chat_dispatcher" not in st.session_state:
        st.session_state["supply_chain_chat_dispatcher"] = (
            MultiAgentChatDispatcher()
        )
    _render_floating_chat(
        st.session_state["supply_chain_chat_dispatcher"],
        inventory,
        result,
        cost_report,
        database,
        documents,
    )


if __name__ == "__main__":
    main()