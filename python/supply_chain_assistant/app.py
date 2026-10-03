"""Streamlit dashboard for inventory, forecasts, sourcing, and approvals."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import streamlit as st

if __package__ in (None, ""):
    package_root = str(Path(__file__).resolve().parent.parent)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)

from supply_chain_assistant.database import Database
from supply_chain_assistant.email_dispatcher import EmailDispatcher, SMTPSettings
from supply_chain_assistant.models import InventoryItem, PurchaseDraft
from supply_chain_assistant.orchestrator import SupplyChainAssistant
from supply_chain_assistant.sample_data import seed_demo_data


def _money(value: float, currency: str = "USD") -> str:
    return f"{currency} {value:,.2f}"


def _render_inventory_tab(database: Database, inventory: tuple[InventoryItem, ...]) -> None:
    st.subheader("Inventory")
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


def _render_contacts(database: Database) -> None:
    st.subheader("Supplier contacts")
    contacts = database.list_supplier_contacts()
    for supplier, email in contacts.items():
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


def main() -> None:
    st.set_page_config(page_title="Supply Chain Assistant", page_icon="S", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
        .stApp { background: #f3f6f3; color: #1e2d27; }
        [data-testid="stHeader"] { background: rgba(243, 246, 243, 0.92); }
        [data-testid="stMetric"] { background: #ffffff; border: 1px solid #dce5df; padding: 14px 16px; border-radius: 6px; }
        [data-testid="stMetricLabel"] { color: #53675e; }
        h1, h2, h3 { color: #173d31; letter-spacing: 0; }
        div.stButton > button[kind="primary"], button[kind="primaryFormSubmit"] { background: #126b52; border-color: #126b52; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    database = Database()
    seed_demo_data(database)
    st.title("Supply Chain Assistant")

    with st.sidebar:
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

    inventory = database.list_inventory()
    usage_history = {
        item.sku: database.get_usage_history(item.sku) for item in inventory
    }
    assistant = SupplyChainAssistant.from_database(database)
    result = assistant.run(
        inventory,
        cash_balance=float(database.get_setting("cash_balance", "1500")),
        minimum_cash_reserve=float(database.get_setting("minimum_cash_reserve", "500")),
        usage_history=usage_history,
    )

    overview_tab, purchasing_tab, inventory_tab, contacts_tab, order_log_tab = st.tabs(
        ["Overview", "Purchase review", "Inventory & usage", "Suppliers", "Order log"]
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

    with inventory_tab:
        _render_inventory_tab(database, inventory)

    with contacts_tab:
        _render_contacts(database)

    with order_log_tab:
        orders = database.list_orders()
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
                        "Status": order["status"],
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


if __name__ == "__main__":
    main()