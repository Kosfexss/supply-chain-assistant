"""Rule-based anomaly detection for current supply-chain dashboard data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

from .models import (
    Document,
    InventoryItem,
    LandedCostRecord,
    SupplierDeliveryRecord,
)


@dataclass(frozen=True)
class SupplyChainRisk:
    severity: str
    category: str
    title: str
    description: str
    action: str


def detect_supply_chain_risks(
    inventory: tuple[InventoryItem, ...],
    orders: tuple[Mapping[str, object], ...],
    supplier_deliveries: tuple[SupplierDeliveryRecord, ...],
    landed_cost_records: tuple[LandedCostRecord, ...],
    supplier_documents: tuple[Document, ...],
    *,
    today: date | None = None,
    delay_warning_days: int = 3,
    delay_critical_days: int = 7,
    cost_variance_threshold: float = 0.20,
) -> tuple[SupplyChainRisk, ...]:
    """Return explainable risks; prices are compared only within matching currencies."""
    current_date = today or date.today()
    risks: list[SupplyChainRisk] = []

    for item in inventory:
        if item.on_hand <= 0:
            risks.append(
                SupplyChainRisk(
                    "Critical",
                    "Inventory",
                    f"{item.sku} is out of stock",
                    f"{item.name} has {item.on_hand} units on hand "
                    f"(reorder point: {item.reorder_point}).",
                    "Expedite replenishment and confirm the earliest available delivery.",
                )
            )
        elif item.on_hand <= item.reorder_point:
            risks.append(
                SupplyChainRisk(
                    "Warning",
                    "Inventory",
                    f"{item.sku} is below its reorder threshold",
                    f"{item.name} has {item.on_hand} units on hand, at or below "
                    f"the configured reorder point of {item.reorder_point}.",
                    "Review the replenishment recommendation and confirm stock coverage "
                    "through supplier lead time.",
                )
            )

    for delivery in supplier_deliveries:
        late_days = (
            (delivery.actual_date - delivery.promised_date).days
            if delivery.actual_date is not None
            else (
                (current_date - delivery.promised_date).days
                if delivery.promised_date < current_date
                else 0
            )
        )
        if late_days >= delay_warning_days:
            severity = "Critical" if late_days >= delay_critical_days else "Warning"
            delivery_status = (
                "has not arrived"
                if delivery.actual_date is None
                else f"arrived on {delivery.actual_date.isoformat()}"
            )
            risks.append(
                SupplyChainRisk(
                    severity,
                    "Supplier delay",
                    f"{delivery.supplier} delivery is {late_days} days late",
                    f"Delivery {delivery.delivery_id} for {delivery.sku} was promised "
                    f"for {delivery.promised_date.isoformat()} and {delivery_status}.",
                    "Contact the supplier for a confirmed delivery date and assess "
                    "alternate supply or inventory coverage.",
                )
            )

        if (
            delivery.actual_date is not None
            and delivery.delivered_quantity < delivery.promised_quantity
        ):
            shortage = delivery.promised_quantity - delivery.delivered_quantity
            risks.append(
                SupplyChainRisk(
                    "Critical" if delivery.delivered_quantity == 0 else "Warning",
                    "Supplier short shipment",
                    f"{delivery.supplier} delivered short for {delivery.sku}",
                    f"Delivery {delivery.delivery_id} was short "
                    f"{shortage} of {delivery.promised_quantity} promised units.",
                    "Reconcile the received quantity with the supplier and confirm "
                    "whether the balance is still in transit.",
                )
            )

    for order in orders:
        status = str(order.get("status", "")).casefold()
        error = str(order.get("error") or "").strip()
        if status == "failed" or error:
            risks.append(
                SupplyChainRisk(
                    "Critical",
                    "Order execution",
                    f"Order {order.get('order_id', 'unknown')} needs attention",
                    f"Order for {order.get('sku', 'unknown SKU')} has status "
                    f"{status or 'unknown'}."
                    + (f" Dispatch error: {error}" if error else ""),
                    "Resolve the dispatch or supplier issue, then verify the order "
                    "status before retrying.",
                )
            )

    current_prices: dict[tuple[str, str, str], float] = {}
    for document in supplier_documents:
        metadata = document.metadata
        if document.category != "price_list":
            continue
        sku = metadata.get("sku")
        supplier = metadata.get("supplier")
        unit_price = metadata.get("unit_price")
        if sku is None or supplier is None or unit_price is None:
            continue
        try:
            price = float(unit_price)
        except (TypeError, ValueError):
            continue
        if price <= 0:
            continue
        currency = str(metadata.get("currency", "USD")).upper()
        current_prices[(str(sku), str(supplier), currency)] = price

    for order in orders:
        try:
            sku = str(order["sku"])
            supplier = str(order["supplier"])
            order_price = float(order["unit_price"])
        except (KeyError, TypeError, ValueError):
            continue
        currency = str(order.get("currency", "USD")).upper()
        quoted_price = current_prices.get((sku, supplier, currency))
        if quoted_price is None or quoted_price <= 0:
            continue
        variance = order_price / quoted_price - 1
        if variance >= cost_variance_threshold:
            risks.append(
                SupplyChainRisk(
                    "Warning" if variance < 0.50 else "Critical",
                    "Cost variance",
                    f"{sku} order price is {variance:.0%} above the current quote",
                    f"Order {order.get('order_id', 'unknown')} records "
                    f"{currency} {order_price:,.2f}/unit from {supplier}; the current "
                    f"price list shows {currency} {quoted_price:,.2f}/unit.",
                    "Verify the invoice and negotiated terms with the supplier before "
                    "approving another purchase at this price.",
                )
            )

    landed_cost_history: dict[tuple[str, str], list[tuple[date, float, str]]] = {}
    for record in landed_cost_records:
        if record.quantity <= 0:
            continue
        unit_cost = record.unit_price + (
            record.freight_cost + record.duties + record.handling_cost
        ) / record.quantity
        landed_cost_history.setdefault(
            (record.sku, record.currency.upper()), []
        ).append((record.received_at, unit_cost, record.supplier))

    for (sku, currency), history in landed_cost_history.items():
        history.sort(key=lambda entry: entry[0])
        for previous, current in zip(history, history[1:]):
            previous_date, previous_cost, _ = previous
            current_date, current_cost, supplier = current
            if previous_cost <= 0:
                continue
            variance = current_cost / previous_cost - 1
            if variance >= cost_variance_threshold:
                risks.append(
                    SupplyChainRisk(
                        "Warning" if variance < 0.50 else "Critical",
                        "Landed-cost variance",
                        f"{sku} landed unit cost increased {variance:.0%}",
                        f"Cost from {supplier} rose from {currency} "
                        f"{previous_cost:,.2f} on {previous_date.isoformat()} to "
                        f"{currency} {current_cost:,.2f} on {current_date.isoformat()}.",
                        "Review the invoice components and compare alternate supplier "
                        "offers before the next replenishment.",
                    )
                )

    severity_order = {"Critical": 0, "Warning": 1}
    risks.sort(key=lambda risk: (severity_order[risk.severity], risk.category, risk.title))
    return tuple(risks)
