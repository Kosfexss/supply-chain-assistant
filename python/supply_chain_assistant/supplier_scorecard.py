"""Supplier performance scoring with explicit metric coverage."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from statistics import pstdev

from .models import Document, LandedCostRecord, SupplierDeliveryRecord


@dataclass(frozen=True)
class SupplierScore:
    supplier: str
    score: float
    tier: str
    otif_rate: float | None
    lead_time_consistency: float | None
    cost_competitiveness: float | None
    delivery_count: int
    price_comparison_count: int
    observed_components: int


_WEIGHTS = {
    "otif_rate": 0.50,
    "lead_time_consistency": 0.25,
    "cost_competitiveness": 0.25,
}


def _tier(score: float) -> str:
    if score >= 80:
        return "Tier A"
    if score >= 60:
        return "Tier B"
    return "Tier C"


def _comparable_supplier_costs(
    supplier_documents: tuple[Document, ...],
    landed_cost_records: tuple[LandedCostRecord, ...],
) -> dict[tuple[str, str, str], float]:
    costs: dict[tuple[str, str, str], float] = {}
    received_costs: dict[tuple[str, str, str], tuple[object, float]] = {}

    for record in landed_cost_records:
        if record.quantity <= 0:
            continue
        key = (record.supplier, record.sku, record.currency.upper())
        landed_unit_cost = record.unit_price + (
            record.freight_cost + record.duties + record.handling_cost
        ) / record.quantity
        current = received_costs.get(key)
        if current is None or record.received_at > current[0]:
            received_costs[key] = (record.received_at, landed_unit_cost)

    costs.update(
        {key: value[1] for key, value in received_costs.items()}
    )
    for document in supplier_documents:
        if document.category != "price_list":
            continue
        metadata = document.metadata
        supplier = metadata.get("supplier")
        sku = metadata.get("sku")
        raw_cost = metadata.get("unit_price")
        if supplier is None or sku is None or raw_cost is None:
            continue
        try:
            cost = float(raw_cost)
        except (TypeError, ValueError):
            continue
        if cost <= 0:
            continue
        currency = str(metadata.get("currency", "USD")).upper()
        key = (str(supplier), str(sku), currency)
        costs.setdefault(key, cost)
    return costs


def calculate_supplier_scores(
    supplier_deliveries: tuple[SupplierDeliveryRecord, ...],
    supplier_documents: tuple[Document, ...],
    landed_cost_records: tuple[LandedCostRecord, ...] = (),
    *,
    today: date | None = None,
    consistency_window_days: float = 10.0,
) -> tuple[SupplierScore, ...]:
    """Score suppliers on OTIF, delivery-date consistency, and comparable cost.

    Scores use only available components and renormalize their published weights.
    Delivery-date consistency uses the population standard deviation of actual
    minus promised days; lower variation scores better.
    """
    if consistency_window_days <= 0:
        raise ValueError("Consistency window must be greater than zero.")
    current_date = today or date.today()

    deliveries_by_supplier: dict[str, list[SupplierDeliveryRecord]] = defaultdict(list)
    for delivery in supplier_deliveries:
        deliveries_by_supplier[delivery.supplier].append(delivery)

    supplier_costs = _comparable_supplier_costs(
        supplier_documents,
        landed_cost_records,
    )
    costs_by_item_currency: dict[tuple[str, str], list[float]] = defaultdict(list)
    costs_by_supplier: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for (supplier, sku, currency), cost in supplier_costs.items():
        item_currency = (sku, currency)
        costs_by_item_currency[item_currency].append(cost)
        costs_by_supplier[supplier][item_currency] = cost

    suppliers = (
        set(deliveries_by_supplier)
        | set(costs_by_supplier)
    )
    scores: list[SupplierScore] = []
    for supplier in suppliers:
        deliveries = deliveries_by_supplier.get(supplier, [])
        due_deliveries = [
            delivery
            for delivery in deliveries
            if delivery.promised_date <= current_date
        ]
        otif_rate = None
        lead_time_consistency = None
        if due_deliveries:
            otif_rate = sum(
                delivery.actual_date is not None
                and delivery.actual_date <= delivery.promised_date
                and delivery.delivered_quantity >= delivery.promised_quantity
                for delivery in due_deliveries
            ) / len(due_deliveries)
            lateness_days = [
                (delivery.actual_date - delivery.promised_date).days
                for delivery in due_deliveries
                if delivery.actual_date is not None
            ]
            if lateness_days:
                variability = pstdev(lateness_days)
                lead_time_consistency = max(
                    0.0,
                    100.0 * (1.0 - variability / consistency_window_days),
                )

        supplier_item_costs = costs_by_supplier.get(supplier, {})
        relative_cost_scores = [
            100.0 * min(cost, competitor_costs) / cost
            for item_currency, cost in supplier_item_costs.items()
            if len(costs_by_item_currency[item_currency]) > 1
            if (
                competitor_costs := min(costs_by_item_currency[item_currency],
                                        default=cost)
            ) > 0
        ]
        cost_competitiveness = (
            sum(relative_cost_scores) / len(relative_cost_scores)
            if relative_cost_scores
            else None
        )
        components = {
            "otif_rate": otif_rate * 100 if otif_rate is not None else None,
            "lead_time_consistency": lead_time_consistency,
            "cost_competitiveness": cost_competitiveness,
        }
        observed = [
            (name, value)
            for name, value in components.items()
            if value is not None
        ]
        if not observed:
            continue
        available_weight = sum(_WEIGHTS[name] for name, _ in observed)
        score = sum(
            value * _WEIGHTS[name] for name, value in observed
        ) / available_weight
        scores.append(
            SupplierScore(
                supplier=supplier,
                score=score,
                tier=_tier(score),
                otif_rate=otif_rate,
                lead_time_consistency=lead_time_consistency,
                cost_competitiveness=cost_competitiveness,
                delivery_count=len(due_deliveries),
                price_comparison_count=len(relative_cost_scores),
                observed_components=len(observed),
            )
        )
    return tuple(sorted(scores, key=lambda item: (-item.score, item.supplier.casefold())))
