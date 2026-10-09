"""Deterministic what-if projections for inventory and carrying-cost scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Mapping

from .models import InventoryItem, StockForecast


@dataclass(frozen=True)
class ScenarioProjection:
    sku: str
    product_name: str
    adjusted_lead_time_days: int
    projected_inventory: float
    demand_during_lead_time: float
    stockout_units: float
    stockout_risk: bool
    days_until_stockout: float | None
    risk_level: str
    unit_cost: float | None
    currency: str | None
    stockout_cost_exposure: float | None
    baseline_annual_holding_cost: float | None
    scenario_annual_holding_cost: float | None
    annual_holding_cost_change: float | None


def simulate_inventory(
    inventory: tuple[InventoryItem, ...],
    forecasts: tuple[StockForecast, ...],
    unit_costs: Mapping[str, tuple[float, str]],
    lead_time_multiplier_pct: float,
    demand_surge_pct: float,
    holding_cost_adjustment_pct: float,
    baseline_carrying_rate_pct: float,
) -> tuple[ScenarioProjection, ...]:
    """Project inventory through adjusted supplier lead time and estimate cost impact."""
    if lead_time_multiplier_pct < 0:
        raise ValueError("Lead-time multiplier cannot be negative.")
    if demand_surge_pct < 0:
        raise ValueError("Demand surge cannot be negative.")
    if holding_cost_adjustment_pct < -100:
        raise ValueError("Holding-cost adjustment cannot reduce costs by more than 100%.")
    if baseline_carrying_rate_pct < 0:
        raise ValueError("Baseline carrying-cost rate cannot be negative.")

    forecast_by_sku = {forecast.sku: forecast for forecast in forecasts}
    demand_multiplier = 1 + demand_surge_pct / 100
    scenario_carrying_rate = (
        baseline_carrying_rate_pct
        * (1 + holding_cost_adjustment_pct / 100)
        / 100
    )
    baseline_carrying_rate = baseline_carrying_rate_pct / 100
    projections: list[ScenarioProjection] = []

    for item in inventory:
        forecast = forecast_by_sku.get(item.sku)
        if forecast is None:
            raise ValueError(f"Missing forecast for inventory SKU {item.sku}.")

        lead_time_days = ceil(
            item.supplier_lead_time_days * lead_time_multiplier_pct / 100
        )
        running_demand = 0.0
        days_until_stockout: float | None = None
        for day_offset in range(lead_time_days):
            daily_demand = max(
                0.0,
                forecast.forecast_daily_usage
                + forecast.daily_usage_slope * day_offset,
            ) * demand_multiplier
            previous_demand = running_demand
            running_demand += daily_demand
            if (
                days_until_stockout is None
                and daily_demand > 0
                and running_demand >= item.on_hand
            ):
                days_until_stockout = (
                    day_offset
                    + max(0.0, item.on_hand - previous_demand) / daily_demand
                )

        stockout_units = max(0.0, running_demand - item.on_hand)
        stockout_risk = running_demand >= item.on_hand and running_demand > 0
        projected_inventory = max(0.0, item.on_hand - running_demand)
        risk_level = (
            "Stockout"
            if stockout_risk
            else "At/below reorder point"
            if projected_inventory <= item.reorder_point
            else "Covered"
        )

        unit_cost_entry = unit_costs.get(item.sku)
        if unit_cost_entry is None:
            unit_cost = None
            currency = None
            stockout_cost_exposure = None
            baseline_holding_cost = None
            scenario_holding_cost = None
            holding_cost_change = None
        else:
            unit_cost, currency = unit_cost_entry
            stockout_cost_exposure = round(stockout_units * unit_cost, 2)
            baseline_holding_cost = round(
                item.on_hand * unit_cost * baseline_carrying_rate, 2
            )
            scenario_holding_cost = round(
                projected_inventory * unit_cost * scenario_carrying_rate, 2
            )
            holding_cost_change = round(
                scenario_holding_cost - baseline_holding_cost, 2
            )

        projections.append(
            ScenarioProjection(
                sku=item.sku,
                product_name=item.name,
                adjusted_lead_time_days=lead_time_days,
                projected_inventory=round(projected_inventory, 2),
                demand_during_lead_time=round(running_demand, 2),
                stockout_units=round(stockout_units, 2),
                stockout_risk=stockout_risk,
                days_until_stockout=(
                    round(days_until_stockout, 2)
                    if days_until_stockout is not None
                    else None
                ),
                risk_level=risk_level,
                unit_cost=unit_cost,
                currency=currency,
                stockout_cost_exposure=stockout_cost_exposure,
                baseline_annual_holding_cost=baseline_holding_cost,
                scenario_annual_holding_cost=scenario_holding_cost,
                annual_holding_cost_change=holding_cost_change,
            )
        )

    return tuple(projections)
