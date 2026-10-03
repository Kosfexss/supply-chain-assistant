"""Transparent linear-trend demand forecasts for proactive reorder signals."""

from __future__ import annotations

from .models import InventoryItem, StockForecast


class ForecastingAgent:
    def run(
        self,
        inventory: tuple[InventoryItem, ...],
        usage_history: dict[str, tuple[tuple[str, float], ...]],
    ) -> tuple[StockForecast, ...]:
        return tuple(
            self.forecast(item, usage_history.get(item.sku, ())) for item in inventory
        )

    def forecast(
        self,
        item: InventoryItem,
        history: tuple[tuple[str, float], ...],
    ) -> StockForecast:
        values = [max(0.0, float(units)) for _, units in history]
        slope = self._linear_slope(values)
        mean_usage = (
            sum(values) / len(values) if values else item.average_daily_usage
        )
        mean_index = (len(values) - 1) / 2 if values else 0.0
        intercept = mean_usage - slope * mean_index
        forecast_daily_usage = max(0.0, intercept + slope * len(values))
        lead_days = max(0, item.supplier_lead_time_days)
        projected_demand = sum(
            max(0.0, intercept + slope * (len(values) + offset))
            for offset in range(lead_days)
        )
        projected_stock = max(0.0, item.on_hand - projected_demand)

        days_until_reorder: float | None = None
        running_stock = float(item.on_hand)
        for day in range(1, 366):
            predicted_daily = max(
                0.0, intercept + slope * (len(values) + day - 1)
            )
            running_stock -= predicted_daily
            if running_stock <= item.reorder_point:
                days_until_reorder = float(day)
                break

        proactive_order = (
            item.on_hand > item.reorder_point
            and projected_stock <= item.reorder_point
        )
        suggested_quantity = (
            max(0, item.target_stock - int(projected_stock))
            if proactive_order
            else 0
        )
        return StockForecast(
            sku=item.sku,
            product_name=item.name,
            observations=len(values),
            daily_usage_slope=round(slope, 4),
            forecast_daily_usage=round(forecast_daily_usage, 2),
            projected_stock_at_lead_time=round(projected_stock, 2),
            days_until_reorder_point=days_until_reorder,
            suggested_order_quantity=suggested_quantity,
            proactive_order=proactive_order,
        )

    @staticmethod
    def _linear_slope(values: list[float]) -> float:
        count = len(values)
        if count < 2:
            return 0.0
        mean_x = (count - 1) / 2
        mean_y = sum(values) / count
        denominator = sum((index - mean_x) ** 2 for index in range(count))
        if denominator == 0:
            return 0.0
        return sum(
            (index - mean_x) * (value - mean_y)
            for index, value in enumerate(values)
        ) / denominator