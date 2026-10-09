"""Build supplier, warehouse-risk, and shipping-route map records."""

from __future__ import annotations

from .models import Document, InventoryItem, StockForecast, SupplierRecommendation


def build_supply_chain_map_data(
    inventory: tuple[InventoryItem, ...],
    forecasts: tuple[StockForecast, ...],
    recommendations: tuple[SupplierRecommendation, ...],
    documents: tuple[Document, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Return supplier points, warehouse risk points, and recommendation routes."""
    forecast_by_sku = {forecast.sku: forecast for forecast in forecasts}
    supplier_points_by_location: dict[
        tuple[str, float, float], dict[str, object]
    ] = {}
    for document in documents:
        metadata = document.metadata
        latitude = metadata.get("latitude")
        longitude = metadata.get("longitude")
        supplier = metadata.get("supplier")
        if (
            document.category != "price_list"
            or supplier is None
            or latitude is None
            or longitude is None
        ):
            continue
        try:
            latitude_value = float(latitude)
            longitude_value = float(longitude)
        except (TypeError, ValueError):
            continue
        if not (
            -90 <= latitude_value <= 90
            and -180 <= longitude_value <= 180
        ):
            continue
        location_key = (str(supplier), latitude_value, longitude_value)
        point = supplier_points_by_location.setdefault(
            location_key,
            {
                "name": str(supplier),
                "latitude": latitude_value,
                "longitude": longitude_value,
                "skus": set(),
            },
        )
        sku = metadata.get("sku")
        if sku is not None:
            point["skus"].add(str(sku))
    supplier_points = [
        {
            **point,
            "skus": ", ".join(sorted(point["skus"])),
        }
        for point in supplier_points_by_location.values()
    ]

    warehouse_points: list[dict[str, object]] = []
    warehouses_by_sku: dict[str, dict[str, object]] = {}
    for item in inventory:
        latitude = item.warehouse_latitude
        longitude = item.warehouse_longitude
        if latitude is None or longitude is None:
            continue
        forecast = forecast_by_sku.get(item.sku)
        projected_stock = (
            forecast.projected_stock_at_lead_time if forecast is not None else item.on_hand
        )
        if projected_stock <= item.reorder_point:
            risk = "High"
        elif projected_stock <= item.reorder_point * 1.25:
            risk = "Elevated"
        else:
            risk = "Normal"
        warehouse_point: dict[str, object] = {
            "name": f"{item.name} ({item.sku})",
            "latitude": latitude,
            "longitude": longitude,
            "risk": risk,
            "on_hand": item.on_hand,
            "projected_stock": round(projected_stock, 2),
            "reorder_point": item.reorder_point,
        }
        warehouse_points.append(warehouse_point)
        warehouses_by_sku[item.sku] = warehouse_point

    routes: list[dict[str, object]] = []
    for recommendation in recommendations:
        offer = recommendation.offer
        warehouse = warehouses_by_sku.get(recommendation.need.item.sku)
        if (
            warehouse is None
            or offer.latitude is None
            or offer.longitude is None
        ):
            continue
        routes.append(
            {
                "name": f"{offer.supplier} to {recommendation.need.item.name}",
                "sku": recommendation.need.item.sku,
                "supplier": offer.supplier,
                "quantity": recommendation.order_quantity,
                "source": [offer.longitude, offer.latitude],
                "target": [
                    float(warehouse["longitude"]),
                    float(warehouse["latitude"]),
                ],
            }
        )

    return supplier_points, warehouse_points, routes
