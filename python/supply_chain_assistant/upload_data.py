"""Validation and session-only integration of uploaded supply-chain tables."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from io import BytesIO
from math import isfinite
from typing import Any

import pandas as pd

from .models import Document, InventoryItem


_ALIASES = {
    "sku": {"sku", "product_id", "item_id", "product_code", "item_code"},
    "name": {"name", "product", "product_name", "item_name", "description"},
    "category": {"category", "product_category", "product_type"},
    "on_hand": {"on_hand", "quantity", "stock", "current_stock", "inventory"},
    "reorder_point": {"reorder_point", "reorder_level", "minimum_stock"},
    "average_daily_usage": {
        "average_daily_usage",
        "daily_usage",
        "average_daily_demand",
        "daily_demand",
    },
    "target_stock": {"target_stock", "maximum_stock", "desired_stock"},
    "supplier_lead_time_days": {
        "supplier_lead_time_days",
        "lead_time_days",
        "lead_time",
    },
    "warehouse_latitude": {"warehouse_latitude", "warehouse_lat"},
    "warehouse_longitude": {"warehouse_longitude", "warehouse_lon", "warehouse_lng"},
    "supplier_latitude": {"supplier_latitude", "supplier_lat"},
    "supplier_longitude": {"supplier_longitude", "supplier_lon", "supplier_lng"},
    "supplier": {"supplier", "supplier_name", "vendor", "vendor_name"},
    "unit_price": {"unit_price", "price", "cost", "unit_cost"},
    "currency": {"currency", "currency_code"},
    "minimum_order_quantity": {"minimum_order_quantity", "moq", "min_order_quantity"},
    "discount_threshold_quantity": {
        "discount_threshold_quantity",
        "discount_threshold",
    },
    "discount_percentage": {"discount_percentage", "discount_pct", "discount"},
}
_ALIAS_TO_FIELD = {
    re.sub(r"[^a-z0-9]+", "_", alias.lower()).strip("_"): field
    for field, aliases in _ALIASES.items()
    for alias in aliases
}


@dataclass(frozen=True)
class ProductUpload:
    sku: str
    name: str
    category: str | None = None
    on_hand: int | None = None
    reorder_point: int | None = None
    average_daily_usage: float | None = None
    target_stock: int | None = None
    supplier_lead_time_days: int | None = None
    warehouse_latitude: float | None = None
    warehouse_longitude: float | None = None


def read_uploaded_table(uploaded_file: Any) -> pd.DataFrame:
    """Read a CSV or first-sheet Excel upload using its filename extension."""
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
    file_bytes = uploaded_file.getvalue()
    if suffix == "csv":
        return pd.read_csv(BytesIO(file_bytes))
    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(BytesIO(file_bytes))
    raise ValueError("Upload a CSV (.csv), Excel (.xlsx), or Excel (.xls) file.")


def parse_inventory_table(frame: pd.DataFrame) -> tuple[InventoryItem, ...]:
    columns = _normalize_columns(
        frame,
        required={"sku", "on_hand", "reorder_point"},
        optional={
            "name",
            "category",
            "average_daily_usage",
            "target_stock",
            "supplier_lead_time_days",
            "warehouse_latitude",
            "warehouse_longitude",
        },
        aliases={
            "latitude": "warehouse_latitude",
            "lat": "warehouse_latitude",
            "longitude": "warehouse_longitude",
            "lon": "warehouse_longitude",
            "lng": "warehouse_longitude",
        },
        dataset_name="Inventory",
    )
    items: list[InventoryItem] = []
    seen_skus: set[str] = set()
    for row_number, row in enumerate(columns.to_dict(orient="records"), start=2):
        sku = _required_text(row, "sku", row_number, "Inventory")
        if sku in seen_skus:
            raise ValueError(f"Inventory has duplicate SKU {sku!r} at row {row_number}.")
        seen_skus.add(sku)
        on_hand = _required_integer(row, "on_hand", row_number)
        warehouse_latitude, warehouse_longitude = _coordinates(
            row, "warehouse_latitude", "warehouse_longitude", row_number
        )
        items.append(
            InventoryItem(
                sku=sku,
                name=_optional_text(row.get("name")) or sku,
                on_hand=on_hand,
                reorder_point=_required_integer(row, "reorder_point", row_number),
                average_daily_usage=_number(
                    row.get("average_daily_usage"), "average_daily_usage", row_number, 0.0
                ),
                target_stock=_integer(
                    row.get("target_stock"), "target_stock", row_number, on_hand
                ),
                supplier_lead_time_days=_integer(
                    row.get("supplier_lead_time_days"),
                    "supplier_lead_time_days",
                    row_number,
                    5,
                ),
                warehouse_latitude=warehouse_latitude,
                warehouse_longitude=warehouse_longitude,
                category=_optional_text(row.get("category")),
            )
        )
    return tuple(items)


def parse_product_table(frame: pd.DataFrame) -> tuple[ProductUpload, ...]:
    columns = _normalize_columns(
        frame,
        required={"sku", "name"},
        optional={
            "category",
            "on_hand",
            "reorder_point",
            "average_daily_usage",
            "target_stock",
            "supplier_lead_time_days",
            "warehouse_latitude",
            "warehouse_longitude",
        },
        aliases={
            "latitude": "warehouse_latitude",
            "lat": "warehouse_latitude",
            "longitude": "warehouse_longitude",
            "lon": "warehouse_longitude",
            "lng": "warehouse_longitude",
        },
        dataset_name="Product",
    )
    products: list[ProductUpload] = []
    seen_skus: set[str] = set()
    for row_number, row in enumerate(columns.to_dict(orient="records"), start=2):
        sku = _required_text(row, "sku", row_number, "Product")
        if sku in seen_skus:
            raise ValueError(f"Product data has duplicate SKU {sku!r} at row {row_number}.")
        seen_skus.add(sku)
        warehouse_latitude, warehouse_longitude = _coordinates(
            row, "warehouse_latitude", "warehouse_longitude", row_number
        )
        products.append(
            ProductUpload(
                sku=sku,
                name=_required_text(row, "name", row_number, "Product"),
                category=_optional_text(row.get("category")),
                on_hand=_optional_integer(row, "on_hand", row_number),
                reorder_point=_optional_integer(row, "reorder_point", row_number),
                average_daily_usage=_optional_number(
                    row, "average_daily_usage", row_number
                ),
                target_stock=_optional_integer(row, "target_stock", row_number),
                supplier_lead_time_days=_optional_integer(
                    row, "supplier_lead_time_days", row_number
                ),
                warehouse_latitude=warehouse_latitude,
                warehouse_longitude=warehouse_longitude,
            )
        )
    return tuple(products)


def parse_supplier_table(frame: pd.DataFrame) -> tuple[dict[str, object], ...]:
    columns = _normalize_columns(
        frame,
        required={"sku", "supplier", "unit_price"},
        optional={
            "currency",
            "supplier_lead_time_days",
            "minimum_order_quantity",
            "discount_threshold_quantity",
            "discount_percentage",
            "supplier_latitude",
            "supplier_longitude",
        },
        aliases={
            "latitude": "supplier_latitude",
            "lat": "supplier_latitude",
            "longitude": "supplier_longitude",
            "lon": "supplier_longitude",
            "lng": "supplier_longitude",
        },
        dataset_name="Supplier",
    )
    offers: list[dict[str, object]] = []
    for row_number, row in enumerate(columns.to_dict(orient="records"), start=2):
        discount_percentage = _number(
            row.get("discount_percentage"), "discount_percentage", row_number, 0.0
        )
        if discount_percentage > 100:
            raise ValueError(
                f"Supplier discount_percentage must be at most 100 (row {row_number})."
            )
        latitude, longitude = _coordinates(
            row, "supplier_latitude", "supplier_longitude", row_number
        )
        offers.append(
            {
                "sku": _required_text(row, "sku", row_number, "Supplier"),
                "supplier": _required_text(row, "supplier", row_number, "Supplier"),
                "unit_price": _required_number(row, "unit_price", row_number),
                "currency": _optional_text(row.get("currency")) or "USD",
                "lead_time_days": _integer(
                    row.get("supplier_lead_time_days"),
                    "lead_time_days",
                    row_number,
                    5,
                ),
                "minimum_order_quantity": _integer(
                    row.get("minimum_order_quantity"),
                    "minimum_order_quantity",
                    row_number,
                    1,
                ),
                "discount_threshold_quantity": _integer(
                    row.get("discount_threshold_quantity"),
                    "discount_threshold_quantity",
                    row_number,
                    0,
                ),
                "discount_percentage": discount_percentage,
                "latitude": latitude,
                "longitude": longitude,
            }
        )
    return tuple(offers)


def merge_inventory_uploads(
    current_inventory: tuple[InventoryItem, ...],
    inventory_upload: tuple[InventoryItem, ...] | None,
    product_upload: tuple[ProductUpload, ...] | None,
) -> tuple[InventoryItem, ...]:
    """Replace inventory rows when uploaded, then enrich/add product master rows."""
    items = {
        item.sku: item
        for item in (inventory_upload if inventory_upload is not None else current_inventory)
    }
    for product in product_upload or ():
        existing = items.get(product.sku)
        if existing is None:
            items[product.sku] = InventoryItem(
                sku=product.sku,
                name=product.name,
                on_hand=product.on_hand if product.on_hand is not None else 0,
                reorder_point=(
                    product.reorder_point if product.reorder_point is not None else 0
                ),
                average_daily_usage=(
                    product.average_daily_usage
                    if product.average_daily_usage is not None
                    else 0.0
                ),
                target_stock=(
                    product.target_stock if product.target_stock is not None else 0
                ),
                supplier_lead_time_days=(
                    product.supplier_lead_time_days
                    if product.supplier_lead_time_days is not None
                    else 5
                ),
                warehouse_latitude=product.warehouse_latitude,
                warehouse_longitude=product.warehouse_longitude,
                category=product.category,
            )
        else:
            items[product.sku] = replace(
                existing,
                name=product.name,
                on_hand=(
                    product.on_hand if product.on_hand is not None else existing.on_hand
                ),
                reorder_point=(
                    product.reorder_point
                    if product.reorder_point is not None
                    else existing.reorder_point
                ),
                average_daily_usage=(
                    product.average_daily_usage
                    if product.average_daily_usage is not None
                    else existing.average_daily_usage
                ),
                target_stock=(
                    product.target_stock
                    if product.target_stock is not None
                    else existing.target_stock
                ),
                supplier_lead_time_days=(
                    product.supplier_lead_time_days
                    if product.supplier_lead_time_days is not None
                    else existing.supplier_lead_time_days
                ),
                warehouse_latitude=(
                    product.warehouse_latitude
                    if product.warehouse_latitude is not None
                    else existing.warehouse_latitude
                ),
                warehouse_longitude=(
                    product.warehouse_longitude
                    if product.warehouse_longitude is not None
                    else existing.warehouse_longitude
                ),
                category=(
                    product.category
                    if product.category is not None
                    else existing.category
                ),
            )
    return tuple(items[sku] for sku in sorted(items))


def merge_supplier_documents(
    documents: tuple[Document, ...],
    offers: tuple[dict[str, object], ...] | None,
    source_name: str,
) -> tuple[Document, ...]:
    """Override existing price/contract evidence for uploaded SKUs, in memory only."""
    if not offers:
        return documents
    uploaded_skus = {str(offer["sku"]) for offer in offers}
    merged = [
        document
        for document in documents
        if not (
            document.category in {"price_list", "supplier_contract"}
            and str(document.metadata.get("sku", "")) in uploaded_skus
        )
    ]
    for index, offer in enumerate(offers):
        supplier = str(offer["supplier"])
        sku = str(offer["sku"])
        price = float(offer["unit_price"])
        currency = str(offer["currency"])
        lead_time = int(offer["lead_time_days"])
        minimum_quantity = int(offer["minimum_order_quantity"])
        threshold = int(offer["discount_threshold_quantity"])
        discount = float(offer["discount_percentage"])
        merged.append(
            Document(
                document_id=f"upload:{source_name}:{index}:{sku}:{supplier}",
                source=source_name,
                category="price_list",
                content=(
                    f"{supplier} price for {sku}: {currency} {price:.4f} per unit; "
                    f"lead time {lead_time} days; minimum order {minimum_quantity} units."
                ),
                metadata={
                    "sku": sku,
                    "supplier": supplier,
                    "unit_price": price,
                    "currency": currency,
                    "lead_time_days": lead_time,
                    "minimum_order_quantity": minimum_quantity,
                    "discount_threshold_quantity": threshold,
                    "discount_percentage": discount,
                    "latitude": offer["latitude"],
                    "longitude": offer["longitude"],
                    "source": source_name,
                },
            )
        )
    return tuple(merged)


def _normalize_columns(
    frame: pd.DataFrame,
    *,
    required: set[str],
    optional: set[str],
    aliases: dict[str, str] | None = None,
    dataset_name: str,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"{dataset_name} file contains no data rows.")
    normalized_names: dict[str, str] = {}
    for column in frame.columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        canonical = (aliases or {}).get(
            normalized,
            _ALIAS_TO_FIELD.get(normalized),
        )
        if canonical in required | optional:
            if canonical in normalized_names.values():
                raise ValueError(
                    f"{dataset_name} file has multiple columns for {canonical!r}."
                )
            normalized_names[str(column)] = canonical
    missing = required - set(normalized_names.values())
    if missing:
        raise ValueError(
            f"{dataset_name} file is missing required columns: "
            f"{', '.join(sorted(missing))}."
        )
    return frame.loc[:, list(normalized_names)].rename(
        columns=normalized_names
    ).rename_axis(None, axis="columns")


def _required_text(
    row: dict[str, Any], field: str, row_number: int, dataset_name: str
) -> str:
    value = _optional_text(row.get(field))
    if value is None:
        raise ValueError(
            f"{dataset_name} {field} is required and cannot be blank (row {row_number})."
        )
    return value


def _required_number(row: dict[str, Any], field: str, row_number: int) -> float:
    if pd.isna(row.get(field)):
        raise ValueError(f"{field} is required and cannot be blank (row {row_number}).")
    return _number(row[field], field, row_number, 0.0)


def _required_integer(row: dict[str, Any], field: str, row_number: int) -> int:
    result = _required_number(row, field, row_number)
    if not result.is_integer():
        raise ValueError(f"{field} must be a whole number (row {row_number}).")
    return int(result)


def _optional_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _number(value: Any, field: str, row_number: int, default: float) -> float:
    if pd.isna(value):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field} must be numeric and non-negative (row {row_number})."
        ) from error
    if not isfinite(result) or result < 0:
        raise ValueError(
            f"{field} must be numeric and non-negative (row {row_number})."
        )
    return result


def _integer(value: Any, field: str, row_number: int, default: int) -> int:
    result = _number(value, field, row_number, float(default))
    if not result.is_integer():
        raise ValueError(f"{field} must be a whole number (row {row_number}).")
    return int(result)


def _optional_number(
    row: dict[str, Any], field: str, row_number: int
) -> float | None:
    if pd.isna(row.get(field)):
        return None
    return _number(row[field], field, row_number, 0.0)


def _optional_integer(
    row: dict[str, Any], field: str, row_number: int
) -> int | None:
    if pd.isna(row.get(field)):
        return None
    return _integer(row[field], field, row_number, 0)


def _coordinates(
    row: dict[str, Any],
    latitude_field: str,
    longitude_field: str,
    row_number: int,
) -> tuple[float | None, float | None]:
    latitude_value = row.get(latitude_field)
    longitude_value = row.get(longitude_field)
    if pd.isna(latitude_value) and pd.isna(longitude_value):
        return None, None
    if pd.isna(latitude_value) or pd.isna(longitude_value):
        raise ValueError(
            f"{latitude_field} and {longitude_field} must both be provided "
            f"(row {row_number})."
        )
    try:
        latitude = float(latitude_value)
        longitude = float(longitude_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Coordinates must be numeric (row {row_number})."
        ) from error
    if not isfinite(latitude) or not isfinite(longitude):
        raise ValueError(f"Coordinates must be finite numbers (row {row_number}).")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError(
            f"Coordinates must be within latitude [-90, 90] and longitude [-180, 180] "
            f"(row {row_number})."
        )
    return latitude, longitude
