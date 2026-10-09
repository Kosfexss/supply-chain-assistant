"""Generate review-only replenishment drafts for below-threshold inventory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .models import Document, InventoryItem


@dataclass(frozen=True)
class AutoPurchaseDraft:
    sku: str
    supplier: str
    quantity: int
    unit_price: float
    currency: str
    expected_date: date


def build_auto_purchase_drafts(
    inventory: tuple[InventoryItem, ...],
    supplier_documents: tuple[Document, ...],
    *,
    today: date | None = None,
) -> tuple[AutoPurchaseDraft, ...]:
    """Draft the lowest-estimated-cost offer per SKU at/below its reorder threshold."""
    current_date = today or date.today()
    items_by_sku = {item.sku: item for item in inventory}
    offers_by_sku: dict[str, list[tuple[float, int, int, str, str]]] = {}
    for document in supplier_documents:
        if document.category != "price_list":
            continue
        metadata = document.metadata
        sku = metadata.get("sku")
        supplier = metadata.get("supplier")
        if sku is None or supplier is None or str(sku) not in items_by_sku:
            continue
        try:
            unit_price = float(metadata["unit_price"])
            lead_time_days = int(metadata.get("lead_time_days", 0))
            minimum_quantity = int(metadata.get("minimum_order_quantity", 1))
        except (KeyError, TypeError, ValueError):
            continue
        if unit_price < 0 or lead_time_days < 0 or minimum_quantity <= 0:
            continue
        offers_by_sku.setdefault(str(sku), []).append(
            (
                unit_price,
                lead_time_days,
                minimum_quantity,
                str(supplier),
                str(metadata.get("currency", "USD")).upper(),
            )
        )

    drafts = []
    for sku, item in items_by_sku.items():
        if item.on_hand > item.reorder_point:
            continue
        offers = offers_by_sku.get(sku)
        if not offers:
            continue
        unit_price, lead_time, minimum_quantity, supplier, currency = min(
            offers,
            key=lambda offer: (
                offer[0] * max(item.target_stock - item.on_hand, offer[2]),
                offer[0],
                offer[1],
                offer[3].casefold(),
            ),
        )
        quantity = max(item.target_stock - item.on_hand, minimum_quantity, 1)
        drafts.append(
            AutoPurchaseDraft(
                sku=sku,
                supplier=supplier,
                quantity=quantity,
                unit_price=unit_price,
                currency=currency,
                expected_date=current_date + timedelta(days=lead_time),
            )
        )
    return tuple(drafts)
