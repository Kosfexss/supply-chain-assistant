"""Idempotent sample records for a fresh local database."""

from datetime import date, timedelta

from .database import Database
from .models import Document, InventoryItem

SAMPLE_DOCUMENTS = (
    Document(
        document_id="price-list-harbor-2026-09",
        source="harbor-roasters-price-list",
        category="price_list",
        content=(
            "Harbor Roasters September 2026 price list. COF-001 House coffee beans "
            "1 kg costs USD 18.40 per unit. Standard lead time 5 days. Minimum order 10 units."
        ),
        metadata={
            "sku": "COF-001", "supplier": "Harbor Roasters", "unit_price": 18.4,
            "currency": "USD", "lead_time_days": 5, "minimum_order_quantity": 10,
            "source": "September 2026 price list",
        },
    ),
    Document(
        document_id="price-list-mountain-2026-09",
        source="mountain-coffee-price-list",
        category="price_list",
        content=(
            "Mountain Coffee Supply September 2026 price list. COF-001 House coffee "
            "beans 1 kg costs USD 17.95 per unit. Standard lead time 9 days. Minimum order 20 units."
        ),
        metadata={
            "sku": "COF-001", "supplier": "Mountain Coffee Supply", "unit_price": 17.95,
            "currency": "USD", "lead_time_days": 9, "minimum_order_quantity": 20,
            "source": "September 2026 price list",
        },
    ),
    Document(
        document_id="price-list-greenpack-2026-09",
        source="greenpack-price-list",
        category="price_list",
        content=(
            "GreenPack Wholesale September 2026 price list. CUP-012 Compostable takeaway "
            "cups 12 oz cost USD 0.08 per unit. Standard lead time 4 days. Minimum order 100 units."
        ),
        metadata={
            "sku": "CUP-012", "supplier": "GreenPack Wholesale", "unit_price": 0.08,
            "currency": "USD", "lead_time_days": 4, "minimum_order_quantity": 100,
            "source": "September 2026 price list",
        },
    ),
    Document(
        document_id="contract-harbor-coffee-2026",
        source="harbor-roasters-contract",
        category="supplier_contract",
        content=(
            "Harbor Roasters contract for COF-001. Orders of 30 or more units receive a "
            "3 percent discount. Quotes are valid for 30 days and delivery is subject to confirmation."
        ),
        metadata={
            "sku": "COF-001", "supplier": "Harbor Roasters",
            "discount_threshold_quantity": 30, "discount_percentage": 3,
        },
    ),
    Document(
        document_id="invoice-coffee-history-2026",
        source="coffee-invoice-history",
        category="historical_invoice",
        content=(
            "Invoice history for COF-001: the business purchased 25 units from Harbor "
            "Roasters at USD 18.70 per unit in July 2026, with delivery in 5 days."
        ),
        metadata={"sku": "COF-001", "supplier": "Harbor Roasters"},
    ),
    Document(
        document_id="compliance-packaging-2026",
        source="packaging-compliance-note",
        category="compliance",
        content=(
            "For compostable food-service packaging, retain supplier declarations that "
            "the product meets applicable food-contact and compostability requirements. "
            "Confirm current certificates and local disposal rules before ordering."
        ),
        metadata={"sku": "CUP-012", "supplier": "GreenPack Wholesale"},
    ),
)

SAMPLE_INVENTORY = (
    InventoryItem("COF-001", "House coffee beans, 1 kg", 8, 12, 2.0, 40, 5),
    InventoryItem("CUP-012", "Compostable takeaway cups, 12 oz", 18, 24, 3.0, 90, 4),
)

SAMPLE_CONTACTS = {
    "Harbor Roasters": "purchasing@harbor-roasters.example",
    "Mountain Coffee Supply": "orders@mountain-coffee.example",
    "GreenPack Wholesale": "sales@greenpack.example",
}


def seed_demo_data(database: Database) -> None:
    if not database.list_documents():
        for document in SAMPLE_DOCUMENTS:
            database.save_document(document)
    if not database.list_inventory():
        for item in SAMPLE_INVENTORY:
            database.save_inventory_item(item, item.supplier_lead_time_days)
    for supplier, email in SAMPLE_CONTACTS.items():
        if database.get_supplier_contact(supplier) is None:
            database.save_supplier_contact(supplier, email)
    if not database.get_usage_history("COF-001", limit=1):
        today = date.today()
        for item in SAMPLE_INVENTORY:
            for day_offset in range(30, 0, -1):
                trend = (30 - day_offset) * 0.015
                baseline = item.average_daily_usage
                daily_units = max(0.0, baseline - 0.18 + trend)
                database.save_usage(
                    item.sku,
                    today - timedelta(days=day_offset),
                    round(daily_units, 2),
                )
    if not database.get_setting("cash_balance"):
        database.set_setting("cash_balance", "1500")
    if not database.get_setting("minimum_cash_reserve"):
        database.set_setting("minimum_cash_reserve", "500")