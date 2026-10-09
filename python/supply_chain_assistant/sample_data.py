"""Idempotent sample records for a fresh local database."""

from datetime import date, timedelta

from .database import Database
from .models import (
    Document,
    FinancialKPIRecord,
    FulfillmentRecord,
    InventoryItem,
    LandedCostRecord,
    PurchaseDraft,
    SupplierDeliveryRecord,
)

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
    Document(
        document_id="price-list-cedar-tea-demo",
        source="cedar-tea-demo-price-list",
        category="price_list",
        content=(
            "Illustrative demo offer for TEA-023 Mountain mint tea, 500 g: "
            "USD 14.25 per unit from Cedar Leaf Tea. Lead time 6 days; minimum "
            "order 12 units. Supplier map point is an approximate demo location."
        ),
        metadata={
            "sku": "TEA-023",
            "supplier": "Cedar Leaf Tea",
            "unit_price": 14.25,
            "currency": "USD",
            "lead_time_days": 6,
            "minimum_order_quantity": 12,
            "source": "Illustrative demo price list",
            "latitude": 45.5152,
            "longitude": -122.6784,
        },
    ),
    Document(
        document_id="price-list-vermont-maple-demo",
        source="vermont-maple-demo-price-list",
        category="price_list",
        content=(
            "Illustrative demo offer for SYR-014 maple syrup, 500 ml: USD 9.80 "
            "per unit from Green Mountain Pantry. Lead time 5 days; minimum "
            "order 10 units. Supplier map point is an approximate demo location."
        ),
        metadata={
            "sku": "SYR-014",
            "supplier": "Green Mountain Pantry",
            "unit_price": 9.8,
            "currency": "USD",
            "lead_time_days": 5,
            "minimum_order_quantity": 10,
            "source": "Illustrative demo price list",
            "latitude": 44.4759,
            "longitude": -73.2121,
        },
    ),
    Document(
        document_id="price-list-atlas-ceramics-demo",
        source="atlas-ceramics-demo-price-list",
        category="price_list",
        content=(
            "Illustrative demo offer for MUG-018 ceramic takeaway mug: USD 6.50 "
            "per unit from Atlas Ceramics. Lead time 12 days; minimum order "
            "6 units. Supplier map point is an approximate demo location."
        ),
        metadata={
            "sku": "MUG-018",
            "supplier": "Atlas Ceramics",
            "unit_price": 6.5,
            "currency": "USD",
            "lead_time_days": 12,
            "minimum_order_quantity": 6,
            "source": "Illustrative demo price list",
            "latitude": 35.0844,
            "longitude": -106.6504,
        },
    ),
    Document(
        document_id="price-list-evergreen-packaging-demo",
        source="evergreen-packaging-demo-price-list",
        category="price_list",
        content=(
            "Illustrative demo offer for BOX-024 kraft pastry box: USD 0.42 "
            "per unit from Evergreen Packaging. Lead time 3 days; minimum order "
            "100 units. Supplier map point is an approximate demo location."
        ),
        metadata={
            "sku": "BOX-024",
            "supplier": "Evergreen Packaging",
            "unit_price": 0.42,
            "currency": "USD",
            "lead_time_days": 3,
            "minimum_order_quantity": 100,
            "source": "Illustrative demo price list",
            "latitude": 39.7392,
            "longitude": -104.9903,
        },
    ),
)

def _global_demo_offer(
    *,
    sku: str,
    product_name: str,
    supplier: str,
    city: str,
    country: str,
    unit_price: float,
    currency: str,
    lead_time_days: int,
    minimum_order_quantity: int,
    latitude: float,
    longitude: float,
    reliability_score: float,
) -> Document:
    return Document(
        document_id=f"global-demo-price-{sku.lower()}",
        source=f"Illustrative demo price list — {supplier}",
        category="price_list",
        content=(
            f"Illustrative demo offer for {product_name} ({sku}) from {supplier} "
            f"in {city}, {country}: {currency} {unit_price:.2f} per unit; "
            f"lead time {lead_time_days} days; minimum order "
            f"{minimum_order_quantity} units. Demo reliability score: "
            f"{reliability_score:.0%}. Coordinates are approximate demo locations."
        ),
        metadata={
            "sku": sku,
            "supplier": supplier,
            "unit_price": unit_price,
            "currency": currency,
            "lead_time_days": lead_time_days,
            "minimum_order_quantity": minimum_order_quantity,
            "source": "Illustrative demo price list",
            "latitude": latitude,
            "longitude": longitude,
            "country": country,
            "reliability_score": reliability_score,
            "is_illustrative_demo": True,
        },
    )


GLOBAL_DEMO_OFFERS = (
    _global_demo_offer(
        sku="SPC-031", product_name="Spanish saffron threads",
        supplier="Iberia Spice Cooperative", city="Valencia", country="Spain",
        unit_price=6.20, currency="EUR", lead_time_days=13,
        minimum_order_quantity=10, latitude=39.4699, longitude=-0.3763,
        reliability_score=0.96,
    ),
    _global_demo_offer(
        sku="OIL-042", product_name="Extra-virgin olive oil",
        supplier="Aegean Grove Exports", city="Athens", country="Greece",
        unit_price=8.10, currency="EUR", lead_time_days=11,
        minimum_order_quantity=12, latitude=37.9838, longitude=23.7275,
        reliability_score=0.93,
    ),
    _global_demo_offer(
        sku="RCE-055", product_name="Jasmine rice, 5 kg",
        supplier="Chao Phraya Foods", city="Bangkok", country="Thailand",
        unit_price=68.00, currency="THB", lead_time_days=16,
        minimum_order_quantity=20, latitude=13.7563, longitude=100.5018,
        reliability_score=0.91,
    ),
    _global_demo_offer(
        sku="FAB-067", product_name="Reusable cotton produce bags",
        supplier="Deccan Cotton Works", city="Mumbai", country="India",
        unit_price=145.00, currency="INR", lead_time_days=24,
        minimum_order_quantity=50, latitude=19.0760, longitude=72.8777,
        reliability_score=0.89,
    ),
    _global_demo_offer(
        sku="CAC-078", product_name="Natural cocoa powder, 1 kg",
        supplier="Ashanti Cocoa Partners", city="Accra", country="Ghana",
        unit_price=92.00, currency="GHS", lead_time_days=21,
        minimum_order_quantity=15, latitude=5.6037, longitude=-0.1870,
        reliability_score=0.87,
    ),
    _global_demo_offer(
        sku="BAT-089", product_name="Rechargeable scanner battery",
        supplier="Osaka Power Components", city="Osaka", country="Japan",
        unit_price=1200.00, currency="JPY", lead_time_days=10,
        minimum_order_quantity=8, latitude=34.6937, longitude=135.5023,
        reliability_score=0.98,
    ),
    _global_demo_offer(
        sku="GLV-091", product_name="Nitrile work gloves, 100-pack",
        supplier="Penang Safety Goods", city="George Town", country="Malaysia",
        unit_price=12.80, currency="MYR", lead_time_days=14,
        minimum_order_quantity=20, latitude=5.4141, longitude=100.3288,
        reliability_score=0.92,
    ),
    _global_demo_offer(
        sku="PPR-103", product_name="Recycled shipping paper, 10 kg",
        supplier="Nordic Circular Packaging", city="Stockholm", country="Sweden",
        unit_price=42.00, currency="SEK", lead_time_days=8,
        minimum_order_quantity=10, latitude=59.3293, longitude=18.0686,
        reliability_score=0.95,
    ),
)
SAMPLE_DOCUMENTS += GLOBAL_DEMO_OFFERS

SAMPLE_INVENTORY = (
    InventoryItem(
        "COF-001", "House coffee beans, 1 kg", 8, 12, 2.0, 40, 5,
        category="Coffee",
    ),
    InventoryItem(
        "CUP-012", "Compostable takeaway cups, 12 oz", 18, 24, 3.0, 90, 4,
        category="Food-service packaging",
    ),
    InventoryItem(
        "TEA-023",
        "Mountain mint tea, 500 g",
        75,
        18,
        1.5,
        120,
        6,
        41.8781,
        -87.6298,
        "Tea and beverages",
    ),
    InventoryItem(
        "SYR-014",
        "Maple syrup, 500 ml",
        48,
        12,
        0.8,
        80,
        5,
        41.8781,
        -87.6298,
        "Sweeteners",
    ),
    InventoryItem(
        "MUG-018",
        "Ceramic takeaway mug",
        60,
        20,
        1.2,
        100,
        12,
        41.8781,
        -87.6298,
        "Reusable serveware",
    ),
    InventoryItem(
        "BOX-024",
        "Kraft pastry box",
        250,
        60,
        8.0,
        400,
        3,
        41.8781,
        -87.6298,
        "Food-service packaging",
    ),
    InventoryItem(
        "SPC-031", "Spanish saffron threads", 500, 25, 0.3, 700, 13,
        41.8781, -87.6298, "Spices",
    ),
    InventoryItem(
        "OIL-042", "Extra-virgin olive oil", 500, 40, 1.4, 700, 11,
        41.8781, -87.6298, "Oils and condiments",
    ),
    InventoryItem(
        "RCE-055", "Jasmine rice, 5 kg", 1000, 100, 3.0, 1500, 16,
        41.8781, -87.6298, "Grains and staples",
    ),
    InventoryItem(
        "FAB-067", "Reusable cotton produce bags", 2400, 200, 6.0, 3600, 24,
        41.8781, -87.6298, "Reusable packaging",
    ),
    InventoryItem(
        "CAC-078", "Natural cocoa powder, 1 kg", 500, 50, 1.0, 750, 21,
        41.8781, -87.6298, "Baking ingredients",
    ),
    InventoryItem(
        "BAT-089", "Rechargeable scanner battery", 200, 30, 0.6, 300, 10,
        41.8781, -87.6298, "Equipment accessories",
    ),
    InventoryItem(
        "GLV-091", "Nitrile work gloves, 100-pack", 1000, 100, 4.0, 1500, 14,
        41.8781, -87.6298, "Safety supplies",
    ),
    InventoryItem(
        "PPR-103", "Recycled shipping paper, 10 kg", 500, 50, 2.0, 750, 8,
        41.8781, -87.6298, "Shipping materials",
    ),
)

SAMPLE_CONTACTS = {
    "Harbor Roasters": "purchasing@harbor-roasters.example",
    "Mountain Coffee Supply": "orders@mountain-coffee.example",
    "GreenPack Wholesale": "sales@greenpack.example",
    "Cedar Leaf Tea": "orders@cedar-leaf-tea.example",
    "Green Mountain Pantry": "sales@green-mountain-pantry.example",
    "Atlas Ceramics": "wholesale@atlas-ceramics.example",
    "Evergreen Packaging": "orders@evergreen-packaging.example",
    "Iberia Spice Cooperative": "orders@iberia-spice.example",
    "Aegean Grove Exports": "sales@aegean-grove.example",
    "Chao Phraya Foods": "orders@chaophraya-foods.example",
    "Deccan Cotton Works": "sales@deccan-cotton.example",
    "Ashanti Cocoa Partners": "orders@ashanti-cocoa.example",
    "Osaka Power Components": "sales@osaka-power.example",
    "Penang Safety Goods": "orders@penang-safety.example",
    "Nordic Circular Packaging": "sales@nordic-circular.example",
}

SAMPLE_ORDERS = (
    (
        "demo-order-coffee-sent",
        PurchaseDraft(
            sku="COF-001",
            product_name="House coffee beans, 1 kg",
            supplier="Harbor Roasters",
            quantity=20,
            unit_price=18.40,
            total_cost=368.00,
            currency="USD",
            subject="[SAMPLE] Coffee replenishment",
            body="Illustrative demo order only. No purchase was placed and no email was sent.",
            evidence_sources=("Illustrative demo price list",),
        ),
        "purchasing@harbor-roasters.example",
        "sent",
    ),
    (
        "demo-order-cups-approved",
        PurchaseDraft(
            sku="CUP-012",
            product_name="Compostable takeaway cups, 12 oz",
            supplier="GreenPack Wholesale",
            quantity=100,
            unit_price=0.08,
            total_cost=8.00,
            currency="USD",
            subject="[SAMPLE] Packaging replenishment",
            body="Illustrative approved-status demo only. No purchase was placed.",
            evidence_sources=("Illustrative demo price list",),
        ),
        "sales@greenpack.example",
        "approved",
    ),
    (
        "demo-order-tea-approved",
        PurchaseDraft(
            sku="TEA-023",
            product_name="Mountain mint tea, 500 g",
            supplier="Cedar Leaf Tea",
            quantity=12,
            unit_price=14.25,
            total_cost=171.00,
            currency="USD",
            subject="[SAMPLE] Tea replenishment",
            body="Illustrative approved-status demo only. No purchase was placed.",
            evidence_sources=("Illustrative demo price list",),
        ),
        "orders@cedar-leaf-tea.example",
        "approved",
    ),
)

SAMPLE_FINANCIAL_KPIS = (
    FinancialKPIRecord(
        period_start=date.today() - timedelta(days=365),
        period_end=date.today() - timedelta(days=1),
        currency="USD",
        cost_of_goods_sold=60000.0,
        average_inventory_value=12000.0,
    ),
)

SAMPLE_FULFILLMENT_RECORDS = (
    FulfillmentRecord("demo-demand-001", "COF-001", date.today() - timedelta(days=30), 100, 98),
    FulfillmentRecord("demo-demand-002", "COF-001", date.today() - timedelta(days=20), 200, 191),
    FulfillmentRecord("demo-demand-003", "CUP-012", date.today() - timedelta(days=10), 150, 150),
)

SAMPLE_SUPPLIER_DELIVERIES = (
    SupplierDeliveryRecord(
        "demo-delivery-001", "Harbor Roasters", "COF-001",
        date.today() - timedelta(days=35), date.today() - timedelta(days=35), 100, 100,
    ),
    SupplierDeliveryRecord(
        "demo-delivery-002", "GreenPack Wholesale", "CUP-012",
        date.today() - timedelta(days=28), date.today() - timedelta(days=27), 500, 500,
    ),
    SupplierDeliveryRecord(
        "demo-delivery-003", "Harbor Roasters", "COF-001",
        date.today() - timedelta(days=21), date.today() - timedelta(days=21), 50, 45,
    ),
    SupplierDeliveryRecord(
        "demo-delivery-004", "GreenPack Wholesale", "CUP-012",
        date.today() - timedelta(days=14), date.today() - timedelta(days=14), 1000, 1000,
    ),
)

SAMPLE_LANDED_COST_RECORDS = (
    LandedCostRecord(
        "demo-receipt-cof-001",
        "COF-001",
        "Harbor Roasters",
        date.today() - timedelta(days=12),
        100,
        18.40,
        95.00,
        20.00,
        25.00,
        "USD",
    ),
    LandedCostRecord(
        "demo-receipt-cup-012",
        "CUP-012",
        "GreenPack Wholesale",
        date.today() - timedelta(days=8),
        1000,
        0.08,
        35.00,
        4.00,
        12.00,
        "USD",
    ),
)


def seed_demo_data(database: Database) -> None:
    existing_document_ids = {
        document.document_id for document in database.list_documents()
    }
    for document in SAMPLE_DOCUMENTS:
        if document.document_id not in existing_document_ids:
            database.save_document(document)

    existing_skus = {item.sku for item in database.list_inventory()}
    for item in SAMPLE_INVENTORY:
        if item.sku not in existing_skus:
            database.save_inventory_item(item, item.supplier_lead_time_days)
    if not database.list_financial_kpis():
        for record in SAMPLE_FINANCIAL_KPIS:
            database.save_financial_kpi(record)
    if not database.list_fulfillment_records():
        for record in SAMPLE_FULFILLMENT_RECORDS:
            database.save_fulfillment_record(record)
    if not database.list_supplier_deliveries():
        for record in SAMPLE_SUPPLIER_DELIVERIES:
            database.save_supplier_delivery(record)
    if not database.list_landed_cost_records():
        for record in SAMPLE_LANDED_COST_RECORDS:
            database.save_landed_cost_record(record)
    for supplier, email in SAMPLE_CONTACTS.items():
        if database.get_supplier_contact(supplier) is None:
            database.save_supplier_contact(supplier, email)
    for demo_key, draft, recipient, status in SAMPLE_ORDERS:
        database.seed_demo_order(
            demo_key,
            draft,
            recipient=recipient,
            status=status,
        )
    today = date.today()
    for item in SAMPLE_INVENTORY:
        if database.get_usage_history(item.sku, limit=1):
            continue
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