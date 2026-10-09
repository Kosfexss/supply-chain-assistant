"""Shared data contracts for the supply-chain workflow."""

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class Document:
    document_id: str
    source: str
    category: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedDocument:
    document: Document
    score: float
    excerpt: str


@dataclass(frozen=True)
class InventoryItem:
    sku: str
    name: str
    on_hand: int
    reorder_point: int
    average_daily_usage: float
    target_stock: int
    supplier_lead_time_days: int = 5
    warehouse_latitude: float | None = None
    warehouse_longitude: float | None = None
    category: str | None = None


@dataclass(frozen=True)
class ReplenishmentNeed:
    item: InventoryItem
    quantity: int
    days_until_stockout: float | None
    trigger: str = "reorder_point"


@dataclass(frozen=True)
class StockForecast:
    sku: str
    product_name: str
    observations: int
    daily_usage_slope: float
    forecast_daily_usage: float
    projected_stock_at_lead_time: float
    days_until_reorder_point: float | None
    suggested_order_quantity: int
    proactive_order: bool


@dataclass(frozen=True)
class SupplierOffer:
    sku: str
    supplier: str
    unit_price: float
    currency: str
    lead_time_days: int
    minimum_order_quantity: int
    source: str
    discount_threshold_quantity: int = 0
    discount_percentage: float = 0.0
    latitude: float | None = None
    longitude: float | None = None


@dataclass(frozen=True)
class SupplierRecommendation:
    need: ReplenishmentNeed
    offer: SupplierOffer
    order_quantity: int
    total_cost: float
    evidence: tuple[RetrievedDocument, ...]


@dataclass(frozen=True)
class FinancialAssessment:
    approved: bool
    requested_amount: float
    spendable_cash: float
    currency: str
    reason: str


@dataclass(frozen=True)
class PurchaseDraft:
    sku: str
    product_name: str
    supplier: str
    quantity: int
    unit_price: float
    total_cost: float
    currency: str
    subject: str
    body: str
    evidence_sources: tuple[str, ...]
    discount_percentage: float = 0.0
    status: str = "draft_pending_human_approval"


@dataclass(frozen=True)
class WorkflowResult:
    low_stock: tuple[ReplenishmentNeed, ...]
    forecasts: tuple[StockForecast, ...]
    recommendations: tuple[SupplierRecommendation, ...]
    unmatched_skus: tuple[str, ...]
    financial_assessment: FinancialAssessment
    purchase_drafts: tuple[PurchaseDraft, ...]


@dataclass(frozen=True)
class FinancialKPIRecord:
    period_start: date
    period_end: date
    currency: str
    cost_of_goods_sold: float
    average_inventory_value: float


@dataclass(frozen=True)
class FulfillmentRecord:
    record_id: str
    sku: str
    recorded_at: date
    requested_quantity: int
    fulfilled_quantity: int


@dataclass(frozen=True)
class SupplierDeliveryRecord:
    delivery_id: str
    supplier: str
    sku: str
    promised_date: date
    actual_date: date | None
    promised_quantity: int
    delivered_quantity: int


@dataclass(frozen=True)
class LandedCostRecord:
    receipt_id: str
    sku: str
    supplier: str
    received_at: date
    quantity: int
    unit_price: float
    freight_cost: float
    duties: float
    handling_cost: float
    currency: str