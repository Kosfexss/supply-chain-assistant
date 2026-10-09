"""SQLite persistence for inventory, RAG documents, usage, contacts, and orders."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Generator

from .models import (
    Document,
    FinancialKPIRecord,
    FulfillmentRecord,
    InventoryItem,
    LandedCostRecord,
    PurchaseDraft,
    SupplierDeliveryRecord,
)


class Database:
    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = os.environ.get(
                "SUPPLY_CHAIN_DB_PATH",
                str(Path(__file__).parent / "data" / "supply_chain.sqlite3"),
            )
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS inventory_items (
                    sku TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    on_hand INTEGER NOT NULL CHECK (on_hand >= 0),
                    reorder_point INTEGER NOT NULL CHECK (reorder_point >= 0),
                    average_daily_usage REAL NOT NULL CHECK (average_daily_usage >= 0),
                    target_stock INTEGER NOT NULL CHECK (target_stock >= 0),
                    supplier_lead_time_days INTEGER NOT NULL DEFAULT 5,
                    warehouse_latitude REAL,
                    warehouse_longitude REAL,
                    category TEXT
                );
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS daily_usage (
                    sku TEXT NOT NULL REFERENCES inventory_items(sku) ON DELETE CASCADE,
                    usage_date TEXT NOT NULL,
                    units_used REAL NOT NULL CHECK (units_used >= 0),
                    PRIMARY KEY (sku, usage_date)
                );
                CREATE TABLE IF NOT EXISTS supplier_contacts (
                    supplier TEXT PRIMARY KEY,
                    email TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS order_logs (
                    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sku TEXT NOT NULL,
                    supplier TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    unit_price REAL NOT NULL,
                    total_cost REAL NOT NULL,
                    currency TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    body TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('approved', 'sent', 'failed')),
                    approved_by TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    sent_at TEXT,
                    error TEXT,
                    is_demo INTEGER NOT NULL DEFAULT 0 CHECK (is_demo IN (0, 1)),
                    demo_key TEXT UNIQUE
                );
                CREATE TABLE IF NOT EXISTS pending_purchase_orders (
                    pending_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sku TEXT NOT NULL,
                    supplier TEXT NOT NULL,
                    quantity INTEGER NOT NULL CHECK (quantity > 0),
                    unit_price REAL NOT NULL CHECK (unit_price >= 0),
                    total_cost REAL NOT NULL CHECK (total_cost >= 0),
                    currency TEXT NOT NULL,
                    expected_date TEXT NOT NULL,
                    source TEXT NOT NULL CHECK (source IN ('auto', 'manual')),
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'cancelled')),
                    idempotency_key TEXT UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS financial_kpis (
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    cost_of_goods_sold REAL NOT NULL CHECK (cost_of_goods_sold >= 0),
                    average_inventory_value REAL NOT NULL
                        CHECK (average_inventory_value >= 0),
                    PRIMARY KEY (period_start, period_end, currency)
                );
                CREATE TABLE IF NOT EXISTS fulfillment_records (
                    record_id TEXT PRIMARY KEY,
                    sku TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    requested_quantity INTEGER NOT NULL CHECK (requested_quantity >= 0),
                    fulfilled_quantity INTEGER NOT NULL CHECK (fulfilled_quantity >= 0)
                );
                CREATE TABLE IF NOT EXISTS supplier_deliveries (
                    delivery_id TEXT PRIMARY KEY,
                    supplier TEXT NOT NULL,
                    sku TEXT NOT NULL,
                    promised_date TEXT NOT NULL,
                    actual_date TEXT,
                    promised_quantity INTEGER NOT NULL CHECK (promised_quantity >= 0),
                    delivered_quantity INTEGER NOT NULL CHECK (delivered_quantity >= 0)
                );
                CREATE TABLE IF NOT EXISTS landed_cost_records (
                    receipt_id TEXT PRIMARY KEY,
                    sku TEXT NOT NULL,
                    supplier TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    quantity INTEGER NOT NULL CHECK (quantity > 0),
                    unit_price REAL NOT NULL CHECK (unit_price >= 0),
                    freight_cost REAL NOT NULL CHECK (freight_cost >= 0),
                    duties REAL NOT NULL CHECK (duties >= 0),
                    handling_cost REAL NOT NULL CHECK (handling_cost >= 0),
                    currency TEXT NOT NULL
                );
                """
            )
            inventory_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(inventory_items)"
                ).fetchall()
            }
            for column in ("warehouse_latitude", "warehouse_longitude"):
                if column not in inventory_columns:
                    connection.execute(
                        f"ALTER TABLE inventory_items ADD COLUMN {column} REAL"
                    )
            if "category" not in inventory_columns:
                connection.execute(
                    "ALTER TABLE inventory_items ADD COLUMN category TEXT"
                )
            order_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(order_logs)").fetchall()
            }
            if "is_demo" not in order_columns:
                connection.execute(
                    "ALTER TABLE order_logs ADD COLUMN is_demo INTEGER NOT NULL DEFAULT 0"
                )
            if "demo_key" not in order_columns:
                connection.execute("ALTER TABLE order_logs ADD COLUMN demo_key TEXT")
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_order_logs_demo_key "
                "ON order_logs(demo_key)"
            )

    def save_financial_kpi(self, record: FinancialKPIRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO financial_kpis
                    (period_start, period_end, currency, cost_of_goods_sold,
                     average_inventory_value)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(period_start, period_end, currency) DO UPDATE SET
                    cost_of_goods_sold=excluded.cost_of_goods_sold,
                    average_inventory_value=excluded.average_inventory_value
                """,
                (
                    record.period_start.isoformat(),
                    record.period_end.isoformat(),
                    record.currency,
                    record.cost_of_goods_sold,
                    record.average_inventory_value,
                ),
            )

    def list_financial_kpis(self) -> tuple[FinancialKPIRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM financial_kpis ORDER BY period_end DESC, currency"
            ).fetchall()
        return tuple(
            FinancialKPIRecord(
                period_start=date.fromisoformat(row["period_start"]),
                period_end=date.fromisoformat(row["period_end"]),
                currency=row["currency"],
                cost_of_goods_sold=row["cost_of_goods_sold"],
                average_inventory_value=row["average_inventory_value"],
            )
            for row in rows
        )

    def save_fulfillment_record(self, record: FulfillmentRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO fulfillment_records
                    (record_id, sku, recorded_at, requested_quantity, fulfilled_quantity)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    sku=excluded.sku,
                    recorded_at=excluded.recorded_at,
                    requested_quantity=excluded.requested_quantity,
                    fulfilled_quantity=excluded.fulfilled_quantity
                """,
                (
                    record.record_id,
                    record.sku,
                    record.recorded_at.isoformat(),
                    record.requested_quantity,
                    record.fulfilled_quantity,
                ),
            )

    def list_fulfillment_records(self) -> tuple[FulfillmentRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM fulfillment_records ORDER BY recorded_at, record_id"
            ).fetchall()
        return tuple(
            FulfillmentRecord(
                record_id=row["record_id"],
                sku=row["sku"],
                recorded_at=date.fromisoformat(row["recorded_at"]),
                requested_quantity=row["requested_quantity"],
                fulfilled_quantity=row["fulfilled_quantity"],
            )
            for row in rows
        )

    def save_supplier_delivery(self, record: SupplierDeliveryRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO supplier_deliveries
                    (delivery_id, supplier, sku, promised_date, actual_date,
                     promised_quantity, delivered_quantity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(delivery_id) DO UPDATE SET
                    supplier=excluded.supplier,
                    sku=excluded.sku,
                    promised_date=excluded.promised_date,
                    actual_date=excluded.actual_date,
                    promised_quantity=excluded.promised_quantity,
                    delivered_quantity=excluded.delivered_quantity
                """,
                (
                    record.delivery_id,
                    record.supplier,
                    record.sku,
                    record.promised_date.isoformat(),
                    record.actual_date.isoformat() if record.actual_date else None,
                    record.promised_quantity,
                    record.delivered_quantity,
                ),
            )

    def list_supplier_deliveries(self) -> tuple[SupplierDeliveryRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM supplier_deliveries ORDER BY promised_date, delivery_id"
            ).fetchall()
        return tuple(
            SupplierDeliveryRecord(
                delivery_id=row["delivery_id"],
                supplier=row["supplier"],
                sku=row["sku"],
                promised_date=date.fromisoformat(row["promised_date"]),
                actual_date=(
                    date.fromisoformat(row["actual_date"])
                    if row["actual_date"] else None
                ),
                promised_quantity=row["promised_quantity"],
                delivered_quantity=row["delivered_quantity"],
            )
            for row in rows
        )

    def save_landed_cost_record(self, record: LandedCostRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO landed_cost_records
                    (receipt_id, sku, supplier, received_at, quantity, unit_price,
                     freight_cost, duties, handling_cost, currency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(receipt_id) DO UPDATE SET
                    sku=excluded.sku,
                    supplier=excluded.supplier,
                    received_at=excluded.received_at,
                    quantity=excluded.quantity,
                    unit_price=excluded.unit_price,
                    freight_cost=excluded.freight_cost,
                    duties=excluded.duties,
                    handling_cost=excluded.handling_cost,
                    currency=excluded.currency
                """,
                (
                    record.receipt_id,
                    record.sku,
                    record.supplier,
                    record.received_at.isoformat(),
                    record.quantity,
                    record.unit_price,
                    record.freight_cost,
                    record.duties,
                    record.handling_cost,
                    record.currency,
                ),
            )

    def list_landed_cost_records(self) -> tuple[LandedCostRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM landed_cost_records ORDER BY received_at, receipt_id"
            ).fetchall()
        return tuple(
            LandedCostRecord(
                receipt_id=row["receipt_id"],
                sku=row["sku"],
                supplier=row["supplier"],
                received_at=date.fromisoformat(row["received_at"]),
                quantity=row["quantity"],
                unit_price=row["unit_price"],
                freight_cost=row["freight_cost"],
                duties=row["duties"],
                handling_cost=row["handling_cost"],
                currency=row["currency"],
            )
            for row in rows
        )

    def save_inventory_item(self, item: InventoryItem, supplier_lead_time_days: int = 5) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO inventory_items
                    (sku, name, on_hand, reorder_point, average_daily_usage,
                     target_stock, supplier_lead_time_days, warehouse_latitude,
                     warehouse_longitude, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sku) DO UPDATE SET
                    name=excluded.name,
                    on_hand=excluded.on_hand,
                    reorder_point=excluded.reorder_point,
                    average_daily_usage=excluded.average_daily_usage,
                    target_stock=excluded.target_stock,
                    supplier_lead_time_days=excluded.supplier_lead_time_days,
                    warehouse_latitude=excluded.warehouse_latitude,
                    warehouse_longitude=excluded.warehouse_longitude,
                    category=excluded.category
                """,
                (
                    item.sku,
                    item.name,
                    item.on_hand,
                    item.reorder_point,
                    item.average_daily_usage,
                    item.target_stock,
                    supplier_lead_time_days,
                    item.warehouse_latitude,
                    item.warehouse_longitude,
                    item.category,
                ),
            )

    def list_inventory(self) -> tuple[InventoryItem, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM inventory_items ORDER BY sku"
            ).fetchall()
        return tuple(
            InventoryItem(
                sku=row["sku"],
                name=row["name"],
                on_hand=row["on_hand"],
                reorder_point=row["reorder_point"],
                average_daily_usage=row["average_daily_usage"],
                target_stock=row["target_stock"],
                supplier_lead_time_days=row["supplier_lead_time_days"],
                warehouse_latitude=row["warehouse_latitude"],
                warehouse_longitude=row["warehouse_longitude"],
                category=row["category"],
            )
            for row in rows
        )

    def save_usage(self, sku: str, usage_date: date | str, units_used: float) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO daily_usage (sku, usage_date, units_used) VALUES (?, ?, ?)
                ON CONFLICT(sku, usage_date) DO UPDATE SET units_used=excluded.units_used
                """,
                (sku, usage_date.isoformat() if isinstance(usage_date, date) else usage_date, units_used),
            )

    def get_usage_history(
        self, sku: str, *, limit: int = 90
    ) -> tuple[tuple[str, float], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT usage_date, units_used FROM (
                    SELECT usage_date, units_used FROM daily_usage
                    WHERE sku = ? ORDER BY usage_date DESC LIMIT ?
                ) ORDER BY usage_date ASC
                """,
                (sku, limit),
            ).fetchall()
        return tuple((row["usage_date"], row["units_used"]) for row in rows)

    def save_document(self, document: Document) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (document_id, source, category, content, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    source=excluded.source,
                    category=excluded.category,
                    content=excluded.content,
                    metadata_json=excluded.metadata_json
                """,
                (
                    document.document_id,
                    document.source,
                    document.category,
                    document.content,
                    json.dumps(document.metadata),
                ),
            )

    def list_documents(self) -> tuple[Document, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY document_id"
            ).fetchall()
        return tuple(
            Document(
                document_id=row["document_id"],
                source=row["source"],
                category=row["category"],
                content=row["content"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        )

    def save_supplier_contact(self, supplier: str, email: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO supplier_contacts (supplier, email) VALUES (?, ?)
                ON CONFLICT(supplier) DO UPDATE SET email=excluded.email
                """,
                (supplier, email.strip()),
            )

    def get_supplier_contact(self, supplier: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT email FROM supplier_contacts WHERE supplier = ?", (supplier,)
            ).fetchone()
        return row["email"] if row else None

    def list_supplier_contacts(self) -> dict[str, str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT supplier, email FROM supplier_contacts ORDER BY supplier"
            ).fetchall()
        return {row["supplier"]: row["email"] for row in rows}

    def set_setting(self, key: str, value: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, str(value)),
            )

    def get_setting(self, key: str, default: str = "") -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else default

    def create_approved_order(
        self, draft: PurchaseDraft, *, recipient: str, approved_by: str
    ) -> int:
        if not approved_by.strip() or not recipient.strip():
            raise ValueError("An approver and recipient email are required")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO order_logs
                    (sku, supplier, recipient, quantity, unit_price, total_cost,
                     currency, subject, body, status, approved_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'approved', ?)
                """,
                (
                    draft.sku,
                    draft.supplier,
                    recipient.strip(),
                    draft.quantity,
                    draft.unit_price,
                    draft.total_cost,
                    draft.currency,
                    draft.subject,
                    draft.body,
                    approved_by.strip(),
                ),
            )
            return int(cursor.lastrowid)

    def get_order(self, order_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM order_logs WHERE order_id = ?", (order_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_orders(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM order_logs ORDER BY order_id DESC LIMIT ?", (limit,)
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def create_pending_purchase_order(
        self,
        *,
        sku: str,
        supplier: str,
        quantity: int,
        unit_price: float,
        currency: str,
        expected_date: date,
        source: str,
        idempotency_key: str | None = None,
    ) -> tuple[int, bool]:
        if not sku.strip() or not supplier.strip():
            raise ValueError("A product SKU and supplier are required.")
        if quantity <= 0:
            raise ValueError("Pending purchase order quantity must be positive.")
        if unit_price < 0:
            raise ValueError("Pending purchase order unit price cannot be negative.")
        if source not in {"auto", "manual"}:
            raise ValueError("Pending purchase order source must be 'auto' or 'manual'.")
        with self._connect() as connection:
            if idempotency_key is not None:
                cursor = connection.execute(
                    """
                    INSERT INTO pending_purchase_orders
                        (sku, supplier, quantity, unit_price, total_cost, currency,
                         expected_date, source, idempotency_key)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(idempotency_key) DO NOTHING
                    """,
                    (
                        sku.strip(),
                        supplier.strip(),
                        quantity,
                        unit_price,
                        round(quantity * unit_price, 2),
                        currency.strip().upper(),
                        expected_date.isoformat(),
                        source,
                        idempotency_key,
                    ),
                )
                row = connection.execute(
                    "SELECT pending_id FROM pending_purchase_orders "
                    "WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("Pending purchase order was not persisted.")
                return int(row["pending_id"]), cursor.rowcount == 1

            cursor = connection.execute(
                """
                INSERT INTO pending_purchase_orders
                    (sku, supplier, quantity, unit_price, total_cost, currency,
                     expected_date, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sku.strip(),
                    supplier.strip(),
                    quantity,
                    unit_price,
                    round(quantity * unit_price, 2),
                    currency.strip().upper(),
                    expected_date.isoformat(),
                    source,
                ),
            )
            return int(cursor.lastrowid), True

    def list_pending_purchase_orders(
        self, *, limit: int = 500
    ) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM pending_purchase_orders
                WHERE status = 'pending'
                ORDER BY created_at DESC, pending_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def seed_demo_order(
        self,
        demo_key: str,
        draft: PurchaseDraft,
        *,
        recipient: str,
        status: str,
    ) -> None:
        """Insert a clearly identified, non-operational sample order once."""
        if not demo_key.strip():
            raise ValueError("A demo order key is required.")
        if status not in {"approved", "sent"}:
            raise ValueError("Demo order status must be 'approved' or 'sent'.")
        if not recipient.lower().endswith(".example"):
            raise ValueError("Demo order recipients must use the reserved .example domain.")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO order_logs
                    (sku, supplier, recipient, quantity, unit_price, total_cost,
                     currency, subject, body, status, approved_by, sent_at,
                     is_demo, demo_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        CASE WHEN ? = 'sent' THEN CURRENT_TIMESTAMP ELSE NULL END,
                        1, ?)
                """,
                (
                    draft.sku,
                    draft.supplier,
                    recipient,
                    draft.quantity,
                    draft.unit_price,
                    draft.total_cost,
                    draft.currency,
                    draft.subject,
                    draft.body,
                    status,
                    "Demo fixture — no real approval",
                    status,
                    demo_key,
                ),
            )

    def mark_order_sent(self, order_id: int) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE order_logs SET status='sent', sent_at=CURRENT_TIMESTAMP, error=NULL
                WHERE order_id=? AND status='approved'
                """,
                (order_id,),
            )
            if cursor.rowcount != 1:
                raise ValueError("Only an approved, unsent order can be dispatched")

    def mark_order_failed(self, order_id: int, error: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE order_logs SET status='failed', error=?
                WHERE order_id=? AND status='approved'
                """,
                (error[:1000], order_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Only an approved order can be marked as failed")