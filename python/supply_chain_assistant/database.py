"""SQLite persistence for inventory, RAG documents, usage, contacts, and orders."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Generator

from .models import Document, InventoryItem, PurchaseDraft


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
                    supplier_lead_time_days INTEGER NOT NULL DEFAULT 5
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
                    error TEXT
                );
                """
            )

    def save_inventory_item(self, item: InventoryItem, supplier_lead_time_days: int = 5) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO inventory_items
                    (sku, name, on_hand, reorder_point, average_daily_usage,
                     target_stock, supplier_lead_time_days)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sku) DO UPDATE SET
                    name=excluded.name,
                    on_hand=excluded.on_hand,
                    reorder_point=excluded.reorder_point,
                    average_daily_usage=excluded.average_daily_usage,
                    target_stock=excluded.target_stock,
                    supplier_lead_time_days=excluded.supplier_lead_time_days
                """,
                (
                    item.sku,
                    item.name,
                    item.on_hand,
                    item.reorder_point,
                    item.average_daily_usage,
                    item.target_stock,
                    supplier_lead_time_days,
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