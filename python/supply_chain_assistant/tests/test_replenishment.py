"""Tests for pending automated and manual replenishment orders."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from supply_chain_assistant.auto_replenishment import build_auto_purchase_drafts
from supply_chain_assistant.database import Database
from supply_chain_assistant.models import Document, InventoryItem


class AutoReplenishmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.today = date(2026, 10, 4)
        self.inventory = (
            InventoryItem("LOW", "Low stock", 4, 5, 1.0, 20, category="Test"),
            InventoryItem("OK", "Healthy stock", 8, 5, 1.0, 20, category="Test"),
            InventoryItem("NO-OFFER", "No supplier offer", 2, 5, 1.0, 10),
        )
        self.documents = (
            Document(
                "offer-slow",
                "Slow supplier",
                "price_list",
                "Offer",
                {
                    "sku": "LOW",
                    "supplier": "Slow supplier",
                    "unit_price": 4.0,
                    "currency": "USD",
                    "lead_time_days": 10,
                    "minimum_order_quantity": 1,
                },
            ),
            Document(
                "offer-best",
                "Best total supplier",
                "price_list",
                "Offer",
                {
                    "sku": "LOW",
                    "supplier": "Best total supplier",
                    "unit_price": 3.0,
                    "currency": "USD",
                    "lead_time_days": 4,
                    "minimum_order_quantity": 1,
                },
            ),
            Document(
                "offer-moq",
                "High MOQ supplier",
                "price_list",
                "Offer",
                {
                    "sku": "LOW",
                    "supplier": "High MOQ supplier",
                    "unit_price": 2.0,
                    "currency": "USD",
                    "lead_time_days": 2,
                    "minimum_order_quantity": 30,
                },
            ),
        )

    def test_autopo_only_drafts_below_threshold_products_with_quotes(self) -> None:
        drafts = build_auto_purchase_drafts(
            self.inventory,
            self.documents,
            today=self.today,
        )

        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0].sku, "LOW")
        self.assertEqual(drafts[0].supplier, "Best total supplier")
        self.assertEqual(drafts[0].quantity, 16)
        self.assertEqual(drafts[0].expected_date, date(2026, 10, 8))

    def test_autopo_respects_supplier_minimum_order_quantity(self) -> None:
        documents = tuple(
            document
            for document in self.documents
            if document.metadata["supplier"] == "High MOQ supplier"
        )
        (draft,) = build_auto_purchase_drafts(
            self.inventory,
            documents,
            today=self.today,
        )

        self.assertEqual(draft.quantity, 30)

    def test_pending_order_persistence_and_autopo_idempotency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "test.sqlite3")
            arguments = {
                "sku": "LOW",
                "supplier": "Best total supplier",
                "quantity": 16,
                "unit_price": 3.0,
                "currency": "usd",
                "expected_date": date(2026, 10, 8),
                "source": "auto",
                "idempotency_key": "auto-replenishment:LOW",
            }
            first_id, first_created = database.create_pending_purchase_order(
                **arguments
            )
            second_id, second_created = database.create_pending_purchase_order(
                **arguments
            )

            self.assertTrue(first_created)
            self.assertFalse(second_created)
            self.assertEqual(first_id, second_id)
            (saved,) = database.list_pending_purchase_orders()
            self.assertEqual(saved["status"], "pending")
            self.assertEqual(saved["currency"], "USD")
            self.assertEqual(saved["total_cost"], 48.0)
            self.assertEqual(saved["expected_date"], "2026-10-08")

    def test_manual_pending_orders_allow_multiple_entries_without_autopo_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "test.sqlite3")
            arguments = {
                "sku": "LOW",
                "supplier": "Selected supplier",
                "quantity": 2,
                "unit_price": 5.0,
                "currency": "USD",
                "expected_date": self.today,
                "source": "manual",
            }
            first_id, _ = database.create_pending_purchase_order(**arguments)
            second_id, _ = database.create_pending_purchase_order(**arguments)

            self.assertNotEqual(first_id, second_id)
            self.assertEqual(len(database.list_pending_purchase_orders()), 2)

    def test_pending_order_rejects_invalid_quantity_or_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "test.sqlite3")
            arguments = {
                "sku": "LOW",
                "supplier": "Selected supplier",
                "quantity": 0,
                "unit_price": 5.0,
                "currency": "USD",
                "expected_date": self.today,
                "source": "manual",
            }
            with self.assertRaises(ValueError):
                database.create_pending_purchase_order(**arguments)


if __name__ == "__main__":
    unittest.main()
