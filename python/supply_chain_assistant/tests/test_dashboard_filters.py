"""Tests for global dashboard filter behavior."""

from __future__ import annotations

import unittest
from datetime import date

from supply_chain_assistant.app import (
    _filter_inventory_and_documents,
    _location_label,
    _record_date,
)
from supply_chain_assistant.models import Document, InventoryItem


class DashboardFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inventory = (
            InventoryItem("A-1", "Product A", 10, 2, 1.0, 20, category="A"),
            InventoryItem("B-1", "Product B", 8, 2, 1.0, 20, category="B"),
        )
        self.documents = (
            Document(
                "price-a",
                "A supplier",
                "price_list",
                "Offer for A-1",
                {
                    "sku": "A-1",
                    "supplier": "A supplier",
                    "latitude": 10.0,
                    "longitude": 20.0,
                },
            ),
            Document(
                "contract-a",
                "A supplier",
                "supplier_contract",
                "Contract for A supplier",
                {"supplier": "A supplier"},
            ),
            Document(
                "unmapped-price",
                "Unmapped supplier",
                "price_list",
                "Offer without location",
                {"sku": "A-1", "supplier": "Unmapped supplier"},
            ),
            Document(
                "price-b",
                "B supplier",
                "price_list",
                "Offer for B-1",
                {
                    "sku": "B-1",
                    "supplier": "B supplier",
                    "latitude": 30.0,
                    "longitude": 40.0,
                },
            ),
        )
        self.locations = {
            _location_label(("A supplier", 10.0, 20.0)): (
                "A supplier",
                10.0,
                20.0,
            ),
            _location_label(("B supplier", 30.0, 40.0)): (
                "B supplier",
                30.0,
                40.0,
            ),
        }

    def test_all_selected_locations_preserve_category_filtered_products(self) -> None:
        inventory, documents, suppliers = _filter_inventory_and_documents(
            self.inventory,
            self.documents,
            {"A"},
            set(self.locations),
            self.locations,
        )

        self.assertEqual(tuple(item.sku for item in inventory), ("A-1",))
        self.assertEqual(
            {document.document_id for document in documents},
            {"price-a", "contract-a", "unmapped-price"},
        )
        self.assertEqual(suppliers, {"A supplier", "Unmapped supplier"})

    def test_selected_supplier_location_scopes_products_and_evidence(self) -> None:
        selected_location = _location_label(("A supplier", 10.0, 20.0))
        inventory, documents, suppliers = _filter_inventory_and_documents(
            self.inventory,
            self.documents,
            {"A", "B"},
            {selected_location},
            self.locations,
        )

        self.assertEqual(tuple(item.sku for item in inventory), ("A-1",))
        self.assertEqual(
            {document.document_id for document in documents},
            {"price-a", "contract-a"},
        )
        self.assertEqual(suppliers, {"A supplier"})

    def test_empty_selection_returns_no_products_or_sku_scoped_documents(self) -> None:
        inventory, documents, suppliers = _filter_inventory_and_documents(
            self.inventory,
            self.documents,
            set(),
            set(self.locations),
            self.locations,
        )

        self.assertEqual(inventory, ())
        self.assertEqual(documents, ())
        self.assertEqual(suppliers, set())

    def test_record_date_accepts_iso_timestamp_and_date(self) -> None:
        self.assertEqual(_record_date("2026-10-04 12:30:00"), date(2026, 10, 4))
        self.assertEqual(_record_date(date(2026, 10, 4)), date(2026, 10, 4))
        self.assertIsNone(_record_date(None))


if __name__ == "__main__":
    unittest.main()
