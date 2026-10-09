"""Regression tests for CSV/Excel upload parsing and session data integration."""

from __future__ import annotations

import unittest
from io import BytesIO

import pandas as pd

from supply_chain_assistant.models import Document, InventoryItem
from supply_chain_assistant.orchestrator import SupplyChainAssistant
from supply_chain_assistant.rag import LocalDocumentStore
from supply_chain_assistant.geospatial import build_supply_chain_map_data
from supply_chain_assistant.upload_data import (
    merge_inventory_uploads,
    merge_supplier_documents,
    parse_inventory_table,
    parse_product_table,
    parse_supplier_table,
    read_uploaded_table,
)


class UploadedFile:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


class UploadDataTests(unittest.TestCase):
    def test_csv_and_excel_uploads_are_parsed(self) -> None:
        csv_file = UploadedFile(
            "inventory.csv",
            b"SKU,Current Stock,Reorder Level\nSKU-1,12,4\n",
        )
        inventory = parse_inventory_table(read_uploaded_table(csv_file))
        self.assertEqual(inventory[0].sku, "SKU-1")
        self.assertEqual(inventory[0].on_hand, 12)
        self.assertEqual(inventory[0].target_stock, 12)

        excel_buffer = BytesIO()
        pd.DataFrame(
            [{"SKU": "SKU-2", "Product Name": "Widget", "On Hand": 7, "Reorder Point": 2}]
        ).to_excel(excel_buffer, index=False)
        excel_file = UploadedFile("products.xlsx", excel_buffer.getvalue())
        product = parse_product_table(read_uploaded_table(excel_file))
        self.assertEqual(product[0].sku, "SKU-2")
        self.assertEqual(product[0].name, "Widget")

    def test_uploaded_inventory_replaces_and_product_data_enriches(self) -> None:
        existing = (
            InventoryItem("OLD", "Old product", 5, 2, 1.0, 10, 5),
            InventoryItem("MATCH", "Saved name", 8, 3, 2.0, 12, 4),
        )
        uploaded_inventory = parse_inventory_table(
            pd.DataFrame(
                [
                    {
                        "SKU": "MATCH",
                        "Name": "Inventory name",
                        "On hand": 20,
                        "Reorder Point": 7,
                        "Latitude": -33.86,
                        "Longitude": 151.2,
                    }
                ]
            )
        )
        product_upload = parse_product_table(
            pd.DataFrame(
                [
                    {
                        "SKU": "MATCH",
                        "Product Name": "Master name",
                        "Product Category": "Coffee",
                    },
                    {
                        "SKU": "NEW",
                        "Product Name": "New item",
                        "Product Category": "Accessories",
                    },
                ]
            )
        )

        merged = merge_inventory_uploads(existing, uploaded_inventory, product_upload)

        self.assertEqual([item.sku for item in merged], ["MATCH", "NEW"])
        self.assertEqual(merged[0].name, "Master name")
        self.assertEqual(merged[0].category, "Coffee")
        self.assertEqual(merged[0].on_hand, 20)
        self.assertEqual(merged[0].warehouse_latitude, -33.86)
        self.assertEqual(merged[0].warehouse_longitude, 151.2)
        self.assertEqual(merged[1].on_hand, 0)
        self.assertEqual(merged[1].category, "Accessories")

    def test_supplier_upload_replaces_stale_offer_and_drives_workflow(self) -> None:
        inventory = (
            InventoryItem("SKU-1", "Widget", 0, 2, 1.0, 10, 5, 35.68, 139.69),
        )
        old_offer = Document(
            "old-offer",
            "old.csv",
            "price_list",
            "Old Supplier price for SKU-1",
            {"sku": "SKU-1", "supplier": "Old Supplier", "unit_price": 99},
        )
        offers = parse_supplier_table(
            pd.DataFrame(
                [
                    {
                        "SKU": "SKU-1",
                        "Vendor": "New Supplier",
                        "Price": 3.5,
                        "Currency": "USD",
                        "Lead Time": 4,
                        "MOQ": 1,
                        "Latitude": 40.71,
                        "Longitude": -74.0,
                    }
                ]
            )
        )
        documents = merge_supplier_documents((old_offer,), offers, "suppliers.csv")
        store = LocalDocumentStore()
        for document in documents:
            store.add(document)

        result = SupplyChainAssistant(store).run(
            inventory,
            cash_balance=1000,
            minimum_cash_reserve=0,
        )

        self.assertEqual(len(documents), 1)
        self.assertEqual(result.recommendations[0].offer.supplier, "New Supplier")
        self.assertEqual(result.recommendations[0].offer.unit_price, 3.5)
        self.assertEqual(result.recommendations[0].offer.latitude, 40.71)
        supplier_points, warehouse_points, routes = build_supply_chain_map_data(
            inventory,
            result.forecasts,
            result.recommendations,
            documents,
        )
        self.assertEqual(supplier_points[0]["name"], "New Supplier")
        self.assertEqual(warehouse_points[0]["risk"], "High")
        self.assertEqual(routes[0]["sku"], "SKU-1")

    def test_invalid_required_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "on_hand is required"):
            parse_inventory_table(
                pd.DataFrame(
                    [{"SKU": "SKU-1", "On Hand": None, "Reorder Point": 1}]
                )
            )
        with self.assertRaisesRegex(ValueError, "at most 100"):
            parse_supplier_table(
                pd.DataFrame(
                    [
                        {
                            "SKU": "SKU-1",
                            "Supplier": "Vendor",
                            "Unit Price": 2,
                            "Discount": 120,
                        }
                    ]
                )
            )
        with self.assertRaisesRegex(ValueError, "Coordinates must be within"):
            parse_product_table(
                pd.DataFrame(
                    [
                        {
                            "SKU": "SKU-1",
                            "Name": "Invalid location",
                            "Warehouse Latitude": 91,
                            "Warehouse Longitude": 20,
                        }
                    ]
                )
            )


if __name__ == "__main__":
    unittest.main()
