"""Tests for explainable supply-chain anomaly detection."""

from __future__ import annotations

import unittest
from datetime import date, timedelta

from supply_chain_assistant.models import (
    Document,
    InventoryItem,
    LandedCostRecord,
    SupplierDeliveryRecord,
)
from supply_chain_assistant.risk_radar import detect_supply_chain_risks


class SupplyChainRiskRadarTests(unittest.TestCase):
    def test_detects_stock_thresholds_and_late_or_short_deliveries(self) -> None:
        today = date(2026, 10, 4)
        inventory = (
            InventoryItem("OUT", "Out of stock", 0, 5, 1.0, 10),
            InventoryItem("LOW", "Low stock", 3, 5, 1.0, 10),
            InventoryItem("OK", "Healthy stock", 20, 5, 1.0, 25),
        )
        deliveries = (
            SupplierDeliveryRecord(
                "late",
                "Late Supplier",
                "OUT",
                today - timedelta(days=8),
                None,
                10,
                0,
            ),
            SupplierDeliveryRecord(
                "short",
                "Short Supplier",
                "LOW",
                today - timedelta(days=2),
                today - timedelta(days=1),
                10,
                7,
            ),
            SupplierDeliveryRecord(
                "on-time",
                "Reliable Supplier",
                "OK",
                today - timedelta(days=1),
                today - timedelta(days=1),
                10,
                10,
            ),
        )

        risks = detect_supply_chain_risks(
            inventory, (), deliveries, (), (), today=today
        )

        self.assertEqual(
            {risk.category for risk in risks},
            {"Inventory", "Supplier delay", "Supplier short shipment"},
        )
        self.assertEqual(risks[0].severity, "Critical")
        self.assertIn("OUT", risks[0].title)
        self.assertTrue(all(risk.action for risk in risks))
        late_risk = next(risk for risk in risks if risk.category == "Supplier delay")
        self.assertEqual(late_risk.severity, "Critical")
        self.assertIn("8 days late", late_risk.title)

    def test_detects_failed_orders_and_price_above_current_supplier_quote(self) -> None:
        today = date(2026, 10, 4)
        orders = (
            {
                "order_id": 7,
                "sku": "SKU-1",
                "supplier": "Supplier A",
                "unit_price": 15.0,
                "currency": "USD",
                "status": "sent",
                "error": None,
            },
            {
                "order_id": 8,
                "sku": "SKU-2",
                "supplier": "Supplier B",
                "unit_price": 5.0,
                "currency": "USD",
                "status": "failed",
                "error": "SMTP unavailable",
            },
        )
        documents = (
            Document(
                "offer-1",
                "Supplier A",
                "price_list",
                "Current offer",
                {
                    "sku": "SKU-1",
                    "supplier": "Supplier A",
                    "unit_price": 10.0,
                    "currency": "USD",
                },
            ),
            Document(
                "offer-2",
                "Supplier B",
                "price_list",
                "Different-currency offer",
                {
                    "sku": "SKU-2",
                    "supplier": "Supplier B",
                    "unit_price": 1.0,
                    "currency": "EUR",
                },
            ),
        )

        risks = detect_supply_chain_risks(
            (),
            orders,
            (),
            (),
            documents,
            today=today,
        )

        self.assertEqual(
            {risk.category for risk in risks},
            {"Order execution", "Cost variance"},
        )
        price_risk = next(risk for risk in risks if risk.category == "Cost variance")
        self.assertIn("50% above", price_risk.title)
        self.assertEqual(price_risk.severity, "Critical")
        order_risk = next(risk for risk in risks if risk.category == "Order execution")
        self.assertIn("SMTP unavailable", order_risk.description)

    def test_detects_landed_cost_increase_only_for_comparable_currency(self) -> None:
        today = date(2026, 10, 4)
        landed_costs = (
            LandedCostRecord(
                "receipt-1", "SKU-1", "Supplier A", today - timedelta(days=30),
                10, 10.0, 0.0, 0.0, 0.0, "USD",
            ),
            LandedCostRecord(
                "receipt-2", "SKU-1", "Supplier A", today - timedelta(days=1),
                10, 13.0, 0.0, 0.0, 0.0, "USD",
            ),
            LandedCostRecord(
                "receipt-eur", "SKU-1", "Supplier A", today,
                10, 100.0, 0.0, 0.0, 0.0, "EUR",
            ),
        )

        risks = detect_supply_chain_risks(
            (),
            (),
            (),
            landed_costs,
            (),
            today=today,
        )

        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0].category, "Landed-cost variance")
        self.assertIn("increased 30%", risks[0].title)

    def test_no_anomalies_for_healthy_inventory_and_matched_prices(self) -> None:
        item = InventoryItem("SKU-1", "Healthy item", 20, 5, 1.0, 25)
        order = {
            "order_id": 3,
            "sku": "SKU-1",
            "supplier": "Supplier A",
            "unit_price": 10.0,
            "currency": "USD",
            "status": "sent",
            "error": None,
        }
        offer = Document(
            "offer",
            "Supplier A",
            "price_list",
            "Current offer",
            {
                "sku": "SKU-1",
                "supplier": "Supplier A",
                "unit_price": 10.0,
                "currency": "USD",
            },
        )

        risks = detect_supply_chain_risks((item,), (order,), (), (), (offer,))

        self.assertEqual(risks, ())


if __name__ == "__main__":
    unittest.main()
