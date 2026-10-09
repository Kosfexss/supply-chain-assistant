"""Tests for portfolio performance metrics and guarded suggestions."""

from __future__ import annotations

import unittest
from datetime import date, timedelta

from supply_chain_assistant.diversification import ProductDiversificationAgent
from supply_chain_assistant.models import (
    FinancialKPIRecord,
    FulfillmentRecord,
    InventoryItem,
    SupplierDeliveryRecord,
)


class FakeTextClient:
    def __init__(self) -> None:
        self.called = False

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        self.called = True
        if "untrusted evidence" not in system_prompt:
            raise AssertionError("Model instructions must mark business data untrusted.")
        if "Portfolio metrics:" not in user_prompt:
            raise AssertionError("Portfolio metrics must ground model commentary.")
        return "Validate demand and supplier readiness before a small pilot."


class ProductDiversificationTests(unittest.TestCase):
    def setUp(self) -> None:
        today = date.today()
        self.inventory = (
            InventoryItem(
                "GOOD-1", "Strong item", 100, 10, 2.0, 150,
                category="Core",
            ),
            InventoryItem(
                "WEAK-1", "Weak item", 20, 5, 0.2, 30,
                category="Legacy",
            ),
        )
        self.fulfillment = (
            FulfillmentRecord("f1", "GOOD-1", today - timedelta(days=4), 100, 98),
            FulfillmentRecord("f2", "GOOD-1", today - timedelta(days=2), 50, 50),
            FulfillmentRecord("f3", "WEAK-1", today - timedelta(days=1), 10, 6),
        )
        self.deliveries = (
            SupplierDeliveryRecord(
                "d1", "Reliable", "GOOD-1", today - timedelta(days=3),
                today - timedelta(days=3), 100, 100,
            ),
            SupplierDeliveryRecord(
                "d2", "Unreliable", "WEAK-1", today - timedelta(days=2),
                today - timedelta(days=1), 20, 20,
            ),
        )
        self.financials = (
            FinancialKPIRecord(
                today - timedelta(days=365),
                today - timedelta(days=1),
                "USD",
                1000.0,
                250.0,
            ),
        )

    def test_category_metrics_and_safety_recommendations(self) -> None:
        report = ProductDiversificationAgent().analyze(
            self.inventory,
            self.fulfillment,
            self.deliveries,
            self.financials,
        )

        core = next(item for item in report.categories if item.category == "Core")
        legacy = next(item for item in report.categories if item.category == "Legacy")
        self.assertEqual(core.fulfillment_events, 2)
        self.assertAlmostEqual(core.fill_rate or 0.0, 148 / 150)
        self.assertEqual(core.estimated_annual_turns, 7.3)
        self.assertEqual(core.supplier_otif, 1.0)
        self.assertEqual(legacy.fill_rate, 0.6)
        self.assertEqual(legacy.supplier_otif, 0.0)
        self.assertEqual(report.overall_inventory_turns, 4.0)
        self.assertAlmostEqual(report.overall_fill_rate, 154 / 160)
        self.assertEqual(report.overall_supplier_otif, 0.5)
        titles = {suggestion.title for suggestion in report.suggestions}
        self.assertIn("Pilot a complementary offer adjacent to Core", titles)
        self.assertIn("Resolve service issues before expanding Legacy", titles)
        self.assertIn("Qualify a backup source for Legacy", titles)
        self.assertIn("Review Legacy for rationalization", titles)
        self.assertTrue(all(suggestion.risk_guardrail for suggestion in report.suggestions))
        pilot = next(
            suggestion
            for suggestion in report.suggestions
            if suggestion.title.startswith("Pilot a complementary")
        )
        self.assertIn("approval process", pilot.risk_guardrail.lower())

    def test_optional_model_commentary_is_grounded_in_metrics(self) -> None:
        client = FakeTextClient()
        report = ProductDiversificationAgent(client).analyze(
            self.inventory,
            self.fulfillment,
            self.deliveries,
            self.financials,
        )

        self.assertTrue(client.called)
        self.assertEqual(
            report.llm_commentary,
            "Validate demand and supplier readiness before a small pilot.",
        )

    def test_missing_classification_and_activity_are_not_overstated(self) -> None:
        unclassified_inventory = (
            InventoryItem("UNKNOWN", "Unknown category item", 10, 2, 1.0, 20),
        )
        report = ProductDiversificationAgent().analyze(
            unclassified_inventory,
            (),
            (),
        )

        self.assertEqual(report.categories[0].category, "Unclassified")
        self.assertIsNone(report.categories[0].fill_rate)
        self.assertIsNone(report.categories[0].supplier_otif)
        self.assertIn(
            "Classify products before portfolio expansion",
            {suggestion.title for suggestion in report.suggestions},
        )


if __name__ == "__main__":
    unittest.main()
