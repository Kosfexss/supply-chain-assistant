"""Tests for supplier score calculation and evidence coverage."""

from __future__ import annotations

import unittest
from datetime import date, timedelta

from supply_chain_assistant.models import Document, LandedCostRecord, SupplierDeliveryRecord
from supply_chain_assistant.supplier_scorecard import calculate_supplier_scores


class SupplierScorecardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.today = date(2026, 10, 4)
        self.deliveries = (
            SupplierDeliveryRecord(
                "a1", "Reliable", "SKU-1", self.today - timedelta(days=5),
                self.today - timedelta(days=5), 10, 10,
            ),
            SupplierDeliveryRecord(
                "a2", "Reliable", "SKU-1", self.today - timedelta(days=4),
                self.today - timedelta(days=4), 10, 10,
            ),
            SupplierDeliveryRecord(
                "b1", "Variable", "SKU-1", self.today - timedelta(days=5),
                self.today - timedelta(days=5), 10, 10,
            ),
            SupplierDeliveryRecord(
                "b2", "Variable", "SKU-1", self.today - timedelta(days=4),
                self.today - timedelta(days=2), 10, 10,
            ),
            SupplierDeliveryRecord(
                "c1", "At Risk", "SKU-1", self.today - timedelta(days=5),
                self.today - timedelta(days=15), 10, 8,
            ),
            SupplierDeliveryRecord(
                "c2", "At Risk", "SKU-1", self.today - timedelta(days=4),
                self.today - timedelta(days=4), 10, 10,
            ),
            SupplierDeliveryRecord(
                "future", "Future Only", "SKU-1", self.today + timedelta(days=5),
                None, 10, 0,
            ),
        )
        self.documents = tuple(
            Document(
                f"offer-{supplier}",
                supplier,
                "price_list",
                f"Offer from {supplier}",
                {
                    "sku": "SKU-1",
                    "supplier": supplier,
                    "unit_price": price,
                    "currency": "USD",
                },
            )
            for supplier, price in (
                ("Reliable", 8.0),
                ("Variable", 9.0),
                ("At Risk", 10.0),
            )
        )

    def test_scores_use_otif_consistency_and_comparable_cost_then_tier(self) -> None:
        scores = calculate_supplier_scores(
            self.deliveries,
            self.documents,
            today=self.today,
        )

        self.assertEqual(
            [score.supplier for score in scores],
            ["Reliable", "Variable", "At Risk"],
        )
        reliable, variable, at_risk = scores
        self.assertEqual(reliable.score, 100)
        self.assertEqual(reliable.tier, "Tier A")
        self.assertEqual(reliable.otif_rate, 1)
        self.assertEqual(reliable.lead_time_consistency, 100)
        self.assertEqual(reliable.cost_competitiveness, 100)
        self.assertEqual(reliable.observed_components, 3)
        self.assertAlmostEqual(variable.score, 69.7222222222)
        self.assertEqual(variable.tier, "Tier B")
        self.assertEqual(variable.lead_time_consistency, 90)
        self.assertEqual(at_risk.score, 57.5)
        self.assertEqual(at_risk.tier, "Tier C")
        self.assertEqual(at_risk.otif_rate, 0.5)
        self.assertEqual(at_risk.lead_time_consistency, 50)
        self.assertEqual(at_risk.cost_competitiveness, 80)

    def test_future_deliveries_are_not_counted_as_due_performance(self) -> None:
        scores = calculate_supplier_scores(
            self.deliveries,
            self.documents,
            today=self.today,
        )

        self.assertNotIn("Future Only", {score.supplier for score in scores})

    def test_costs_are_compared_only_for_the_same_currency(self) -> None:
        euro_offer = Document(
            "euro-offer",
            "Euro Supplier",
            "price_list",
            "Euro offer",
            {
                "sku": "SKU-1",
                "supplier": "Euro Supplier",
                "unit_price": 1.0,
                "currency": "EUR",
            },
        )
        euro_delivery = SupplierDeliveryRecord(
            "euro-delivery", "Euro Supplier", "SKU-1",
            self.today - timedelta(days=2),
            self.today - timedelta(days=2),
            10,
            10,
        )
        scores = calculate_supplier_scores(
            (euro_delivery,),
            self.documents + (euro_offer,),
            today=self.today,
        )

        euro_score = next(score for score in scores if score.supplier == "Euro Supplier")
        self.assertIsNone(euro_score.cost_competitiveness)
        self.assertEqual(euro_score.observed_components, 2)

    def test_latest_landed_cost_takes_precedence_over_supplier_quote(self) -> None:
        landed_costs = (
            LandedCostRecord(
                "old", "SKU-1", "Reliable", self.today - timedelta(days=10),
                10, 10, 0, 0, 0, "USD",
            ),
            LandedCostRecord(
                "new", "SKU-1", "Reliable", self.today - timedelta(days=1),
                10, 12, 0, 0, 0, "USD",
            ),
        )
        scores = calculate_supplier_scores(
            (),
            self.documents,
            landed_costs,
            today=self.today,
        )

        reliable = next(score for score in scores if score.supplier == "Reliable")
        self.assertEqual(reliable.cost_competitiveness, 75)

    def test_invalid_consistency_window_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_supplier_scores(
                self.deliveries,
                self.documents,
                today=self.today,
                consistency_window_days=0,
            )


if __name__ == "__main__":
    unittest.main()
