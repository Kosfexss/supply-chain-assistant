"""Focused regression tests for persistence, forecasting, and SMTP gating."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from supply_chain_assistant.agents import CostOptimizationRiskAgent, SupplyChainChatAgent
from supply_chain_assistant.database import Database
from supply_chain_assistant.email_dispatcher import EmailDispatcher, SMTPSettings
from supply_chain_assistant.forecasting import ForecastingAgent
from supply_chain_assistant.models import InventoryItem, PurchaseDraft
from supply_chain_assistant.orchestrator import SupplyChainAssistant
from supply_chain_assistant.rag import LocalDocumentStore
from supply_chain_assistant.sample_data import seed_demo_data
from supply_chain_assistant.scenario import simulate_inventory


class FakeSMTP:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.sent = False

    def __enter__(self) -> "FakeSMTP":
        return self

    def __exit__(self, *args: object) -> bool:
        del args
        return False

    def ehlo(self) -> None:
        pass

    def starttls(self, **kwargs: object) -> None:
        del kwargs

    def login(self, *args: str) -> None:
        del args

    def send_message(self, message: object) -> None:
        del message
        self.sent = True


class FakeChatModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.system_prompt = ""

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        if "Current supply-chain snapshot:" not in user_prompt:
            raise AssertionError("Chat generation must include the current snapshot.")
        return self.response


class SupplyChainCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temporary_directory.name) / "test.sqlite3")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_seeded_documents_are_persisted_and_searchable(self) -> None:
        seed_demo_data(self.database)
        seed_demo_data(self.database)
        self.assertEqual(len(self.database.list_documents()), 18)
        self.assertEqual(len(self.database.list_inventory()), 14)
        self.assertEqual(len(self.database.list_financial_kpis()), 1)
        self.assertEqual(len(self.database.list_fulfillment_records()), 3)
        self.assertEqual(len(self.database.list_supplier_deliveries()), 4)
        self.assertEqual(len(self.database.list_landed_cost_records()), 2)
        orders = self.database.list_orders()
        self.assertEqual(len(orders), 3)
        self.assertEqual(
            {order["status"] for order in orders},
            {"approved", "sent"},
        )
        self.assertTrue(all(order["is_demo"] for order in orders))
        self.assertTrue(
            all(order["recipient"].endswith(".example") for order in orders)
        )
        reliability_scores = [
            document.metadata["reliability_score"]
            for document in self.database.list_documents()
            if document.metadata.get("is_illustrative_demo")
        ]
        self.assertEqual(len(reliability_scores), 8)
        self.assertTrue(all(0 <= score <= 1 for score in reliability_scores))
        results = LocalDocumentStore(self.database).search(
            "COF-001 coffee supplier price", filters={"sku": "COF-001"}
        )
        self.assertTrue(results)
        self.assertEqual(results[0].document.category, "price_list")

    def test_demo_products_add_supplier_costs_and_map_locations(self) -> None:
        seed_demo_data(self.database)

        inventory_by_sku = {
            item.sku: item for item in self.database.list_inventory()
        }
        documents_by_sku = {
            document.metadata.get("sku"): document
            for document in self.database.list_documents()
            if document.category == "price_list"
        }
        added_skus = {"TEA-023", "SYR-014", "MUG-018", "BOX-024"}

        self.assertTrue(added_skus.issubset(inventory_by_sku))
        self.assertTrue(added_skus.issubset(documents_by_sku))
        for sku in added_skus:
            item = inventory_by_sku[sku]
            offer = documents_by_sku[sku]
            self.assertIsNotNone(item.category)
            self.assertEqual(item.warehouse_latitude, 41.8781)
            self.assertEqual(item.warehouse_longitude, -87.6298)
            self.assertGreater(offer.metadata["unit_price"], 0)
            self.assertTrue(offer.metadata["supplier"])
            self.assertIsNotNone(offer.metadata["latitude"])
            self.assertIsNotNone(offer.metadata["longitude"])

    def test_seeding_adds_missing_demo_records_without_overwriting_saved_inventory(self) -> None:
        seed_demo_data(self.database)
        existing_item = next(
            item for item in self.database.list_inventory() if item.sku == "COF-001"
        )
        updated_item = InventoryItem(
            existing_item.sku,
            existing_item.name,
            3,
            existing_item.reorder_point,
            existing_item.average_daily_usage,
            existing_item.target_stock,
            existing_item.supplier_lead_time_days,
            existing_item.warehouse_latitude,
            existing_item.warehouse_longitude,
            existing_item.category,
        )
        self.database.save_inventory_item(updated_item)

        seed_demo_data(self.database)

        refreshed_item = next(
            item for item in self.database.list_inventory() if item.sku == "COF-001"
        )
        self.assertEqual(refreshed_item.on_hand, 3)
        self.assertEqual(len(self.database.list_inventory()), 14)
        self.assertEqual(len(self.database.list_documents()), 18)
        self.assertEqual(len(self.database.list_orders()), 3)

    def test_demo_orders_are_added_without_replacing_operational_order_history(self) -> None:
        real_order = PurchaseDraft(
            "REAL-1", "Operational item", "Verified supplier", 2,
            5.0, 10.0, "USD", "Operational order", "Real order body", (),
        )
        self.database.create_approved_order(
            real_order,
            recipient="buyer@verified-supplier.test",
            approved_by="Business owner",
        )

        seed_demo_data(self.database)
        seed_demo_data(self.database)

        orders = self.database.list_orders()
        self.assertEqual(len(orders), 4)
        self.assertEqual(
            sum(not order["is_demo"] for order in orders),
            1,
        )
        operational_order = next(order for order in orders if not order["is_demo"])
        self.assertEqual(operational_order["sku"], "REAL-1")

    def test_kpi_schema_initializes_on_an_existing_database(self) -> None:
        legacy_path = Path(self.temporary_directory.name) / "legacy.sqlite3"
        with sqlite3.connect(legacy_path) as connection:
            connection.execute(
                """
                CREATE TABLE inventory_items (
                    sku TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    on_hand INTEGER NOT NULL,
                    reorder_point INTEGER NOT NULL,
                    average_daily_usage REAL NOT NULL,
                    target_stock INTEGER NOT NULL,
                    supplier_lead_time_days INTEGER NOT NULL DEFAULT 5
                )
                """
            )
            connection.execute(
                "INSERT INTO inventory_items VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("LEGACY", "Legacy item", 10, 2, 1.0, 20, 5),
            )
        connection.close()

        migrated_database = Database(legacy_path)

        self.assertEqual(migrated_database.list_inventory()[0].sku, "LEGACY")
        self.assertIsNone(migrated_database.list_inventory()[0].warehouse_latitude)
        self.assertEqual(migrated_database.list_financial_kpis(), ())
        self.assertEqual(migrated_database.list_landed_cost_records(), ())

    def test_inventory_warehouse_coordinates_round_trip(self) -> None:
        item = InventoryItem(
            "LOCATED",
            "Located warehouse stock",
            10,
            2,
            1.0,
            20,
            5,
            35.68,
            139.69,
            "Test category",
        )

        self.database.save_inventory_item(item)

        loaded_item = self.database.list_inventory()[0]
        self.assertEqual(loaded_item.warehouse_latitude, 35.68)
        self.assertEqual(loaded_item.warehouse_longitude, 139.69)
        self.assertEqual(loaded_item.category, "Test category")

    def test_rising_usage_triggers_proactive_order(self) -> None:
        item = InventoryItem("TREND", "Trend test", 80, 20, 5.0, 120, 5)
        history = tuple((str(day), float(day + 1)) for day in range(10))
        forecast = ForecastingAgent().forecast(item, history)
        self.assertEqual(forecast.forecast_daily_usage, 11.0)
        self.assertEqual(forecast.projected_stock_at_lead_time, 15.0)
        self.assertTrue(forecast.proactive_order)
        self.assertEqual(forecast.suggested_order_quantity, 105)

    def test_scenario_simulation_recalculates_inventory_risk_and_costs(self) -> None:
        item = InventoryItem("SIM-001", "Scenario test", 10, 2, 2.0, 20, 4)
        forecast = ForecastingAgent().forecast(item, ())

        baseline = simulate_inventory(
            (item,), (forecast,), {"SIM-001": (4.0, "USD")},
            100, 0, 0, 20,
        )[0]
        self.assertEqual(baseline.adjusted_lead_time_days, 4)
        self.assertEqual(baseline.projected_inventory, 2.0)
        self.assertFalse(baseline.stockout_risk)
        self.assertEqual(baseline.days_until_stockout, None)
        self.assertEqual(baseline.annual_holding_cost_change, -6.4)

        stressed = simulate_inventory(
            (item,), (forecast,), {"SIM-001": (4.0, "USD")},
            200, 25, 50, 20,
        )[0]
        self.assertEqual(stressed.adjusted_lead_time_days, 8)
        self.assertEqual(stressed.projected_inventory, 0.0)
        self.assertEqual(stressed.stockout_units, 10.0)
        self.assertTrue(stressed.stockout_risk)
        self.assertEqual(stressed.days_until_stockout, 4.0)
        self.assertEqual(stressed.stockout_cost_exposure, 40.0)
        self.assertEqual(stressed.annual_holding_cost_change, -8.0)

    def test_database_workflow_ranks_contract_and_blocks_over_budget(self) -> None:
        seed_demo_data(self.database)
        inventory = self.database.list_inventory()
        usage = {
            item.sku: self.database.get_usage_history(item.sku) for item in inventory
        }
        assistant = SupplyChainAssistant.from_database(self.database)
        approved = assistant.run(
            inventory,
            cash_balance=1500,
            minimum_cash_reserve=500,
            usage_history=usage,
        )
        coffee = next(
            recommendation
            for recommendation in approved.recommendations
            if recommendation.need.item.sku == "COF-001"
        )
        self.assertEqual(coffee.offer.supplier, "Harbor Roasters")
        self.assertEqual(coffee.total_cost, 571.14)
        self.assertTrue(approved.financial_assessment.approved)
        self.assertTrue(approved.purchase_drafts)

        blocked = assistant.run(
            inventory,
            cash_balance=600,
            minimum_cash_reserve=500,
            usage_history=usage,
        )
        self.assertFalse(blocked.financial_assessment.approved)
        self.assertFalse(blocked.purchase_drafts)

    def test_cost_optimization_report_uses_supplier_and_forecast_evidence(self) -> None:
        seed_demo_data(self.database)
        inventory = self.database.list_inventory()
        usage = {
            item.sku: self.database.get_usage_history(item.sku) for item in inventory
        }
        workflow = SupplyChainAssistant.from_database(self.database).run(
            inventory,
            cash_balance=1500,
            minimum_cash_reserve=500,
            usage_history=usage,
        )
        report = CostOptimizationRiskAgent().analyze(
            inventory,
            workflow.forecasts,
            workflow.recommendations,
            self.database.list_documents(),
            usage,
            self.database.list_financial_kpis(),
            self.database.list_fulfillment_records(),
            self.database.list_supplier_deliveries(),
            self.database.list_landed_cost_records(),
        )

        self.assertEqual(len(report.metrics), 4)
        self.assertEqual(report.metrics[0].value, "5.00x")
        self.assertTrue(report.metrics[0].available)
        self.assertEqual(report.metrics[1].value, "97.6%")
        self.assertTrue(report.metrics[1].available)
        self.assertIn("COF-001: USD 19.80", report.metrics[2].value)
        self.assertIn("CUP-012: USD 0.131", report.metrics[2].value)
        self.assertTrue(report.metrics[2].available)
        self.assertEqual(report.metrics[3].value, "50.0%")
        self.assertTrue(report.metrics[3].available)
        self.assertTrue(
            any(
                "Lower unit-price alternative listed for COF-001" == finding.title
                for finding in report.findings
            )
        )
        self.assertTrue(
            any("Routing savings cannot yet be quantified" == finding.title
                for finding in report.findings)
        )
        self.assertFalse(
            any(
                "delivery" in gap.lower() or "landed cost" in gap.lower()
                for gap in report.data_gaps
            )
        )
        chat_agent = SupplyChainChatAgent()
        stock_answer = chat_agent.respond(
            "What is the current stock for COF-001?",
            inventory=inventory,
            forecasts=workflow.forecasts,
            recommendations=workflow.recommendations,
            financial_assessment=workflow.financial_assessment,
            cost_report=report,
        )
        self.assertIn("COF-001", stock_answer)
        self.assertIn("on hand 8", stock_answer)
        self.assertTrue(stock_answer.startswith("**Direct answer:**"))
        self.assertIn("### Current stock", stock_answer)
        self.assertIn("- **COF-001", stock_answer)
        self.assertIn("### Recommended action", stock_answer)

        supplier_answer = chat_agent.respond(
            "Which supplier is recommended for COF-001?",
            inventory=inventory,
            forecasts=workflow.forecasts,
            recommendations=workflow.recommendations,
            financial_assessment=workflow.financial_assessment,
            cost_report=report,
        )
        self.assertIn("Harbor Roasters", supplier_answer)
        self.assertNotIn("GreenPack Wholesale", supplier_answer)

        risk_answer = chat_agent.respond(
            "What are the current KPIs and risks?",
            inventory=inventory,
            forecasts=workflow.forecasts,
            recommendations=workflow.recommendations,
            financial_assessment=workflow.financial_assessment,
            cost_report=report,
        )
        self.assertIn("Supplier OTIF: 50.0%", risk_answer)

        turkish_stock_answer = chat_agent.respond(
            "COF-001 stok durumu nedir?",
            inventory=inventory,
            forecasts=workflow.forecasts,
            recommendations=workflow.recommendations,
            financial_assessment=workflow.financial_assessment,
            cost_report=report,
        )
        self.assertIn("on hand 8", turkish_stock_answer)

        turkish_forecast_answer = chat_agent.respond(
            "COF-001 için tahmin nedir?",
            inventory=inventory,
            forecasts=workflow.forecasts,
            recommendations=workflow.recommendations,
            financial_assessment=workflow.financial_assessment,
            cost_report=report,
        )
        self.assertIn("Current usage and forecast", turkish_forecast_answer)
        self.assertIn("COF-001", turkish_forecast_answer)

        turkish_sourcing_answer = chat_agent.respond(
            "COF-001 için tedarikçi ve fiyat nedir?",
            inventory=inventory,
            forecasts=workflow.forecasts,
            recommendations=workflow.recommendations,
            financial_assessment=workflow.financial_assessment,
            cost_report=report,
        )
        self.assertIn("Harbor Roasters", turkish_sourcing_answer)
        self.assertIn("USD 18.40/unit", turkish_sourcing_answer)
        self.assertIn("USD 571.14", turkish_sourcing_answer)
        self.assertTrue(turkish_sourcing_answer.startswith("**Direct answer:**"))
        self.assertIn("**estimated total:**", turkish_sourcing_answer)
        self.assertIn("### Recommended action\n-", turkish_sourcing_answer)

        unclear_answer = chat_agent.respond(
            "Merhaba",
            inventory=inventory,
            forecasts=workflow.forecasts,
            recommendations=workflow.recommendations,
            financial_assessment=workflow.financial_assessment,
            cost_report=report,
        )
        self.assertIn("stock and reorder levels", unclear_answer)
        self.assertIn("KPIs or supply risks", unclear_answer)
        self.assertIn("### Recommended action", unclear_answer)

        chat_model = FakeChatModel("Inventory is low.\nSKU-1 needs review.")
        model_answer = SupplyChainChatAgent(chat_model).respond(
            "Summarize inventory.",
            inventory=inventory,
            forecasts=workflow.forecasts,
            recommendations=workflow.recommendations,
            financial_assessment=workflow.financial_assessment,
            cost_report=report,
        )
        self.assertTrue(model_answer.startswith("**Direct answer:**"))
        self.assertIn("- Inventory is low.", model_answer)
        self.assertIn("Respond in readable Markdown", chat_model.system_prompt)
        self.assertTrue(model_answer.rstrip().endswith(
            "the required human review before acting."
        ))

    def test_approved_order_dispatches_once_and_is_logged(self) -> None:
        draft = PurchaseDraft(
            "SKU-1", "Test product", "Supplier", 2, 5.0, 10.0,
            "USD", "Order inquiry", "Confirm availability", (),
        )
        order_id = self.database.create_approved_order(
            draft, recipient="buyer@supplier.test", approved_by="owner"
        )
        dispatcher = EmailDispatcher(
            self.database,
            SMTPSettings("localhost", 587, "buyer@business.test", None, None, False),
        )
        with patch("supply_chain_assistant.email_dispatcher.smtplib.SMTP", FakeSMTP):
            dispatcher.send_approved_order(order_id)
        self.assertEqual(self.database.get_order(order_id)["status"], "sent")
        with self.assertRaises(ValueError):
            dispatcher.send_approved_order(order_id)

    def test_sample_recipient_cannot_be_dispatched(self) -> None:
        draft = PurchaseDraft(
            "SKU-1", "Test product", "Supplier", 2, 5.0, 10.0,
            "USD", "Order inquiry", "Confirm availability", (),
        )
        order_id = self.database.create_approved_order(
            draft, recipient="buyer@supplier.example", approved_by="owner"
        )
        dispatcher = EmailDispatcher(
            self.database,
            SMTPSettings("localhost", 587, "buyer@business.test", None, None, False),
        )
        with self.assertRaisesRegex(ValueError, "sample .example"):
            dispatcher.send_approved_order(order_id)
        self.assertEqual(self.database.get_order(order_id)["status"], "approved")


if __name__ == "__main__":
    unittest.main()