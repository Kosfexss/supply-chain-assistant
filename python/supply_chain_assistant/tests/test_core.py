"""Focused regression tests for persistence, forecasting, and SMTP gating."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from supply_chain_assistant.database import Database
from supply_chain_assistant.email_dispatcher import EmailDispatcher, SMTPSettings
from supply_chain_assistant.forecasting import ForecastingAgent
from supply_chain_assistant.models import InventoryItem, PurchaseDraft
from supply_chain_assistant.orchestrator import SupplyChainAssistant
from supply_chain_assistant.rag import LocalDocumentStore
from supply_chain_assistant.sample_data import seed_demo_data


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


class SupplyChainCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temporary_directory.name) / "test.sqlite3")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_seeded_documents_are_persisted_and_searchable(self) -> None:
        seed_demo_data(self.database)
        seed_demo_data(self.database)
        self.assertEqual(len(self.database.list_documents()), 6)
        results = LocalDocumentStore(self.database).search(
            "COF-001 coffee supplier price", filters={"sku": "COF-001"}
        )
        self.assertTrue(results)
        self.assertEqual(results[0].document.category, "price_list")

    def test_rising_usage_triggers_proactive_order(self) -> None:
        item = InventoryItem("TREND", "Trend test", 80, 20, 5.0, 120, 5)
        history = tuple((str(day), float(day + 1)) for day in range(10))
        forecast = ForecastingAgent().forecast(item, history)
        self.assertEqual(forecast.forecast_daily_usage, 11.0)
        self.assertEqual(forecast.projected_stock_at_lead_time, 15.0)
        self.assertTrue(forecast.proactive_order)
        self.assertEqual(forecast.suggested_order_quantity, 105)

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