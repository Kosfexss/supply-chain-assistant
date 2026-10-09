"""Tests for intent routing and supply-chain chat specialists."""

from __future__ import annotations

import unittest
from dataclasses import replace
from threading import Barrier

from supply_chain_assistant.agents import (
    AgentToolRegistry,
    CostOptimizationFinding,
    CostOptimizationMetric,
    CostOptimizationReport,
)
from supply_chain_assistant.chat_agents import (
    FinancialCostAgent,
    InventoryAgent,
    LogisticsAgent,
    MultiAgentChatDispatcher,
    SupplierRiskAgent,
    SupplyChainChatContext,
)
from supply_chain_assistant.models import (
    FinancialAssessment,
    InventoryItem,
    ReplenishmentNeed,
    StockForecast,
    SupplierOffer,
    SupplierRecommendation,
)


class MultiAgentChatTests(unittest.TestCase):
    def setUp(self) -> None:
        item = InventoryItem(
            "SKU-1", "Coffee beans", 4, 5, 1.0, 20, category="Coffee"
        )
        offer = SupplierOffer(
            "SKU-1", "Reliable Supply", 3.5, "USD", 4, 1, "price list"
        )
        self.context = SupplyChainChatContext(
            inventory=(item,),
            forecasts=(
                StockForecast(
                    "SKU-1", "Coffee beans", 10, 0.0, 1.2, 0.5, 2.0, 16, True
                ),
            ),
            recommendations=(
                SupplierRecommendation(
                    ReplenishmentNeed(item, 16, 4.0),
                    offer,
                    16,
                    56.0,
                    (),
                ),
            ),
            financial_assessment=FinancialAssessment(
                False,
                56.0,
                40.0,
                "USD",
                "Purchase exceeds available cash after the protected reserve.",
            ),
            cost_report=CostOptimizationReport(
                metrics=(
                    CostOptimizationMetric(
                        "Supplier OTIF", "80.0%", "4 deliveries assessed", True
                    ),
                    CostOptimizationMetric(
                        "Landed cost", "SKU-1: USD 3.50", "Latest receipt", True
                    ),
                ),
                findings=(
                    CostOptimizationFinding(
                        "High",
                        "Supply risk",
                        "Delivery delay for SKU-1",
                        "The latest shipment was late.",
                        "Confirm the revised delivery date.",
                    ),
                    CostOptimizationFinding(
                        "Medium",
                        "Inventory",
                        "Cost exposure for SKU-1",
                        "The purchase exceeds available cash.",
                        "Review the order quantity.",
                    ),
                ),
                data_gaps=(),
            ),
        )
        self.dispatcher = MultiAgentChatDispatcher()

    def test_router_selects_inventory_specialist(self) -> None:
        result = self.dispatcher.dispatch(
            "What is the current stock and forecast for SKU-1?",
            context=self.context,
        )

        self.assertEqual(result.agents, (InventoryAgent.name,))
        self.assertTrue(result.response.startswith("**Direct answer:**"))
        self.assertIn("4 units on hand", result.response)
        self.assertIn("projected stock at lead time 0.50", result.response)
        self.assertIn("### Recommended action", result.response)

    def test_router_dispatches_compound_question_to_multiple_agents(self) -> None:
        result = self.dispatcher.dispatch(
            "What is the cost and supplier risk for SKU-1?",
            context=self.context,
        )

        self.assertEqual(
            result.agents,
            (FinancialCostAgent.name, SupplierRiskAgent.name),
        )
        self.assertIn("Purchase assessment: blocked", result.response)
        self.assertIn("Supplier OTIF: 80.0%", result.response)
        self.assertIn("Delivery delay for SKU-1", result.response)

    def test_financial_and_logistics_query_is_synthesized(self) -> None:
        result = self.dispatcher.dispatch(
            "Can we afford this purchase, and what is the logistics lead time for SKU-1?",
            context=self.context,
        )

        self.assertEqual(
            result.agents,
            (FinancialCostAgent.name, "Logistics Agent"),
        )
        self.assertIn("blocked by the current cash limit", result.response)
        self.assertIn("4-day quoted lead time", result.response)
        self.assertLess(
            result.response.index("blocked by the current cash limit"),
            result.response.index("### Financial/Cost Agent"),
        )
        self.assertIn("### Logistics Agent", result.response)
        self.assertIn("Resolve the cash shortfall", result.response)

    def test_multiple_specialists_are_queried_in_parallel(self) -> None:
        barrier = Barrier(2)

        class ConcurrentSpecialist:
            def __init__(
                self,
                delegate: FinancialCostAgent | LogisticsAgent,
            ) -> None:
                self.delegate = delegate
                self.name = delegate.name

            def respond(
                self,
                message: str,
                context: SupplyChainChatContext,
            ) -> str:
                if not message or not context.inventory:
                    raise AssertionError("The specialist received incomplete context.")
                barrier.wait(timeout=2)
                return self.delegate.respond(message, context)

        self.dispatcher._agents[FinancialCostAgent.name] = ConcurrentSpecialist(
            FinancialCostAgent()
        )
        self.dispatcher._agents["Logistics Agent"] = ConcurrentSpecialist(
            LogisticsAgent()
        )

        result = self.dispatcher.dispatch(
            "Review financial cost and logistics lead time",
            context=self.context,
        )

        self.assertIn("Purchase assessment: blocked", result.response)
        self.assertIn("4-day lead time", result.response)

    def test_critic_retries_inconsistent_specialist_response(self) -> None:
        class CorrectedInventorySpecialist:
            name = InventoryAgent.name

            def __init__(self) -> None:
                self.calls = 0

            def respond(
                self,
                _message: str,
                _context: SupplyChainChatContext,
            ) -> str:
                self.calls += 1
                if self.calls == 1:
                    return "SKU-1 (Coffee beans): 400 units on hand."
                return (
                    "Current inventory:\n"
                    "SKU-1 (Coffee beans): 4 units on hand; reorder point 5; "
                    "target 20; average daily usage 1; lead time 5 days.\n"
                    "Forecast daily usage 1.20; projected stock at lead time "
                    "0.50; proactive order recommended."
                )

        specialist = CorrectedInventorySpecialist()
        self.dispatcher._agents[InventoryAgent.name] = specialist

        result = self.dispatcher.dispatch(
            "What is the stock for SKU-1?",
            context=self.context,
        )

        self.assertEqual(specialist.calls, 2)
        self.assertIn("4 units on hand", result.response)
        self.assertNotIn("400 units on hand", result.response)

    def test_critic_uses_live_data_only_if_specialist_cannot_be_refined(self) -> None:
        class InconsistentInventorySpecialist:
            name = InventoryAgent.name

            def respond(
                self,
                _message: str,
                _context: SupplyChainChatContext,
            ) -> str:
                return "SKU-1 (Coffee beans): 400 units on hand."

        self.dispatcher._agents[InventoryAgent.name] = (
            InconsistentInventorySpecialist()
        )

        result = self.dispatcher.dispatch(
            "What is the stock for SKU-1?",
            context=self.context,
        )

        self.assertIn("Validated current data", result.response)
        self.assertIn("4 units on hand", result.response)
        self.assertNotIn("400 units on hand", result.response)

    def test_router_dispatches_turkish_inventory_query(self) -> None:
        result = self.dispatcher.dispatch(
            "SKU-1 stok durumu nedir?",
            context=self.context,
        )

        self.assertEqual(result.agents, (InventoryAgent.name,))
        self.assertIn("4 units on hand", result.response)

    def test_dispatcher_includes_recent_notes_from_other_departments(self) -> None:
        self.dispatcher.dispatch(
            "What is the current stock for SKU-1?",
            context=self.context,
        )

        result = self.dispatcher.dispatch(
            "What is the cash position?",
            context=self.context,
        )

        self.assertIn("### Recent shared context", result.response)
        self.assertIn("Inventory · insight", result.response)
        self.assertIn("4 units on hand", result.response)

    def test_inventory_filter_request_invokes_registered_backend_tool(self) -> None:
        registry = AgentToolRegistry()
        registry.register(
            "filter_inventory",
            agent="InventoryTrackingAgent",
            description="Filter inventory.",
            handler=lambda arguments: (
                f"filtered {arguments['sku']} low_stock={arguments['low_stock']}"
            ),
        )
        context = replace(self.context, tool_registry=registry)

        result = self.dispatcher.dispatch(
            "Filter the inventory to low stock for SKU-1",
            context=context,
        )

        self.assertEqual(result.tool_calls[0].name, "filter_inventory")
        self.assertTrue(result.tool_calls[0].succeeded)
        self.assertIn("filtered SKU-1 low_stock=True", result.response)

    def test_purchase_order_and_risk_requests_invoke_named_tools(self) -> None:
        registry = AgentToolRegistry()
        registry.register(
            "create_auto_purchase_orders",
            agent="NegotiationAndProposalAgent",
            description="Create pending order drafts.",
            handler=lambda _: "Created one pending draft; none sent.",
        )
        registry.register(
            "refresh_risk_radar",
            agent="CostOptimizationRiskAgent",
            description="Refresh current risks.",
            handler=lambda _: "Risk radar refreshed: no current risks.",
        )
        context = replace(self.context, tool_registry=registry)

        purchase_result = self.dispatcher.dispatch(
            "Generate the Auto-PO drafts",
            context=context,
        )
        risk_result = self.dispatcher.dispatch(
            "Refreshing the risk radar now",
            context=context,
        )

        self.assertEqual(
            purchase_result.tool_calls[0].name,
            "create_auto_purchase_orders",
        )
        self.assertIn("none sent", purchase_result.response)
        self.assertEqual(risk_result.tool_calls[0].name, "refresh_risk_radar")
        self.assertIn("no current risks", risk_result.response)

    def test_requested_tool_is_reported_unavailable_without_registry(self) -> None:
        result = self.dispatcher.dispatch(
            "Generate the Auto-PO drafts",
            context=self.context,
        )

        self.assertFalse(result.tool_calls[0].succeeded)
        self.assertIn("no action was taken", result.response)

    def test_tool_registry_rejects_unauthorized_agent(self) -> None:
        registry = AgentToolRegistry()
        registry.register(
            "filter_inventory",
            agent="InventoryTrackingAgent",
            description="Filter inventory.",
            handler=lambda _: "done",
        )

        with self.assertRaises(PermissionError):
            registry.invoke("filter_inventory", agent="NegotiationAndProposalAgent")

    def test_supplier_agent_answers_recommendation_question(self) -> None:
        response = SupplierRiskAgent().respond(
            "Which supplier is recommended for SKU-1?",
            self.context,
        )

        self.assertIn("Reliable Supply", response)
        self.assertIn("4-day lead time", response)

    def test_unmatched_intent_uses_legacy_chat_fallback(self) -> None:
        result = self.dispatcher.dispatch("Hello", context=self.context)

        self.assertEqual(result.agents, ("Supply Chain Chat Agent",))
        self.assertIn("stock and reorder levels", result.response)

    def test_empty_message_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.dispatcher.route("  ")


if __name__ == "__main__":
    unittest.main()
