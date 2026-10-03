"""Coordinates the four agents without sending orders or supplier messages."""

from pathlib import Path

from .agents import (
    FinancialRiskAgent,
    InventoryTrackingAgent,
    MarketAndRAGAgent,
    NegotiationAndProposalAgent,
)
from .database import Database
from .forecasting import ForecastingAgent
from .llm import LLMClient
from .models import InventoryItem, ReplenishmentNeed, WorkflowResult
from .rag import LocalDocumentStore


class SupplyChainAssistant:
    def __init__(
        self, document_store: LocalDocumentStore, llm_client: LLMClient | None = None
    ) -> None:
        self.inventory_agent = InventoryTrackingAgent()
        self.forecasting_agent = ForecastingAgent()
        self.market_agent = MarketAndRAGAgent(document_store)
        self.financial_agent = FinancialRiskAgent()
        self.proposal_agent = NegotiationAndProposalAgent(llm_client)

    @classmethod
    def from_document_directory(cls, path: str | Path) -> "SupplyChainAssistant":
        store = LocalDocumentStore()
        store.ingest_directory(path)
        return cls(store)

    @classmethod
    def from_database(
        cls, database: Database, llm_client: LLMClient | None = None
    ) -> "SupplyChainAssistant":
        return cls(LocalDocumentStore(database), llm_client)

    def run(
        self,
        inventory: tuple[InventoryItem, ...],
        *,
        cash_balance: float,
        minimum_cash_reserve: float,
        usage_history: dict[str, tuple[tuple[str, float], ...]] | None = None,
    ) -> WorkflowResult:
        urgent_needs = self.inventory_agent.run(inventory)
        forecasts = self.forecasting_agent.run(inventory, usage_history or {})
        needs_by_sku = {need.item.sku: need for need in urgent_needs}
        for item, forecast in zip(inventory, forecasts):
            if forecast.proactive_order and item.sku not in needs_by_sku:
                needs_by_sku[item.sku] = ReplenishmentNeed(
                    item=item,
                    quantity=forecast.suggested_order_quantity,
                    days_until_stockout=forecast.days_until_reorder_point,
                    trigger="usage_forecast",
                )
        low_stock = tuple(needs_by_sku.values())
        recommendations, unmatched_skus = self.market_agent.run(low_stock)
        assessment = self.financial_agent.assess(
            recommendations,
            cash_balance=cash_balance,
            minimum_cash_reserve=minimum_cash_reserve,
        )
        drafts = (
            tuple(self.proposal_agent.draft(item) for item in recommendations)
            if assessment.approved
            else ()
        )
        return WorkflowResult(
            low_stock=low_stock,
            forecasts=forecasts,
            recommendations=recommendations,
            unmatched_skus=unmatched_skus,
            financial_assessment=assessment,
            purchase_drafts=drafts,
        )