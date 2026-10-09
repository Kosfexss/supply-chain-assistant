"""Intent-routed specialist agents for questions about the current dashboard snapshot."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from .agents import (
    AgentToolRegistry,
    AgentResponseCritic,
    CostOptimizationReport,
    SharedBlackboard,
    SupplyChainChatAgent,
)
from .models import (
    FinancialAssessment,
    InventoryItem,
    StockForecast,
    SupplierRecommendation,
)


@dataclass(frozen=True)
class SupplyChainChatContext:
    inventory: tuple[InventoryItem, ...]
    forecasts: tuple[StockForecast, ...]
    recommendations: tuple[SupplierRecommendation, ...]
    financial_assessment: FinancialAssessment
    cost_report: CostOptimizationReport
    tool_registry: AgentToolRegistry | None = None


@dataclass(frozen=True)
class ChatDispatchResult:
    response: str
    agents: tuple[str, ...]
    tool_calls: tuple["AgentToolCall", ...] = ()


@dataclass(frozen=True)
class AgentToolCall:
    name: str
    agent: str
    output: str
    succeeded: bool


class _SpecialistAgent(Protocol):
    name: str

    def respond(self, message: str, context: SupplyChainChatContext) -> str:
        ...


def _normalize(text: str) -> str:
    return (
        text.casefold()
        .translate(str.maketrans({"ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c"}))
    )


def _contains_term(message: str, terms: tuple[str, ...]) -> bool:
    normalized = _normalize(message)
    return any(
        re.search(rf"(?<!\w){re.escape(_normalize(term))}(?!\w)", normalized)
        for term in terms
    )


def _contains_verb(message: str, verbs: tuple[str, ...]) -> bool:
    normalized = _normalize(message)
    return any(
        re.search(rf"(?<!\w){re.escape(_normalize(verb))}\w*", normalized)
        for verb in verbs
    )


def _matching_skus(
    message: str,
    inventory: tuple[InventoryItem, ...],
) -> set[str]:
    normalized = _normalize(message)
    return {
        item.sku
        for item in inventory
        if _normalize(item.sku) in normalized
        or _normalize(item.name) in normalized
    }


class InventoryAgent:
    """Answers product stock, reorder, and forecast questions from current data."""

    name = "Inventory Agent"

    def respond(self, message: str, context: SupplyChainChatContext) -> str:
        matching_skus = _matching_skus(message, context.inventory)
        items = [
            item for item in context.inventory
            if not matching_skus or item.sku in matching_skus
        ]
        if not items:
            return "No inventory records match the current question and dashboard filters."

        forecasts = {
            forecast.sku: forecast for forecast in context.forecasts
        }
        lines = []
        for item in items:
            line = (
                f"{item.sku} ({item.name}): {item.on_hand} units on hand; "
                f"reorder point {item.reorder_point}; target {item.target_stock}; "
                f"average daily usage {item.average_daily_usage:g}; "
                f"lead time {item.supplier_lead_time_days} days."
            )
            forecast = forecasts.get(item.sku)
            if forecast is not None:
                line += (
                    f" Forecast daily usage {forecast.forecast_daily_usage:.2f}; "
                    f"projected stock at lead time "
                    f"{forecast.projected_stock_at_lead_time:.2f}; "
                    f"proactive order {'recommended' if forecast.proactive_order else 'not indicated'}."
                )
            lines.append(line)
        return "Current inventory:\n" + "\n".join(lines)


class FinancialCostAgent:
    """Summarizes spend, cash availability, and cost-analysis metrics."""

    name = "Financial/Cost Agent"

    def respond(self, message: str, context: SupplyChainChatContext) -> str:
        matching_skus = _matching_skus(message, context.inventory)
        assessment = context.financial_assessment
        lines = [
            (
                f"Purchase assessment: "
                f"{'approved' if assessment.approved else 'blocked'}; requested "
                f"{assessment.currency} {assessment.requested_amount:,.2f}; "
                f"available after reserve {assessment.currency} "
                f"{assessment.spendable_cash:,.2f}. {assessment.reason}"
            )
        ]
        if context.recommendations:
            recommendations = [
                recommendation
                for recommendation in context.recommendations
                if not matching_skus
                or recommendation.need.item.sku in matching_skus
            ]
            if recommendations:
                lines.append(
                    "Current recommended purchase costs:\n"
                    + "\n".join(
                        f"{item.need.item.sku}: {item.offer.currency} "
                        f"{item.total_cost:,.2f} from {item.offer.supplier}."
                        for item in recommendations
                    )
                )
        metrics = [
            f"{metric.name}: {metric.value} — {metric.detail}"
            for metric in context.cost_report.metrics
        ]
        if metrics:
            lines.append("Cost and financial KPIs:\n" + "\n".join(metrics))

        findings = [
            finding
            for finding in context.cost_report.findings
            if any(
                term in _normalize(f"{finding.area} {finding.title}")
                for term in ("cost", "financial", "bulk", "price", "inventory")
            )
            and (
                not matching_skus
                or any(
                    sku.casefold() in (
                        f"{finding.title} {finding.evidence} {finding.action}"
                    ).casefold()
                    for sku in matching_skus
                )
            )
        ]
        if findings:
            lines.append(
                "Cost findings:\n"
                + "\n".join(
                    f"{finding.priority} {finding.title}: {finding.evidence} "
                    f"Action: {finding.action}"
                    for finding in findings
                )
            )
        return "\n\n".join(lines)


class SupplierRiskAgent:
    """Answers supplier performance, sourcing, and supply-risk questions."""

    name = "Supplier Risk Agent"

    def respond(self, message: str, context: SupplyChainChatContext) -> str:
        matching_skus = _matching_skus(message, context.inventory)
        recommendations = [
            recommendation
            for recommendation in context.recommendations
            if not matching_skus
            or recommendation.need.item.sku in matching_skus
        ]
        lines = []
        if recommendations:
            lines.append(
                "Supplier recommendations:\n"
                + "\n".join(
                    f"{item.need.item.sku}: {item.offer.supplier}, "
                    f"{item.order_quantity} units at {item.offer.currency} "
                    f"{item.offer.unit_price:.2f}/unit "
                    f"({item.offer.lead_time_days}-day lead time)."
                    for item in recommendations
                )
            )
        risk_metrics = [
            f"{metric.name}: {metric.value} — {metric.detail}"
            for metric in context.cost_report.metrics
            if any(
                term in _normalize(metric.name)
                for term in ("supplier", "otif", "fill rate")
            )
        ]
        if risk_metrics:
            lines.append("Supplier performance:\n" + "\n".join(risk_metrics))
        findings = [
            finding
            for finding in context.cost_report.findings
            if any(
                term in _normalize(f"{finding.area} {finding.title}")
                for term in ("supplier", "supply", "risk", "demand")
            )
            and (
                not matching_skus
                or any(
                    sku.casefold() in (
                        f"{finding.title} {finding.evidence} {finding.action}"
                    ).casefold()
                    for sku in matching_skus
                )
            )
        ]
        if findings:
            lines.append(
                "Supply-risk findings:\n"
                + "\n".join(
                    f"{finding.priority} {finding.area}: {finding.title}. "
                    f"{finding.evidence} Action: {finding.action}"
                    for finding in findings
                )
            )
        if not lines:
            return "No supplier recommendations or matching supplier-risk records are available."
        return "\n\n".join(lines)


class LogisticsAgent:
    """Summarizes supplier lead times and stock coverage during transit."""

    name = "Logistics Agent"

    def respond(self, message: str, context: SupplyChainChatContext) -> str:
        matching_skus = _matching_skus(message, context.inventory)
        recommendations = [
            recommendation
            for recommendation in context.recommendations
            if not matching_skus
            or recommendation.need.item.sku in matching_skus
        ]
        forecasts = {
            forecast.sku: forecast
            for forecast in context.forecasts
            if not matching_skus or forecast.sku in matching_skus
        }

        lines = []
        if recommendations:
            lines.append(
                "Recommended shipment timing:\n"
                + "\n".join(
                    f"{item.need.item.sku}: {item.offer.supplier} quotes a "
                    f"{item.offer.lead_time_days}-day lead time for "
                    f"{item.order_quantity} units."
                    for item in recommendations
                )
            )
        if forecasts:
            lines.append(
                "Stock coverage at lead time:\n"
                + "\n".join(
                    f"{forecast.sku}: projected stock "
                    f"{forecast.projected_stock_at_lead_time:.2f}; "
                    f"{'replenishment is indicated' if forecast.proactive_order else 'no proactive order is indicated'}."
                    for forecast in forecasts.values()
                )
            )
        if not recommendations and not forecasts:
            lines.append(
                "No supplier lead-time recommendations or matching forecasts "
                "are available."
            )
        return "\n\n".join(lines)


class MultiAgentChatDispatcher:
    """Route chat questions to one or more domain specialists."""

    _AGENT_DEPARTMENTS = {
        "Inventory Agent": "Inventory",
        "Financial/Cost Agent": "Financial",
        "Supplier Risk Agent": "Risk",
        "Logistics Agent": "Logistics",
    }

    _INTENTS = (
        (
            "Inventory Agent",
            (
                "inventory", "stock", "stockout", "reorder", "forecast",
                "demand", "usage", "on hand", "stok", "envanter", "tahmin",
                "talep", "tuketim", "depo",
            ),
        ),
        (
            "Financial/Cost Agent",
            (
                "financial", "finance", "budget", "cash", "spend", "cost",
                "price", "value", "purchase", "afford", "cash limit",
                "purchase cost", "margin", "bütçe",
                "butce", "nakit", "bakiye", "maliyet", "fiyat", "harcama",
            ),
        ),
        (
            "Supplier Risk Agent",
            (
                "supplier", "vendor", "otif", "risk", "delivery", "late",
                "performance", "sourcing", "tedarikci", "gecikme", "tedarik",
            ),
        ),
        (
            "Logistics Agent",
            (
                "logistics", "lead time", "lead-time", "transit", "shipment",
                "shipping", "delivery time", "arrival", "route", "lojistik",
                "teslimat suresi",
            ),
        ),
    )

    def __init__(
        self,
        fallback: SupplyChainChatAgent | None = None,
        blackboard: SharedBlackboard | None = None,
    ) -> None:
        self.inventory_agent = InventoryAgent()
        self.financial_cost_agent = FinancialCostAgent()
        self.supplier_risk_agent = SupplierRiskAgent()
        self.logistics_agent = LogisticsAgent()
        self.fallback = fallback or SupplyChainChatAgent()
        self.blackboard = blackboard or SharedBlackboard()
        self._agents: dict[str, _SpecialistAgent] = {
            self.inventory_agent.name: self.inventory_agent,
            self.financial_cost_agent.name: self.financial_cost_agent,
            self.supplier_risk_agent.name: self.supplier_risk_agent,
            self.logistics_agent.name: self.logistics_agent,
        }

    def route(self, message: str) -> tuple[str, ...]:
        if not message.strip():
            raise ValueError("Chat message must not be empty.")
        return tuple(
            agent_name
            for agent_name, terms in self._INTENTS
            if _contains_term(message, terms)
        )

    def dispatch(
        self,
        message: str,
        *,
        context: SupplyChainChatContext,
        history: tuple[tuple[str, str], ...] = (),
    ) -> ChatDispatchResult:
        routed_agents = self.route(message)
        self._record_snapshot(context)
        tool_calls = self._invoke_requested_tools(message, context)
        tool_agents = tuple(call.agent for call in tool_calls)
        agent_names = tuple(dict.fromkeys((*routed_agents, *tool_agents)))
        departments = tuple(
            self._AGENT_DEPARTMENTS[agent_name]
            for agent_name in routed_agents
            if agent_name in self._AGENT_DEPARTMENTS
        )
        shared_context = self.blackboard.format_context(
            excluding_departments=departments,
        )
        if not routed_agents and not tool_calls:
            response = self.fallback.respond(
                message,
                inventory=context.inventory,
                forecasts=context.forecasts,
                recommendations=context.recommendations,
                financial_assessment=context.financial_assessment,
                cost_report=context.cost_report,
                history=history,
                shared_context=shared_context,
            )
            return ChatDispatchResult(
                self._append_shared_context(response, shared_context),
                ("Supply Chain Chat Agent",),
            )

        responses = self._query_specialists_in_parallel(message, context, routed_agents)
        responses = tuple(
            AgentResponseCritic.refine(
                response,
                required_facts=self._required_response_facts(
                    agent_name,
                    message,
                    context,
                ),
                regenerate=lambda agent_name=agent_name: self._agents[
                    agent_name
                ].respond(message, context),
            )
            for agent_name, response in zip(routed_agents, responses)
        )
        for agent_name, response in zip(routed_agents, responses):
            department = self._AGENT_DEPARTMENTS[agent_name]
            self.blackboard.log(department, response, kind="insight")
        if responses:
            if len(responses) > 1:
                formatted_response = self._synthesize_responses(
                    message,
                    context,
                    routed_agents,
                    responses,
                )
            else:
                formatted_response = SupplyChainChatAgent._ensure_markdown_response(
                    responses[0]
                )
        else:
            formatted_response = "**Direct answer:** The requested tool action was processed."
        for tool_call in tool_calls:
            department = {
                "InventoryTrackingAgent": "Inventory",
                "NegotiationAndProposalAgent": "Purchasing",
                "CostOptimizationRiskAgent": "Risk",
            }.get(tool_call.agent, "Risk")
            self.blackboard.log(
                department,
                f"{tool_call.name}: {tool_call.output}",
                kind="insight" if tool_call.succeeded else "finding",
            )
        formatted_response = self._append_shared_context(
            formatted_response,
            shared_context,
        )
        if tool_calls:
            tool_results = "\n\n".join(
                f"**{call.name}** ({'completed' if call.succeeded else 'not run'}): "
                f"{call.output}"
                for call in tool_calls
            )
            formatted_response = self._append_section_before_recommendation(
                formatted_response,
                "### Tool results",
                tool_results,
            )
        return ChatDispatchResult(formatted_response, agent_names, tool_calls)

    def _required_response_facts(
        self,
        agent_name: str,
        message: str,
        context: SupplyChainChatContext,
    ) -> tuple[str, ...]:
        matching_skus = _matching_skus(message, context.inventory)
        facts: list[str] = []

        if agent_name == InventoryAgent.name:
            items = [
                item
                for item in context.inventory
                if not matching_skus or item.sku in matching_skus
            ]
            for item in items:
                facts.append(
                    f"{item.sku} ({item.name}): {item.on_hand} units on hand; "
                    f"reorder point {item.reorder_point}; target {item.target_stock}; "
                    f"average daily usage {item.average_daily_usage:g}; "
                    f"lead time {item.supplier_lead_time_days} days."
                )
                forecast = next(
                    (
                        item_forecast
                        for item_forecast in context.forecasts
                        if item_forecast.sku == item.sku
                    ),
                    None,
                )
                if forecast is not None:
                    facts.append(
                        f"Forecast daily usage {forecast.forecast_daily_usage:.2f}; "
                        f"projected stock at lead time "
                        f"{forecast.projected_stock_at_lead_time:.2f}; "
                        f"proactive order "
                        f"{'recommended' if forecast.proactive_order else 'not indicated'}."
                    )
            if not facts:
                facts.append(
                    "No inventory records match the current question and dashboard filters."
                )
        elif agent_name == FinancialCostAgent.name:
            assessment = context.financial_assessment
            facts.extend(
                (
                    f"Purchase assessment: "
                    f"{'approved' if assessment.approved else 'blocked'}; requested "
                    f"{assessment.currency} {assessment.requested_amount:,.2f}; "
                    f"available after reserve {assessment.currency} "
                    f"{assessment.spendable_cash:,.2f}. {assessment.reason}",
                )
            )
            for recommendation in context.recommendations:
                if not matching_skus or recommendation.need.item.sku in matching_skus:
                    facts.append(
                        f"{recommendation.need.item.sku}: "
                        f"{recommendation.offer.currency} {recommendation.total_cost:,.2f} "
                        f"from {recommendation.offer.supplier}."
                    )
            facts.extend(
                f"{metric.name}: {metric.value} — {metric.detail}"
                for metric in context.cost_report.metrics
            )
        elif agent_name == SupplierRiskAgent.name:
            for recommendation in context.recommendations:
                if not matching_skus or recommendation.need.item.sku in matching_skus:
                    facts.append(
                        f"{recommendation.need.item.sku}: {recommendation.offer.supplier}, "
                        f"{recommendation.order_quantity} units at "
                        f"{recommendation.offer.currency} "
                        f"{recommendation.offer.unit_price:.2f}/unit "
                        f"({recommendation.offer.lead_time_days}-day lead time)."
                    )
            facts.extend(
                f"{metric.name}: {metric.value} — {metric.detail}"
                for metric in context.cost_report.metrics
                if any(
                    term in _normalize(metric.name)
                    for term in ("supplier", "otif", "fill rate")
                )
            )
            facts.extend(
                f"{finding.priority} {finding.area}: {finding.title}. "
                f"{finding.evidence} Action: {finding.action}"
                for finding in context.cost_report.findings
                if any(
                    term in _normalize(f"{finding.area} {finding.title}")
                    for term in ("supplier", "supply", "risk", "demand")
                )
                and (
                    not matching_skus
                    or any(
                        sku.casefold() in (
                            f"{finding.title} {finding.evidence} {finding.action}"
                        ).casefold()
                        for sku in matching_skus
                    )
                )
            )
        elif agent_name == LogisticsAgent.name:
            for recommendation in context.recommendations:
                if not matching_skus or recommendation.need.item.sku in matching_skus:
                    facts.append(
                        f"{recommendation.need.item.sku}: {recommendation.offer.supplier} "
                        f"quotes a {recommendation.offer.lead_time_days}-day lead time "
                        f"for {recommendation.order_quantity} units."
                    )
            facts.extend(
                f"{forecast.sku}: projected stock "
                f"{forecast.projected_stock_at_lead_time:.2f}; "
                f"{'replenishment is indicated' if forecast.proactive_order else 'no proactive order is indicated'}."
                for forecast in context.forecasts
                if not matching_skus or forecast.sku in matching_skus
            )
        return tuple(facts)

    def _query_specialists_in_parallel(
        self,
        message: str,
        context: SupplyChainChatContext,
        agent_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not agent_names:
            return ()
        with ThreadPoolExecutor(max_workers=len(agent_names)) as executor:
            futures = tuple(
                executor.submit(self._agents[name].respond, message, context)
                for name in agent_names
            )
            return tuple(future.result() for future in futures)

    def _synthesize_responses(
        self,
        message: str,
        context: SupplyChainChatContext,
        agent_names: tuple[str, ...],
        responses: tuple[str, ...],
    ) -> str:
        matching_skus = _matching_skus(message, context.inventory)
        recommendations = tuple(
            item
            for item in context.recommendations
            if not matching_skus or item.need.item.sku in matching_skus
        )
        sections = [
            f"### {agent_name}\n{response.strip()}"
            for agent_name, response in zip(agent_names, responses)
        ]
        synthesis = self._cross_domain_summary(
            agent_names,
            context,
            recommendations,
        )
        recommendation = self._cross_domain_recommendation(
            agent_names,
            context,
            recommendations,
        )
        return (
            f"**Direct answer:** {synthesis}\n\n"
            + "\n\n".join(sections)
            + f"\n\n### Recommended action\n- {recommendation}"
        )

    @staticmethod
    def _cross_domain_summary(
        agent_names: tuple[str, ...],
        context: SupplyChainChatContext,
        recommendations: tuple[SupplierRecommendation, ...],
    ) -> str:
        domains = ", ".join(agent_names)
        statements = [f"This request was assessed across {domains}."]
        if "Financial/Cost Agent" in agent_names:
            assessment = context.financial_assessment
            status = "approved within the current cash limit" if assessment.approved else "blocked by the current cash limit"
            statements.append(
                f"The purchase assessment is {status}: requested "
                f"{assessment.currency} {assessment.requested_amount:,.2f} versus "
                f"{assessment.currency} {assessment.spendable_cash:,.2f} available "
                "after reserve."
            )
        if "Logistics Agent" in agent_names and recommendations:
            shipment = "; ".join(
                f"{item.need.item.sku} from {item.offer.supplier} has a "
                f"{item.offer.lead_time_days}-day quoted lead time"
                for item in recommendations
            )
            statements.append(f"Logistics reports {shipment}.")
        return " ".join(statements)

    @staticmethod
    def _cross_domain_recommendation(
        agent_names: tuple[str, ...],
        context: SupplyChainChatContext,
        recommendations: tuple[SupplierRecommendation, ...],
    ) -> str:
        if (
            "Financial/Cost Agent" in agent_names
            and not context.financial_assessment.approved
        ):
            if "Logistics Agent" in agent_names and recommendations:
                return (
                    "Resolve the cash shortfall before creating a purchase draft, "
                    "and confirm the quoted delivery timing against stock coverage."
                )
            return (
                "Review the cash shortfall and protected reserve before proceeding "
                "to purchase review."
            )
        if "Logistics Agent" in agent_names and recommendations:
            return (
                "Confirm the supplier's delivery date and ensure projected stock "
                "covers demand through the quoted lead time before review."
            )
        return (
            "Review the specialist findings together and complete the required "
            "human approval before taking operational action."
        )

    def _invoke_requested_tools(
        self,
        message: str,
        context: SupplyChainChatContext,
    ) -> tuple[AgentToolCall, ...]:
        requested: list[tuple[str, str, Mapping[str, object]]] = []
        if _contains_term(
            message,
            (
                "filter inventory", "filter the inventory", "filter stock",
                "show only", "list only", "show low stock", "low-stock items",
                "low stock items",
            ),
        ):
            matching_skus = _matching_skus(message, context.inventory)
            low_stock = _contains_term(
                message,
                (
                    "low stock", "below reorder", "under reorder point",
                    "reorder items", "low-stock", "dusuk stok",
                    "yenileme noktasi",
                ),
            )
            requested.append(
                (
                    "filter_inventory",
                    "InventoryTrackingAgent",
                    {
                        "sku": next(iter(matching_skus)) if len(matching_skus) == 1 else "",
                        "low_stock": low_stock,
                    },
                )
            )

        if _contains_term(
            message,
            ("auto-po", "auto po", "automated purchase order", "automatic purchase order"),
        ) and _contains_verb(
            message,
            ("create", "generate", "run", "trigger", "start", "make", "oluştur", "baslat"),
        ):
            requested.append(
                ("create_auto_purchase_orders", "NegotiationAndProposalAgent", {})
            )

        if _contains_term(
            message,
            ("risk radar", "supply chain risks", "risk assessment"),
        ) and _contains_verb(
            message,
            ("refresh", "update", "run", "recalculate", "check", "yenile", "guncelle"),
        ):
            requested.append(("refresh_risk_radar", "CostOptimizationRiskAgent", {}))

        calls = []
        for name, agent, arguments in requested:
            if context.tool_registry is None:
                calls.append(
                    AgentToolCall(
                        name,
                        agent,
                        "No backend tool registry is configured; no action was taken.",
                        False,
                    )
                )
                continue
            try:
                output = context.tool_registry.invoke(
                    name,
                    agent=agent,
                    arguments=arguments,
                )
            except KeyError:
                calls.append(
                    AgentToolCall(
                        name,
                        agent,
                        "This backend tool is not configured; no action was taken.",
                        False,
                    )
                )
            else:
                calls.append(AgentToolCall(name, agent, output, True))
        return tuple(calls)

    def _record_snapshot(self, context: SupplyChainChatContext) -> None:
        for forecast in context.forecasts:
            if forecast.proactive_order:
                self.blackboard.log(
                    "Inventory",
                    f"{forecast.sku}: proactive replenishment recommended; "
                    f"projected stock at lead time "
                    f"{forecast.projected_stock_at_lead_time:.2f}.",
                    kind="alert",
                )

        for recommendation in context.recommendations:
            sku = recommendation.need.item.sku
            self.blackboard.log(
                "Purchasing",
                f"{sku}: recommend {recommendation.order_quantity} units from "
                f"{recommendation.offer.supplier} for "
                f"{recommendation.offer.currency} {recommendation.total_cost:,.2f}.",
                kind="insight",
            )
            self.blackboard.log(
                "Logistics",
                f"{sku}: recommended supplier lead time is "
                f"{recommendation.offer.lead_time_days} days.",
                kind="insight",
            )

        for finding in context.cost_report.findings:
            text = _normalize(f"{finding.area} {finding.title} {finding.evidence}")
            if any(
                term in text for term in ("delivery", "shipment", "logistic", "late")
            ):
                department = "Logistics"
            elif any(
                term in text for term in ("purchas", "procurement", "bulk order")
            ):
                department = "Purchasing"
            elif any(
                term in text for term in ("financial", "cash", "cost", "price")
            ):
                department = "Financial"
            elif any(term in text for term in ("inventory", "stock", "reorder")):
                department = "Inventory"
            else:
                department = "Risk"
            kind = (
                "alert"
                if finding.priority.casefold() in ("high", "critical")
                else "finding"
            )
            self.blackboard.log(
                department,
                f"{finding.title}: {finding.evidence} Action: {finding.action}",
                kind=kind,
            )

    @staticmethod
    def _append_shared_context(response: str, shared_context: str) -> str:
        if not shared_context:
            return response
        return MultiAgentChatDispatcher._append_section_before_recommendation(
            response,
            "### Recent shared context",
            shared_context,
        )

    @staticmethod
    def _append_section_before_recommendation(
        response: str,
        heading: str,
        content: str,
    ) -> str:
        recommendation_heading = "\n### Recommended action\n"
        before, separator, after = response.partition(recommendation_heading)
        if separator:
            return f"{before}\n\n{heading}\n{content}{separator}{after}"
        return f"{response}\n\n{heading}\n{content}"
