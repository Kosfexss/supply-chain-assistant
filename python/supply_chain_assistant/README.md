# Small Business Supply Chain Assistant

An operational portfolio starter combining a multi-agent purchasing workflow, SQLite persistence, local RAG retrieval, trend-based inventory forecasts, Streamlit review screens, and explicitly approved SMTP dispatch.

## Start

From the parent `python` directory, create an environment, install the UI dependency, and start the dashboard:

```powershell
python -m venv supply_chain_assistant/.venv
supply_chain_assistant/.venv/Scripts/python.exe -m pip install -r supply_chain_assistant/requirements.txt
supply_chain_assistant/.venv/Scripts/python.exe -m streamlit run supply_chain_assistant/app.py
```

The database is created at `data/supply_chain.sqlite3` on first run. Override it with `SUPPLY_CHAIN_DB_PATH`. Missing sample documents, inventory, contacts, cash settings, sample order-log rows, and 30 days of usage history are seeded idempotently into SQLite; existing inventory values, documents, and operational order records are not overwritten. Order-log demo rows are marked as sample data and use reserved `.example` addresses; a sample `sent` status does not mean an email was sent. Demo supplier prices and map coordinates are illustrative, not verified commercial or delivery data. The demo is also available with `python -m supply_chain_assistant.demo` from the parent `python` directory.

## Project structure

```text
supply_chain_assistant/
  agents.py              Domain agents and the shared cross-department blackboard
  app.py                 Streamlit dashboard, approval workflow, and map
  auto_replenishment.py  Review-only automatic purchase-order drafts
  database.py            SQLite schema and repository methods
  email_dispatcher.py    TLS SMTP dispatch for approved orders
  forecasting.py         Linear-trend usage forecast
  llm.py                 Provider-neutral LLM protocol
  models.py              Shared workflow contracts
  orchestrator.py        Agent coordination and financial gating
  rag.py                 SQLite-backed document search
  sample_data.py         Idempotent seed data for a new database
  geospatial.py          Supplier, warehouse-risk, and route map data
  diversification.py     Category performance metrics and portfolio suggestions
  tests/                 Regression tests
```

## Workflow and safety

Inventory rules detect current reorder-point breaches. `ForecastingAgent` fits a simple linear trend to daily usage and proposes a proactive order when predicted stock at supplier lead time reaches the reorder point. The procurement section automatically creates an idempotent pending Auto-PO draft when on-hand stock is at or below the configured reorder point and a supplier quote is available; it chooses the lowest estimated total quote, observes its minimum order quantity, and estimates an expected delivery date from quoted lead time. A manual form can also save pending orders using a selected product/supplier offer, quantity, and expected date. Pending orders are drafts only: neither path approves, emails, or places an order. RAG retrieves price lists, contracts, invoices, and compliance records from SQLite; supplier selection compares total order cost including structured contract discounts. The finance agent checks the combined estimate against cash remaining after the protected reserve.

The dashboard displays the recommendation and draft for review. The **Cost & risk analysis** tab adds evidence-based inventory, demand, and supplier findings using current forecasts, price lists, contract terms, and usage history. It also calculates inventory turns from COGS and average inventory value, fill rate from requested and fulfilled units, supplier OTIF from due delivery records, and per-SKU landed cost from receipt unit price, freight, duties, and handling. Demo KPI inputs are seeded idempotently into dedicated SQLite tables; replace them with verified operational records for production decisions. Sending requires the explicit **Approve & send email** action, a passing financial assessment, a named approver, a configured SMTP service, and a real supplier address. Each approved/sent/failed order is recorded. Seeded `.example` contacts are blocked from dispatch. SMTP settings are read from environment variables; copy `.env.example` as a reference and set values in your shell or secret manager. Do not commit credentials.

The chat dispatcher keeps a bounded shared blackboard for the current Streamlit session. Inventory, financial, risk, logistics, and purchasing notes are recorded as findings, insights, or active alerts; routed answers include recent notes from other departments. Agents can write through `dispatcher.blackboard.log(...)`, review unresolved alerts with `active_alerts()`, and resolve them with `resolve_alert(entry_id)`.

Chat also supports allow-listed backend tools via `AgentToolRegistry`: requests such as “filter inventory to low stock,” “generate the Auto-PO drafts,” and “refresh the risk radar” invoke the corresponding inventory query, idempotent pending-order draft workflow, or latest-data risk analysis. Tools are injected for each current dashboard snapshot, tool ownership is checked before execution, and results are shown explicitly. Auto-PO drafts remain pending for human review; chat cannot approve or send purchase orders.

Compound chat questions are routed to all matching specialists concurrently, with results combined in one answer containing a cross-domain assessment, department findings, and an actionable recommendation. A financial-and-logistics response, for example, weighs the current cash gate together with supplier lead time and projected stock coverage.

The **What-if simulation** tab projects per-SKU inventory through adjusted lead times and demand surges without changing saved inventory or purchase approvals. Its sidebar controls adjust lead time, demand, and annual carrying cost. Stockout exposure is estimated at latest landed unit cost (or current recommendation price when no landed cost exists); annual holding-cost changes use a 20% default baseline carrying rate that can be adjusted in the sidebar. These estimates are deterministic and should be replaced with verified rates and prices for operational decisions.

### Session dataset uploads

Use the sidebar uploaders to provide CSV, XLSX, or XLS inventory, product, and supplier tables. Inventory uploads replace the inventory list for the current session. Product uploads update matching SKU names and any supplied stock fields, and add new products with unspecified stock values set to zero. Supplier uploads replace existing price-list and contract documents for the uploaded SKUs while retaining evidence for other SKUs; uploaded prices participate in normal supplier selection and financial review. Uploads are validated and kept in memory only: clearing an uploader or ending the session returns to saved database/sample data, and uploaded values are not persisted. Inventory editing is read-only while inventory or product uploads are active.

Inventory tables require `sku`, `on_hand`, and `reorder_point`; `name`, `average_daily_usage`, `target_stock`, and `supplier_lead_time_days` are optional. Product tables require `sku` and `name`, and accept the optional inventory fields. Supplier tables require `sku`, `supplier`, and `unit_price`; `currency`, `lead_time_days`, `minimum_order_quantity`, `discount_threshold_quantity`, and `discount_percentage` are optional. Column names are case-insensitive and common aliases such as `product_id`, `current_stock`, `vendor`, and `price` are recognized. Excel files use the first worksheet.

The **Geographic map** tab uses optional decimal-degree coordinates: inventory and product tables may include `warehouse_latitude` and `warehouse_longitude` (or generic `latitude` and `longitude`); supplier tables may include `supplier_latitude` and `supplier_longitude` (or generic `latitude` and `longitude`). Latitude/longitude pairs are required together and validated against geographic bounds. Suppliers are blue markers, warehouse markers show forecast stock-risk levels, and arcs connect recommended suppliers to warehouses when both endpoint locations are available. The expanded global sample catalog includes explicitly illustrative approximate supplier coordinates and reliability scores; these values are demo fixtures, not verified commercial or delivery data. The map prompts for location data when no mapped locations are supplied. Saved inventory coordinates are migrated into SQLite and retained when inventory rows are edited.

The **Product diversification** tab compares categories using SKU counts, recorded fulfillment-event frequency and units, fill rate, annualized demand-to-current-stock turns, and category-linked supplier OTIF. Add a `category` / `product_category` field to product or inventory uploads to classify SKUs; saved inventory also supports category editing. Suggestions are evidence-driven rules: they propose a tightly capped complementary pilot only from categories with observed demand and acceptable service metrics, flag low service or supplier performance before expansion, and frame rationalization as a review rather than an automatic replacement. Category turns are explicitly an estimate because category-level COGS and average inventory value are not recorded; portfolio turns use reported COGS divided by average inventory value. No external generative model is configured by default, so recommendations are deterministic; an optional text-generation client can provide guarded commentary without approving purchases.

### KPI source records

`Database.save_financial_kpi`, `save_fulfillment_record`, `save_supplier_delivery`, and `save_landed_cost_record` persist their corresponding typed records; matching `list_*` methods load them for analysis. Inventory turns use the latest period's COGS divided by average inventory value. Fill rate is fulfilled units capped at requested units divided by all requested units. OTIF is the share of deliveries due by today that arrived by the promised date and met or exceeded the promised quantity. Landed cost is calculated per SKU and currency, weighted by receipt quantity, as unit price plus freight, duties, and handling per unit. When source records are missing, the dashboard continues to mark only that metric unavailable.

## RAG and LLM

`LocalDocumentStore` reads persisted `Document` rows and retains its deterministic TF-IDF query interface. New documents can be written with `Database.save_document`. The dashboard chat uses a local snapshot-based responder by default; `SupplyChainChatAgent` also accepts an optional `LLMClient` for conversational answers grounded in the current workflow context. The same provider-neutral `LLMClient` can be injected into `SupplyChainAssistant` to draft supplier inquiries. A model may draft wording and answer questions but never controls inventory decisions, financial approval, or dispatch.

The dashboard chat is available from the floating **Chat** button at the bottom-right of the screen. Open and close it without leaving the current dashboard tab; conversation history and open state are stored in the Streamlit session.

## Validation

From the parent `python` directory, run:

```powershell
python -m unittest discover -s supply_chain_assistant/tests -v
```

This is a production-oriented portfolio baseline, not a compliance-certified purchasing system. Before real deployment, add authentication/authorization, migrations and backups, source-document validation, audit retention, currency/tax rules, supplier verification, SMTP provider delivery monitoring, and human approval policy appropriate to the business.