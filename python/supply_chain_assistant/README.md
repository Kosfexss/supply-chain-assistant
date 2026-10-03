# Small Business Supply Chain Assistant

An operational portfolio starter combining a multi-agent purchasing workflow, SQLite persistence, local RAG retrieval, trend-based inventory forecasts, Streamlit review screens, and explicitly approved SMTP dispatch.

## Start

From the parent `python` directory, create an environment, install the UI dependency, and start the dashboard:

```powershell
python -m venv supply_chain_assistant/.venv
supply_chain_assistant/.venv/Scripts/python.exe -m pip install -r supply_chain_assistant/requirements.txt
supply_chain_assistant/.venv/Scripts/python.exe -m streamlit run supply_chain_assistant/app.py
```

The database is created at `data/supply_chain.sqlite3` on first run. Override it with `SUPPLY_CHAIN_DB_PATH`. Sample documents, inventory, contacts, cash settings, and 30 days of usage history are seeded once into SQLite. The demo is also available with `python -m supply_chain_assistant.demo` from the parent `python` directory.

## Project structure

```text
supply_chain_assistant/
  agents.py              Inventory, supplier/RAG, financial, and proposal agents
  app.py                 Streamlit dashboard and approval workflow
  database.py            SQLite schema and repository methods
  email_dispatcher.py    TLS SMTP dispatch for approved orders
  forecasting.py         Linear-trend usage forecast
  llm.py                 Provider-neutral LLM protocol
  models.py              Shared workflow contracts
  orchestrator.py        Agent coordination and financial gating
  rag.py                 SQLite-backed document search
  sample_data.py         Idempotent seed data for a new database
  tests/                 Standard-library regression tests
```

## Workflow and safety

Inventory rules detect current reorder-point breaches. `ForecastingAgent` fits a simple linear trend to daily usage and proposes a proactive order when predicted stock at supplier lead time reaches the reorder point. RAG retrieves price lists, contracts, invoices, and compliance records from SQLite; supplier selection compares total order cost including structured contract discounts. The finance agent checks the combined estimate against cash remaining after the protected reserve.

The dashboard displays the recommendation and draft for review. Sending requires the explicit **Approve & send email** action, a passing financial assessment, a named approver, a configured SMTP service, and a real supplier address. Each approved/sent/failed order is recorded. Seeded `.example` contacts are blocked from dispatch. SMTP settings are read from environment variables; copy `.env.example` as a reference and set values in your shell or secret manager. Do not commit credentials.

## RAG and LLM

`LocalDocumentStore` reads persisted `Document` rows and retains its deterministic TF-IDF query interface. New documents can be written with `Database.save_document`. The `LLMClient` protocol can be implemented for a chosen provider and injected into `SupplyChainAssistant`; the model only drafts wording and never controls inventory decisions, financial approval, or dispatch. The demo app uses the local template by default.

## Validation

From the parent `python` directory, run:

```powershell
python -m unittest discover -s supply_chain_assistant/tests -v
```

This is a production-oriented portfolio baseline, not a compliance-certified purchasing system. Before real deployment, add authentication/authorization, migrations and backups, source-document validation, audit retention, currency/tax rules, supplier verification, SMTP provider delivery monitoring, and human approval policy appropriate to the business.