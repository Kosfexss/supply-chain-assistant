"""Run the sample inventory-to-purchase-draft workflow."""

from dataclasses import asdict
import json

from .database import Database
from .orchestrator import SupplyChainAssistant
from .sample_data import seed_demo_data


def main() -> None:
    database = Database()
    seed_demo_data(database)
    inventory = database.list_inventory()
    usage_history = {
        item.sku: database.get_usage_history(item.sku) for item in inventory
    }
    assistant = SupplyChainAssistant.from_database(database)
    result = assistant.run(
        inventory,
        cash_balance=float(database.get_setting("cash_balance", "1500")),
        minimum_cash_reserve=float(database.get_setting("minimum_cash_reserve", "500")),
        usage_history=usage_history,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()