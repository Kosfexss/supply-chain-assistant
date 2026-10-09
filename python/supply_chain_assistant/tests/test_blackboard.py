"""Tests for shared cross-department agent memory."""

from __future__ import annotations

import unittest

from supply_chain_assistant.agents import SharedBlackboard


class SharedBlackboardTests(unittest.TestCase):
    def test_logs_all_departments_and_formats_cross_department_context(self) -> None:
        blackboard = SharedBlackboard()
        blackboard.log("Logistics", "Delivery is two days late.", kind="alert")
        blackboard.log("Purchasing", "Supplier quote totals USD 120.", kind="insight")

        context = blackboard.format_context(excluding_departments=("Purchasing",))

        self.assertIn("Logistics · alert (active)", context)
        self.assertIn("Delivery is two days late.", context)
        self.assertNotIn("Supplier quote totals", context)
        self.assertEqual(len(blackboard.active_alerts()), 1)

    def test_alert_can_be_resolved(self) -> None:
        blackboard = SharedBlackboard()
        alert = blackboard.log("Risk", "Supplier disruption reported.", kind="alert")

        resolved = blackboard.resolve_alert(alert.id)

        self.assertFalse(resolved.active)
        self.assertEqual(blackboard.active_alerts(), ())
        self.assertFalse(blackboard.recent_entries()[0].active)

    def test_duplicate_entries_refresh_instead_of_filling_memory(self) -> None:
        blackboard = SharedBlackboard(max_entries=1)
        first = blackboard.log("Inventory", "Reorder recommended.")
        duplicate = blackboard.log("Inventory", "Reorder recommended.")

        self.assertEqual(first.id, duplicate.id)
        self.assertEqual(len(blackboard.recent_entries()), 1)

    def test_unresolved_alerts_do_not_discard_new_notes(self) -> None:
        blackboard = SharedBlackboard(max_entries=2)
        blackboard.log("Risk", "Supplier disruption reported.", kind="alert")
        old_note = blackboard.log("Financial", "Old cash-flow note.")
        note = blackboard.log("Purchasing", "Quote needs approval.")

        self.assertNotIn(old_note, blackboard.recent_entries())
        self.assertIn(note, blackboard.recent_entries())
        self.assertEqual(len(blackboard.active_alerts()), 1)

    def test_full_active_alert_buffer_rejects_new_alerts(self) -> None:
        blackboard = SharedBlackboard(max_entries=1)
        first = blackboard.log("Risk", "Supplier disruption reported.", kind="alert")

        with self.assertRaises(OverflowError):
            blackboard.log("Logistics", "Route disruption reported.", kind="alert")

        self.assertEqual(blackboard.active_alerts(), (first,))

    def test_invalid_entries_are_rejected(self) -> None:
        blackboard = SharedBlackboard()

        with self.assertRaises(ValueError):
            blackboard.log("Operations", "Not an allowed department.")
        with self.assertRaises(ValueError):
            blackboard.log("Risk", " ")
        with self.assertRaises(ValueError):
            blackboard.log("Risk", "Alert must have alert kind.", active=True)


if __name__ == "__main__":
    unittest.main()
