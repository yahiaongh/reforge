"""Budget guard and ledger tests. No paid calls; all fixtures are local."""

import os
import unittest

from tests import TempDirTest
from reforge.budget import (
    BudgetGuard,
    BudgetGuardUnavailable,
    BudgetPolicy,
    ResultCache,
)
from reforge.usage_ledger import UsageLedger


def make_guard(case: TempDirTest, **policy_kwargs) -> tuple[BudgetGuard, UsageLedger]:
    policy = BudgetPolicy(**policy_kwargs)
    ledger = UsageLedger(case.path("usage.jsonl"))
    return BudgetGuard(policy, ledger), ledger


class TestAllowedRequests(TempDirTest):
    def test_small_request_permitted(self):
        guard, _ = make_guard(self)
        decision = guard.check("exp-1", 1.0)
        self.assertTrue(decision.allowed, msg=decision.reason)

    def test_decision_reports_spend_and_discretionary(self):
        guard, _ = make_guard(self)
        decision = guard.check("exp-1", 1.0)
        self.assertAlmostEqual(decision.spent_usd, 0.0)
        self.assertAlmostEqual(decision.discretionary_usd, 125.0)  # 150 - 25 reserve


class TestBlockedRequests(TempDirTest):
    def test_exceeding_total_blocked(self):
        guard, ledger = make_guard(self)
        ledger.log(provider="meta", model="m", operation="op",
                   charge_usd=120.0, cost_status="confirmed")
        decision = guard.check("exp-2", 10.0)
        self.assertFalse(decision.allowed)

    def test_reserve_is_protected(self):
        guard, ledger = make_guard(self, reserve_usd=25.0)
        # Spend exactly to the point where only the reserve remains.
        ledger.log(provider="meta", model="m", operation="op",
                   charge_usd=125.0, cost_status="confirmed")
        decision = guard.check("exp-3", 0.50)
        self.assertFalse(decision.allowed, msg=decision.reason)
        self.assertIn("reserve", decision.reason)

    def test_per_experiment_cap_enforced(self):
        guard, _ = make_guard(self, per_experiment_cap_usd=5.0)
        decision = guard.check("exp-4", 5.01)
        self.assertFalse(decision.allowed)

    def test_unknown_cost_never_zero(self):
        guard, ledger = make_guard(
            self, unknown_cost_assumption_usd=5.0, reserve_usd=25.0)
        ledger.log(provider="meta", model="m", operation="op", cost_status="unknown")
        self.assertAlmostEqual(guard.spent_usd(), 5.0)
        # Unknown spend consumes discretionary budget.
        ledger.log(provider="meta", model="m", operation="op",
                   charge_usd=120.0, cost_status="confirmed")
        decision = guard.check("exp-5", 1.0)
        self.assertFalse(decision.allowed)  # 120 + 5 unknown leaves only reserve

    def test_unbounded_request_blocked_without_exception(self):
        guard, _ = make_guard(self)
        decision = guard.check("exp-6", None)
        self.assertFalse(decision.allowed)

    def test_blocked_request_is_recorded(self):
        guard, ledger = make_guard(self, per_experiment_cap_usd=1.0)
        decision = guard.check("exp-7", 50.0)
        self.assertFalse(decision.allowed)
        blocked = [e for e in ledger.entries() if e.status == "blocked"]
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0].experiment_id, "exp-7")

    def test_no_api_after_block(self):
        # The guard returns a decision; a compliant caller sends nothing when
        # blocked. Here we assert the ledger shows no "ok" charge entries.
        guard, ledger = make_guard(self, per_experiment_cap_usd=1.0)
        decision = guard.check("exp-8", 50.0)
        self.assertFalse(decision.allowed)
        self.assertEqual(
            [e for e in ledger.entries() if e.status == "ok"], [])


class TestRetryLimit(TempDirTest):
    def test_retries_capped(self):
        guard, _ = make_guard(self, max_retries=2)
        self.assertTrue(guard.check("exp-r", 1.0, is_retry=True).allowed)
        self.assertTrue(guard.check("exp-r", 1.0, is_retry=True).allowed)
        third = guard.check("exp-r", 1.0, is_retry=True)
        self.assertFalse(third.allowed)
        self.assertIn("retry limit", third.reason)

    def test_retry_budgets_are_per_experiment(self):
        guard, _ = make_guard(self, max_retries=1)
        self.assertTrue(guard.check("exp-a", 1.0, is_retry=True).allowed)
        self.assertTrue(guard.check("exp-b", 1.0, is_retry=True).allowed)


class TestLedgerTotals(TempDirTest):
    def test_totals_separate_confirmed_estimated_unknown(self):
        _, ledger = make_guard(self)
        ledger.log(provider="p", model="m", operation="o",
                   charge_usd=2.0, cost_status="confirmed")
        ledger.log(provider="p", model="m", operation="o",
                   estimate_usd=1.5, cost_basis="unit pricing",
                   cost_status="estimated")
        ledger.log(provider="p", model="m", operation="o", cost_status="unknown")
        ledger.log(provider="p", model="m", operation="o",
                   status="blocked", cost_status="unknown")
        totals = ledger.totals()
        self.assertAlmostEqual(totals["confirmed_usd"], 2.0)
        self.assertAlmostEqual(totals["estimated_usd"], 1.5)
        self.assertEqual(totals["unknown_count"], 1)  # blocked entries excluded

    def test_failed_request_recorded(self):
        _, ledger = make_guard(self)
        ledger.log(provider="p", model="m", operation="o", status="error",
                   cost_status="unknown", note="HTTP 500")
        errors = [e for e in ledger.entries() if e.status == "error"]
        self.assertEqual(len(errors), 1)

    def test_ledger_contains_no_secrets(self):
        _, ledger = make_guard(self)
        ledger.log(provider="p", model="m", operation="o",
                   asset_hash="abc123", experiment_id="e1", cost_status="unknown")
        with open(ledger.path, encoding="utf-8") as handle:
            raw = handle.read()
        for forbidden in ("MODEL_API_KEY", "sk-", "Bearer", "password"):
            self.assertNotIn(forbidden, raw)


class TestResultCache(TempDirTest):
    def test_cache_hit_avoids_resend(self):
        cache = ResultCache(self.path("cache"))
        self.assertIsNone(cache.get("key-1"))
        cache.put("key-1", {"boxes": [], "masks": 1})
        guard, _ = make_guard(self)
        hit = guard.check_cache(cache, "key-1")
        self.assertEqual(hit, {"boxes": [], "masks": 1})

    def test_cache_keys_are_content_addressed(self):
        cache = ResultCache(self.path("cache"))
        cache.put("same-key", {"v": 1})
        self.assertEqual(cache.get("same-key"), {"v": 1})
        self.assertIsNone(cache.get("other-key"))


class TestGuardUnavailable(TempDirTest):
    def test_unreadable_ledger_blocks(self):
        _, ledger = make_guard(self)
        os.makedirs(ledger.path)  # a directory at the ledger path: unreadable as file
        guard = BudgetGuard(BudgetPolicy(), ledger)
        with self.assertRaises(BudgetGuardUnavailable):
            guard.spent_usd()


if __name__ == "__main__":
    unittest.main()
