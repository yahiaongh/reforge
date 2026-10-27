"""Central budget guard for all paid Meta API usage. Hard ceiling: $150.

Policy enforced before EVERY paid request:

1. A reliable pre-call upper bound (``estimate_max_usd``) is required. If the
   caller cannot bound the charge, the request is blocked unless an explicit
   manual exception is supplied (and recorded).
2. The bound must fit inside the discretionary budget:
   ``total_limit - reserve - spent``, where ``spent`` counts confirmed +
   estimated charges, and each unknown-cost entry counts as
   ``unknown_cost_assumption`` (unknown cost is never treated as zero).
3. The bound must also fit inside the per-experiment cap.
4. Retries of a failed request need a retry token: at most ``max_retries``
   retries per experiment.

The guard is conservative local accounting, NOT a billing guarantee: the
provider may bill differently (e.g. retries, streaming overhead). Reconcile
``UsageLedger`` totals against the provider dashboard regularly. No paid call
may proceed when the guard is unavailable; the only bypass is an explicit,
narrowly scoped manual exception from the operator.

All amounts in USD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
import time

from .usage_ledger import UsageLedger


TOTAL_LIMIT_USD = 150.0
DEFAULT_RESERVE_USD = 25.0


@dataclass
class BudgetPolicy:
    total_limit_usd: float = TOTAL_LIMIT_USD
    reserve_usd: float = DEFAULT_RESERVE_USD
    per_experiment_cap_usd: float = 5.0
    max_retries: int = 2
    max_correction_calls: int = 2
    # Conservative stand-in charge for each ledger entry whose true cost is
    # unknown. Never zero.
    unknown_cost_assumption_usd: float = 5.0


@dataclass
class GuardDecision:
    allowed: bool
    reason: str
    spent_usd: float = 0.0
    discretionary_usd: float = 0.0


class BudgetGuardUnavailable(RuntimeError):
    """Raised when the guard cannot evaluate (e.g. ledger unreadable)."""


class BudgetGuard:
    def __init__(self, policy: BudgetPolicy, ledger: UsageLedger):
        self.policy = policy
        self.ledger = ledger
        self._retries_used: dict[str, int] = {}

    # -- accounting ------------------------------------------------------
    def spent_usd(self) -> float:
        """Conservative spend: confirmed + estimated + assumption per unknown."""
        try:
            totals = self.ledger.totals()
        except OSError as exc:
            raise BudgetGuardUnavailable(f"cannot read usage ledger: {exc}") from exc
        return (
            totals["confirmed_usd"]
            + totals["estimated_usd"]
            + totals["unknown_count"] * self.policy.unknown_cost_assumption_usd
        )

    def discretionary_usd(self) -> float:
        """Amount available for new experiments after spend and reserve."""
        return self.policy.total_limit_usd - self.policy.reserve_usd - self.spent_usd()

    # -- gating ----------------------------------------------------------
    def check(
        self,
        experiment_id: str,
        estimate_max_usd: float | None,
        *,
        is_retry: bool = False,
        manual_exception: str = "",
    ) -> GuardDecision:
        """Pre-call gate. Returns a decision; callers must honour ``allowed``."""
        spent = self.spent_usd()
        discretionary = self.discretionary_usd()
        cap = self.policy.per_experiment_cap_usd

        if estimate_max_usd is None:
            if manual_exception:
                return GuardDecision(
                    True,
                    f"allowed by manual exception: {manual_exception}",
                    spent,
                    discretionary,
                )
            self.ledger.log(
                provider="", model="", operation="",
                status="blocked", experiment_id=experiment_id,
                cost_status="unknown",
                note="blocked: no reliable pre-call cost bound and no manual exception",
            )
            return GuardDecision(
                False,
                "blocked: cannot bound maximum charge and no manual exception given",
                spent,
                discretionary,
            )
        if estimate_max_usd < 0:
            return GuardDecision(False, "blocked: negative cost bound", spent, discretionary)
        if estimate_max_usd > cap and not manual_exception:
            self.ledger.log(
                provider="", model="", operation="",
                status="blocked", experiment_id=experiment_id,
                estimate_usd=estimate_max_usd, cost_basis="pre-call bound",
                cost_status="estimated",
                note=f"blocked: bound ${estimate_max_usd:.2f} exceeds per-experiment cap ${cap:.2f}",
            )
            return GuardDecision(
                False,
                f"blocked: ${estimate_max_usd:.2f} exceeds per-experiment cap ${cap:.2f}",
                spent,
                discretionary,
            )
        if estimate_max_usd > discretionary and not manual_exception:
            self.ledger.log(
                provider="", model="", operation="",
                status="blocked", experiment_id=experiment_id,
                estimate_usd=estimate_max_usd, cost_basis="pre-call bound",
                cost_status="estimated",
                note="blocked: bound exceeds discretionary budget (reserve protected)",
            )
            return GuardDecision(
                False,
                f"blocked: ${estimate_max_usd:.2f} exceeds discretionary "
                f"${discretionary:.2f} (reserve ${self.policy.reserve_usd:.2f} protected)",
                spent,
                discretionary,
            )
        if is_retry:
            used = self._retries_used.get(experiment_id, 0)
            if used >= self.policy.max_retries:
                return GuardDecision(
                    False,
                    f"blocked: retry limit ({self.policy.max_retries}) exhausted "
                    f"for experiment {experiment_id!r}",
                    spent,
                    discretionary,
                )
            self._retries_used[experiment_id] = used + 1
        return GuardDecision(True, "allowed", spent, discretionary)

    # -- cache helper ----------------------------------------------------
    def check_cache(self, cache: "ResultCache", key: str) -> dict | None:
        """Return a cached result when valid; a cache hit needs no paid call."""
        return cache.get(key)


class ResultCache:
    """Tiny file-backed cache so identical requests never pay twice.

    Keys must capture model, operation, input-asset hash, and configuration.
    Values are JSON-serializable provider responses (never secrets).
    """

    def __init__(self, directory: str):
        self._directory = directory

    def _path(self, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self._directory, digest + ".json")

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def put(self, key: str, value: dict) -> str:
        os.makedirs(self._directory, exist_ok=True)
        path = self._path(key)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
        return path
