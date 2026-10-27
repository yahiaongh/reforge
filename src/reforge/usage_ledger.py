"""Local usage ledger: append-only JSONL record of every model API call.

Each entry records what the budget guard needs and nothing sensitive: no
prompt text, no image bytes, no credentials. Image inputs are referenced by
SHA-256 hash plus an experiment identifier.

Charge fields: ``charge_usd`` when the provider reports an actual charge,
otherwise ``estimate_usd`` with ``cost_basis`` explaining the estimate, and
``cost_status`` one of "confirmed" | "estimated" | "unknown". Unknown cost is
never treated as zero (see budget.py).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
import threading
import time
from typing import Any


@dataclass
class LedgerEntry:
    timestamp: float
    provider: str
    model: str
    operation: str
    asset_hash: str = ""
    experiment_id: str = ""
    status: str = "ok"  # ok | error | blocked | cached
    input_tokens: int | None = None
    output_tokens: int | None = None
    image_count: int = 0
    charge_usd: float | None = None
    estimate_usd: float | None = None
    cost_basis: str = ""
    cost_status: str = "unknown"  # confirmed | estimated | unknown
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LedgerEntry":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


def sha256_file(path: str) -> str:
    """SHA-256 hex digest of a file (used to reference image inputs)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UsageLedger:
    """Thread-safe append-only JSONL ledger."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def record(self, entry: LedgerEntry) -> LedgerEntry:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_dict()) + "\n")
        return entry

    def log(
        self,
        *,
        provider: str,
        model: str,
        operation: str,
        status: str = "ok",
        asset_hash: str = "",
        experiment_id: str = "",
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        image_count: int = 0,
        charge_usd: float | None = None,
        estimate_usd: float | None = None,
        cost_basis: str = "",
        cost_status: str = "unknown",
        note: str = "",
    ) -> LedgerEntry:
        return self.record(
            LedgerEntry(
                timestamp=time.time(),
                provider=provider,
                model=model,
                operation=operation,
                asset_hash=asset_hash,
                experiment_id=experiment_id,
                status=status,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                image_count=image_count,
                charge_usd=charge_usd,
                estimate_usd=estimate_usd,
                cost_basis=cost_basis,
                cost_status=cost_status,
                note=note,
            )
        )

    def entries(self) -> list[LedgerEntry]:
        if not os.path.lexists(self.path):
            return []
        if not os.path.isfile(self.path):
            raise OSError(f"ledger path is not a readable file: {self.path}")
        out = []
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    out.append(LedgerEntry.from_dict(json.loads(line)))
        return out

    def totals(self) -> dict[str, float]:
        """Separate totals for confirmed, estimated, and unknown-cost usage."""
        confirmed = 0.0
        estimated = 0.0
        unknown = 0
        for entry in self.entries():
            if entry.status in ("blocked", "cached"):
                continue  # blocked/cached entries billed nothing known;
                # "blocked" was never sent, "cached" reused a prior result
            if entry.cost_status == "confirmed" and entry.charge_usd is not None:
                confirmed += entry.charge_usd
            elif entry.cost_status == "estimated" and entry.estimate_usd is not None:
                estimated += entry.estimate_usd
            else:
                unknown += 1
        return {"confirmed_usd": confirmed, "estimated_usd": estimated, "unknown_count": unknown}


def default_ledger_path() -> str:
    return os.path.join(os.getcwd(), "ledger", "usage.jsonl")
