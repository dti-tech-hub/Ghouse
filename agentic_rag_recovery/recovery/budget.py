"""Budget accounting for the bounded control loop."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..models import BudgetSnapshot
from ..settings import BudgetConfig, get_settings
from ..taxonomy import RecoveryAction


class BudgetExhausted(RuntimeError):
    """Raised when a hard limit forbids further work."""

    def __init__(self, resource: str):
        super().__init__(f"budget exhausted: {resource}")
        self.resource = resource


class BudgetTracker:
    """Tracks attempts, latency, LLM calls, tokens and repeat suppression."""

    def __init__(
        self,
        budget: Optional[BudgetConfig] = None,
        baseline_latency_ms: float = 0.0,
        stage_latency_ms: Optional[float] = None,
    ):
        self.limits: BudgetConfig = budget or get_settings().budget
        self.baseline_latency_ms = float(baseline_latency_ms)
        self.stage_latency_ms = float(stage_latency_ms or 0.0)
        self.attempts_used: int = 0
        self.latency_used_ms: float = 0.0
        self.llm_calls_used: int = 0
        self.tokens_used: int = 0
        self.cost: float = 0.0
        self.repeated_actions_blocked: int = 0
        self.attempted: List[RecoveryAction] = []
        self.attempted_set: Set[RecoveryAction] = set()
        self.log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def preflight(self) -> Optional[str]:
        """Return the name of the exhausted resource, or ``None``."""
        if self.attempts_used >= self.limits.max_recovery_attempts:
            return "attempts"
        if self.llm_calls_used >= self.limits.max_llm_calls:
            return "llm_calls"
        if self.tokens_used >= self.limits.max_token_budget:
            return "tokens"
        if self.latency_used_ms + self.stage_latency_ms >= self.limits.max_total_latency_ms:
            return "latency"
        return None

    def can_attempt(self, action: RecoveryAction) -> "tuple[bool, Optional[str]]":
        """Check every hard limit *and* duplicate-action suppression."""
        if action in self.attempted_set:
            self.repeated_actions_blocked += 1
            return False, "BLOCK_REPEAT: action already attempted"
        if self.attempted_set and self.repeated_actions_blocked >= self.limits.max_repeated_actions * len(
            self.attempted_set
        ) + self.limits.max_repeated_actions:
            # safety valve: too many blocked repeats in this request
            self.repeated_actions_blocked += 1
            return False, "BLOCK_REPEAT: repeat budget exhausted"
        resource = self.preflight()
        if resource:
            return False, f"budget exhausted: {resource}"
        return True, None

    # ------------------------------------------------------------------
    def begin_attempt(self, action: RecoveryAction) -> None:
        self.attempts_used += 1
        self.attempted.append(action)
        self.attempted_set.add(action)

    def record_latency(self, latency_ms: float) -> None:
        self.latency_used_ms += float(latency_ms)

    def record_llm(self, calls: int = 1, tokens: int = 0, cost: float = 0.0) -> None:
        self.llm_calls_used += int(calls)
        self.tokens_used += int(tokens)
        self.cost += float(cost)

    def note(self, **kwargs: Any) -> None:
        self.log.append(kwargs)

    # ------------------------------------------------------------------
    def snapshot(self) -> BudgetSnapshot:
        exhausted = {
            "attempts": self.attempts_used >= self.limits.max_recovery_attempts,
            "latency": self.latency_used_ms >= self.limits.max_total_latency_ms,
            "llm_calls": self.llm_calls_used >= self.limits.max_llm_calls,
            "tokens": self.tokens_used >= self.limits.max_token_budget,
        }
        return BudgetSnapshot(
            max_recovery_attempts=self.limits.max_recovery_attempts,
            max_total_latency_ms=self.limits.max_total_latency_ms,
            max_llm_calls=self.limits.max_llm_calls,
            max_token_budget=self.limits.max_token_budget,
            max_repeated_actions=self.limits.max_repeated_actions,
            attempts_used=self.attempts_used,
            attempts_remaining=max(0, self.limits.max_recovery_attempts - self.attempts_used),
            latency_used_ms=round(self.latency_used_ms, 3),
            latency_remaining_ms=round(
                max(0.0, self.limits.max_total_latency_ms - self.latency_used_ms - self.stage_latency_ms), 3
            ),
            llm_calls_used=self.llm_calls_used,
            llm_calls_remaining=max(0, self.limits.max_llm_calls - self.llm_calls_used),
            tokens_used=self.tokens_used,
            tokens_remaining=max(0, self.limits.max_token_budget - self.tokens_used),
            repeated_actions_blocked=self.repeated_actions_blocked,
        )

    def snapshot_dict(self) -> Dict[str, Any]:
        snapshot = self.snapshot()
        payload = snapshot.model_dump()
        payload["exhausted"] = snapshot.exhausted()
        payload["first_exhausted"] = snapshot.first_exhausted()
        payload["attempted_actions"] = [a.value for a in self.attempted]
        return payload

    def consume_baseline(self, latency_ms: float, llm_calls: int, tokens: int, cost: float) -> None:
        """The happy-path execution counts against the request budget too."""
        self.latency_used_ms += float(latency_ms)
        self.llm_calls_used += int(llm_calls)
        self.tokens_used += int(tokens)
        self.cost += float(cost)

    @property
    def added_latency_ms(self) -> float:
        return round(max(0.0, self.latency_used_ms - self.baseline_latency_ms), 3)

    def cost_per_recovery(self, successful: int) -> float:
        return round(self.cost / successful, 6) if successful else 0.0