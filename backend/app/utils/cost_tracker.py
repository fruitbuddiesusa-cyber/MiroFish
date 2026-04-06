"""
Cost Tracker — track LLM API costs across the engine
"""

import time
import json
import os
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

from ..config.settings import Settings
from .token_counter import TokenCounter


@dataclass
class CostEntry:
    """Single cost entry"""
    timestamp: str
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    operation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "agent": self.agent,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "operation": self.operation,
            "metadata": self.metadata,
        }


class CostTracker:
    """
    Thread-safe cost tracker.
    Logs every LLM call with token counts and estimated cost.
    """

    _lock = Lock()
    _entries: list[CostEntry] = []
    _session_start: str = datetime.now().isoformat()

    @classmethod
    def record(
        cls,
        agent: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "complete",
        metadata: dict | None = None,
    ) -> CostEntry:
        """Record a cost entry"""
        if not Settings.COST_TRACKING_ENABLED:
            return CostEntry(
                timestamp=datetime.now().isoformat(),
                agent=agent, model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=0.0,
                operation=operation,
            )

        cost = TokenCounter.estimate_cost(input_tokens, output_tokens, model)

        entry = CostEntry(
            timestamp=datetime.now().isoformat(),
            agent=agent,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            operation=operation,
            metadata=metadata or {},
        )

        with cls._lock:
            cls._entries.append(entry)

        return entry

    @classmethod
    def record_completion(
        cls,
        agent: str,
        model: str,
        prompt: str,
        response: str,
        operation: str = "complete",
    ) -> CostEntry:
        """Record cost from prompt + response text"""
        input_tokens = TokenCounter.count(prompt, model)
        output_tokens = TokenCounter.count(response, model)
        return cls.record(agent, model, input_tokens, output_tokens, operation)

    @classmethod
    def get_summary(cls) -> dict:
        """Get cost summary"""
        with cls._lock:
            entries = list(cls._entries)

        total_cost = sum(e.cost_usd for e in entries)
        total_input = sum(e.input_tokens for e in entries)
        total_output = sum(e.output_tokens for e in entries)

        by_agent: dict[str, float] = {}
        by_model: dict[str, float] = {}
        for e in entries:
            by_agent[e.agent] = by_agent.get(e.agent, 0) + e.cost_usd
            by_model[e.model] = by_model.get(e.model, 0) + e.cost_usd

        return {
            "session_start": cls._session_start,
            "total_calls": len(entries),
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "budget_remaining": round(Settings.MAX_BUDGET_USD - total_cost, 6),
            "budget_exceeded": total_cost > Settings.MAX_BUDGET_USD,
            "by_agent": {k: round(v, 6) for k, v in by_agent.items()},
            "by_model": {k: round(v, 6) for k, v in by_model.items()},
        }

    @classmethod
    def is_over_budget(cls) -> bool:
        """Check if we've exceeded budget"""
        return cls.get_summary()["budget_exceeded"]

    @classmethod
    def save_log(cls, filepath: str):
        """Save cost log to file"""
        with cls._lock:
            entries = [e.to_dict() for e in cls._entries]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "session_start": cls._session_start,
                "entries": entries,
                "summary": cls.get_summary(),
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def reset(cls):
        """Reset tracker (for new session)"""
        with cls._lock:
            cls._entries.clear()
            cls._session_start = datetime.now().isoformat()
