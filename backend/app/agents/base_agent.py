"""
Base Agent — foundation for all specialized agents
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass, field

from ..config.llm_factory import LLMFactory
from ..config.settings import Settings
from ..utils.token_counter import TokenCounter
from ..utils.cost_tracker import CostTracker


@dataclass
class AgentResult:
    """Standard result from any agent"""
    success: bool
    data: Any = None
    error: str | None = None
    agent_name: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "agent_name": self.agent_name,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    Abstract base for all agents.
    Handles LLM calls, cost tracking, retries, and structured output parsing.
    """

    name: str = "base_agent"
    description: str = ""
    use_boost: bool = False  # Use cheaper/faster model

    def __init__(self):
        self._call_count = 0

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent"""
        ...

    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> AgentResult:
        """Core processing method — subclasses implement this"""
        ...

    def llm_call(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        parse_json: bool = False,
    ) -> str | dict | list:
        """
        Make an LLM call with cost tracking and retries.
        """
        temp = temperature if temperature is not None else Settings.AGENT_TEMPERATURE

        # Build full message list with system prompt
        full_messages = [
            {"role": "system", "content": self.get_system_prompt()},
            *messages,
        ]

        # Retry loop
        last_error = None
        for attempt in range(Settings.AGENT_MAX_RETRIES):
            try:
                if parse_json:
                    result = LLMFactory.complete_json(
                        messages=full_messages,
                        use_boost=self.use_boost,
                        temperature=temp,
                    )
                    response_text = json.dumps(result)
                else:
                    response_text = LLMFactory.complete(
                        messages=full_messages,
                        use_boost=self.use_boost,
                        temperature=temp,
                        max_tokens=max_tokens,
                    )
                    result = response_text

                # Track cost
                model = LLMFactory.get_model(use_boost=self.use_boost)
                prompt_text = json.dumps(full_messages)
                CostTracker.record_completion(
                    agent=self.name,
                    model=model,
                    prompt=prompt_text,
                    response=response_text,
                )
                self._call_count += 1

                return result

            except Exception as e:
                last_error = e
                if attempt < Settings.AGENT_MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        raise RuntimeError(f"{self.name} LLM call failed after {Settings.AGENT_MAX_RETRIES} retries: {last_error}")

    def make_result(
        self,
        success: bool,
        data: Any = None,
        error: str | None = None,
        **metadata,
    ) -> AgentResult:
        """Helper to create AgentResult"""
        return AgentResult(
            success=success,
            data=data,
            error=error,
            agent_name=self.name,
            metadata=metadata,
        )

    def get_stats(self) -> dict:
        """Get agent usage stats"""
        return {
            "agent": self.name,
            "call_count": self._call_count,
        }
