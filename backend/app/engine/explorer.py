"""
Explorer — explores the solution space to find optimal generation strategies
Decides WHAT to generate and HOW to generate it
"""

import json
from typing import Any
from dataclasses import dataclass, field

from ..agents.gap_agent import GapAgent
from ..engine.world_model import WorldModel


@dataclass
class ExplorationPlan:
    """Plan for what to generate"""
    target_count: int = 0
    allocations: list[dict] = field(default_factory=list)
    strategies: list[dict] = field(default_factory=list)
    priority_order: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target_count": self.target_count,
            "allocations": self.allocations,
            "strategies": self.strategies,
            "priority_order": self.priority_order,
        }


class Explorer:
    """
    Solution space explorer.
    Uses gap analysis + world model to plan what data to generate.
    """

    def __init__(self):
        self.gap_agent = GapAgent()

    def explore(
        self,
        text: str,
        world_model: WorldModel,
        target_count: int = 50,
        existing_count: int = 0,
    ) -> ExplorationPlan:
        """
        Explore and plan generation strategy.

        1. Identify gaps in existing data
        2. Prioritize what to fill
        3. Allocate sample counts
        4. Choose generation strategies per allocation
        """
        plan = ExplorationPlan(target_count=target_count)

        # Find gaps
        gap_result = self.gap_agent.process({
            "text": text,
            "patterns": world_model.to_dict(),
            "existing_count": existing_count,
        })

        if not gap_result.success:
            # Fallback: simple uniform allocation
            plan.allocations = [{
                "type": "general",
                "count": target_count,
                "strategy": "match_patterns",
                "description": "Generate samples matching discovered patterns",
            }]
            return plan

        gaps = gap_result.data

        # Prioritize
        priority_result = self.gap_agent.prioritize_gaps(gaps, target_count)

        if priority_result.success:
            fill_plan = priority_result.data
            if isinstance(fill_plan, dict):
                items = fill_plan.get("fill_plan", [])
            else:
                items = fill_plan

            for item in items:
                plan.allocations.append({
                    "type": item.get("gap_description", "unknown"),
                    "count": item.get("sample_count", 5),
                    "priority": item.get("priority", 5),
                    "strategy": item.get("generation_strategy", "llm_generate"),
                })
                plan.priority_order.append(item.get("gap_description", "unknown"))
        else:
            # Fallback
            plan.allocations = [{
                "type": "gap_filling",
                "count": target_count,
                "strategy": "llm_generate",
                "description": "Fill identified gaps",
            }]

        # Determine strategies per gap type
        plan.strategies = self._select_strategies(plan.allocations, world_model)

        return plan

    def _select_strategies(
        self,
        allocations: list[dict],
        world_model: WorldModel,
    ) -> list[dict]:
        """Select generation strategies based on gap type"""
        strategies = []

        for alloc in allocations:
            desc = alloc.get("type", "").lower()
            gap_type = alloc.get("gap_type", "").lower()

            if "edge" in desc or "boundary" in desc or "edge" in gap_type:
                strategies.append({
                    "gap": desc,
                    "method": "edge_case_generation",
                    "agent": "edge_agent",
                    "temperature": 0.6,
                })
            elif "code" in desc:
                strategies.append({
                    "gap": desc,
                    "method": "code_synthesis",
                    "agent": "code_agent",
                    "temperature": 0.5,
                })
            elif "reasoning" in desc or "logic" in desc:
                strategies.append({
                    "gap": desc,
                    "method": "reasoning_chain",
                    "agent": "reasoning_agent",
                    "temperature": 0.6,
                })
            elif "distribution" in desc or "rare" in desc:
                strategies.append({
                    "gap": desc,
                    "method": "targeted_generation",
                    "agent": "text_agent",
                    "temperature": 0.7,
                })
            else:
                strategies.append({
                    "gap": desc,
                    "method": "pattern_matching",
                    "agent": "text_agent",
                    "temperature": 0.7,
                })

        return strategies
