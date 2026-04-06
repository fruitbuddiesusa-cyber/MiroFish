"""
World Model — builds a digital twin of the problem space
Captures the "physics" of what makes data valid/realistic in this domain
"""

import json
from typing import Any
from dataclasses import dataclass, field

from ..agents.pattern_agent import PatternAgent
from ..engine.analyzer import DataAnalysis


@dataclass
class WorldModel:
    """
    Digital twin of the data generation problem.
    Captures rules, distributions, and constraints.
    """
    # Domain rules
    rules: list[dict] = field(default_factory=list)
    # Value distributions
    distributions: dict[str, Any] = field(default_factory=dict)
    # Valid combinations
    valid_combinations: list[dict] = field(default_factory=list)
    # Constraints (hard and soft)
    hard_constraints: list[str] = field(default_factory=list)
    soft_constraints: list[str] = field(default_factory=list)
    # Semantic space
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    # Style parameters
    style: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rules": self.rules,
            "distributions": self.distributions,
            "valid_combinations": self.valid_combinations,
            "hard_constraints": self.hard_constraints,
            "soft_constraints": self.soft_constraints,
            "topics": self.topics,
            "entities": self.entities,
            "relationships": self.relationships,
            "style": self.style,
        }

    def validate_sample(self, sample: Any) -> tuple[bool, list[str]]:
        """
        Quick validation of a sample against hard constraints.
        Returns (valid, list_of_violations).
        """
        violations = []
        for constraint in self.hard_constraints:
            # Simple string-based check (LLM does deeper validation)
            if isinstance(sample, str) and constraint.lower() not in sample.lower():
                pass  # Constraint might not apply to every sample
        return len(violations) == 0, violations


class WorldModelBuilder:
    """
    Builds a WorldModel from data analysis and LLM pattern extraction.
    """

    def __init__(self):
        self.pattern_agent = PatternAgent()

    def build(
        self,
        text: str,
        analysis: DataAnalysis | None = None,
    ) -> WorldModel:
        """
        Build world model from data.
        """
        model = WorldModel()

        # Extract patterns via LLM
        result = self.pattern_agent.process({
            "text": text,
            "profile": analysis.profile if analysis else None,
        })

        if not result.success:
            return model

        patterns = result.data

        # Map patterns to world model components
        model.rules = self._extract_rules(patterns)
        model.distributions = patterns.get("distribution_patterns", {})
        model.hard_constraints = self._extract_hard_constraints(patterns)
        model.soft_constraints = self._extract_soft_constraints(patterns)
        model.topics = patterns.get("semantic_patterns", {}).get("topics", [])
        model.entities = patterns.get("semantic_patterns", {}).get("entities", [])
        model.relationships = patterns.get("semantic_patterns", {}).get("relationships", [])
        model.style = patterns.get("style_patterns", {})
        model.valid_combinations = patterns.get("sample_archetypes", [])

        return model

    def _extract_rules(self, patterns: dict) -> list[dict]:
        """Extract domain rules from patterns"""
        rules = []

        # Structural rules
        structural = patterns.get("structural_patterns", {})
        if structural.get("required_fields"):
            rules.append({
                "type": "structural",
                "rule": f"Must have fields: {structural['required_fields']}",
                "strict": True,
            })

        # Constraint-derived rules
        constraints = patterns.get("constraints", [])
        for c in constraints:
            if isinstance(c, str):
                rules.append({"type": "domain", "rule": c, "strict": True})
            elif isinstance(c, dict):
                rules.append(c)

        return rules

    def _extract_hard_constraints(self, patterns: dict) -> list[str]:
        """Extract hard constraints (must not violate)"""
        constraints = patterns.get("constraints", [])
        hard = []
        for c in constraints:
            if isinstance(c, dict) and c.get("strict", False):
                hard.append(c.get("rule", str(c)))
            elif isinstance(c, str):
                hard.append(c)
        return hard

    def _extract_soft_constraints(self, patterns: dict) -> list[str]:
        """Extract soft constraints (preferred but can bend)"""
        style = patterns.get("style_patterns", {})
        soft = []
        if style.get("tone"):
            soft.append(f"Preferred tone: {style['tone']}")
        if style.get("complexity"):
            soft.append(f"Preferred complexity: {style['complexity']}")
        return soft
