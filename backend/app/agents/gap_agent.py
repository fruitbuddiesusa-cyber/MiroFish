"""
Gap Agent — identifies underrepresented regions in the data distribution
Finds where synthetic data would be most valuable
"""

import json
from typing import Any

from .base_agent import BaseAgent, AgentResult


class GapAgent(BaseAgent):
    """
    Identifies gaps in existing data:
    1. Coverage gaps (missing categories, edge cases, formats)
    2. Distribution gaps (underrepresented ranges, rare combinations)
    3. Semantic gaps (unexplored topics, missing relationships)
    4. Complexity gaps (missing difficulty levels)
    """

    name = "gap_agent"
    description = "Identifies underrepresented data regions"

    def get_system_prompt(self) -> str:
        return """You are a data coverage analyst. Your job is to identify GAPS in existing datasets — regions that are underrepresented or missing entirely.

Analyze the data and find:

1. **Coverage Gaps**: Missing categories, formats, or structural variants
2. **Distribution Gaps**: Underrepresented value ranges, rare combinations that should appear more
3. **Semantic Gaps**: Topics, entities, or relationships present in the domain but missing from the data
4. **Complexity Gaps**: Missing difficulty levels (too much simple data? missing complex cases?)
5. **Edge Case Gaps**: Boundary conditions, unusual inputs, corner cases not covered

Output JSON:
{
  "coverage_gaps": [
    {"type": "...", "description": "...", "priority": 0.0-1.0, "suggested_count": N}
  ],
  "distribution_gaps": [...],
  "semantic_gaps": [...],
  "complexity_gaps": [...],
  "edge_case_gaps": [...],
  "highest_priority_gaps": [ ... top 5 gaps by priority ... ],
  "synthesis_recommendations": ["..."]
}

Be specific. Each gap should be actionable — something a data generator could create to fill it."""

    def process(self, input_data: dict[str, Any], **kwargs) -> AgentResult:
        """
        Find gaps in the data.

        Args:
            input_data: {
                "text": str,
                "patterns": dict,  # From PatternAgent
                "existing_count": int,
            }
        """
        text = input_data.get("text", "")[:12000]
        patterns = input_data.get("patterns", {})
        existing_count = input_data.get("existing_count", 0)

        messages = [
            {
                "role": "user",
                "content": f"""Analyze this dataset for coverage gaps. We have {existing_count} existing samples.

Discovered patterns:
{json.dumps(patterns, indent=2, ensure_ascii=False)[:5000]}

Data sample:
{text[:8000]}

Identify what's MISSING. What regions of the data space are underexplored? 
What edge cases are absent? What combinations are rare?
Return as JSON with specific, actionable gap descriptions."""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.4)
            return self.make_result(success=True, data=result, existing_count=existing_count)
        except Exception as e:
            return self.make_result(success=False, error=str(e))

    def prioritize_gaps(
        self,
        gaps: dict,
        target_count: int = 100,
    ) -> AgentResult:
        """
        Given identified gaps, prioritize what to fill first
        and allocate sample counts.
        """
        messages = [
            {
                "role": "user",
                "content": f"""We need to generate {target_count} synthetic samples. 
Here are the gaps we identified:

{json.dumps(gaps, indent=2, ensure_ascii=False)[:6000]}

Prioritize these gaps and allocate sample counts. Return JSON:
{{
  "fill_plan": [
    {{
      "gap_description": "...",
      "priority": 1-10,
      "sample_count": N,
      "generation_strategy": "..."
    }}
  ],
  "total_allocated": N,
  "rationale": "..."
}}

Ensure total_allocated <= {target_count}. Focus on highest-impact gaps first."""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.3)
            return self.make_result(success=True, data=result)
        except Exception as e:
            return self.make_result(success=False, error=str(e))
