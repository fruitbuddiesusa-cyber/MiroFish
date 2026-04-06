"""
Edge Agent — generates boundary/edge case synthetic data
Focuses on corner cases, unusual inputs, and stress-test scenarios
"""

import json
from typing import Any

from .base_agent import BaseAgent, AgentResult


class EdgeAgent(BaseAgent):
    """
    Specializes in generating edge case data:
    1. Boundary values (min, max, zero, empty, overflow)
    2. Unusual combinations (rare but valid)
    3. Stress inputs (very long, deeply nested, special chars)
    4. Adversarial inputs (near-valid but tricky)
    5. Domain-specific edge cases
    """

    name = "edge_agent"
    description = "Generates boundary and edge case data"

    def get_system_prompt(self) -> str:
        return """You are an edge case data generator. Your job is to create data samples that test BOUNDARY CONDITIONS and CORNER CASES.

For each data type, consider:
- **Boundary values**: Minimum/maximum lengths, zero values, empty strings, null-like values
- **Format edge cases**: Special characters, unicode, unusual but valid formats
- **Semantic edge cases**: Contradictory but parseable inputs, ambiguous cases
- **Scale edge cases**: Very large values, deeply nested structures, long sequences
- **Combination edge cases**: Rare but valid combinations of features

Output a JSON array of edge case samples. Each sample should be:
1. A realistic edge case (not nonsense)
2. Clearly labeled with what edge it tests
3. Complete and self-contained

Format:
[
  {
    "data": "... the actual sample ...",
    "edge_type": "boundary|unusual_combination|stress|adversarial|domain_specific",
    "description": "What edge case this tests",
    "expected_behavior": "How the system should handle this"
  }
]"""

    def process(self, input_data: dict[str, Any], **kwargs) -> AgentResult:
        """
        Generate edge cases based on data patterns.

        Args:
            input_data: {
                "patterns": dict,
                "data_type": str,
                "count": int,
                "domain_context": str,
            }
        """
        patterns = input_data.get("patterns", {})
        data_type = input_data.get("data_type", "text")
        count = input_data.get("count", 10)
        domain_context = input_data.get("domain_context", "")

        messages = [
            {
                "role": "user",
                "content": f"""Generate {count} edge case samples for {data_type} data.

Domain context: {domain_context or 'general'}

Data patterns discovered:
{json.dumps(patterns, indent=2, ensure_ascii=False)[:4000]}

Based on these patterns, what are the boundary conditions, unusual combinations, and corner cases?
Generate {count} diverse edge case samples as a JSON array."""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.6)
            if isinstance(result, dict):
                result = result.get("samples", result.get("edge_cases", [result]))
            return self.make_result(
                success=True,
                data=result,
                count=len(result) if isinstance(result, list) else 1,
            )
        except Exception as e:
            return self.make_result(success=False, error=str(e))

    def generate_for_code(
        self,
        language: str,
        patterns: dict,
        count: int = 10,
    ) -> AgentResult:
        """Generate code-specific edge cases"""
        messages = [
            {
                "role": "user",
                "content": f"""Generate {count} edge case code samples for {language}.

Focus on:
- Null/None handling
- Empty collections
- Type boundary values
- Async/concurrency edge cases
- Error handling paths
- Resource limits (memory, file handles)
- Unicode in strings
- Integer overflow/underflow

Patterns in existing code:
{json.dumps(patterns, indent=2, ensure_ascii=False)[:3000]}

Return as JSON array of {{"data": "code...", "edge_type": "...", "description": "..."}}"""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.5)
            if isinstance(result, dict):
                result = result.get("samples", result.get("edge_cases", [result]))
            return self.make_result(success=True, data=result)
        except Exception as e:
            return self.make_result(success=False, error=str(e))
