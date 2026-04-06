"""
Pattern Agent — discovers data patterns and builds a world model
Learns the distribution, structure, and semantic patterns in input data
"""

import json
from typing import Any

from .base_agent import BaseAgent, AgentResult
from ..utils.data_detector import DataProfile, DataType


class PatternAgent(BaseAgent):
    """
    Analyzes input data to discover:
    1. Structural patterns (formats, schemas, templates)
    2. Semantic patterns (topics, entities, relationships)
    3. Distribution patterns (frequency, ranges, outliers)
    4. Style patterns (tone, complexity, length distribution)
    """

    name = "pattern_agent"
    description = "Discovers data patterns and builds world models"

    def get_system_prompt(self) -> str:
        return """You are a data pattern discovery expert. Your job is to deeply analyze data samples and extract:

1. **Structural Patterns**: How the data is organized (fields, types, formats, templates)
2. **Semantic Patterns**: What topics/domains are covered, key entities, relationships
3. **Distribution Patterns**: Statistical characteristics, value ranges, frequency distributions
4. **Style Patterns**: Writing style, tone, complexity level, typical length
5. **Constraints**: What rules or constraints govern valid data in this domain

Output a JSON object with these sections:
{
  "structural_patterns": { ... },
  "semantic_patterns": { ... },
  "distribution_patterns": { ... },
  "style_patterns": { ... },
  "constraints": [ ... ],
  "sample_archetypes": [ ... ],  // 3-5 representative examples
  "variation_axes": [ ... ]       // Dimensions along which data can vary
}

Be precise. Focus on patterns that would help generate new, realistic synthetic data."""

    def process(
        self,
        input_data: dict[str, Any],
        **kwargs,
    ) -> AgentResult:
        """
        Analyze data and extract patterns.

        Args:
            input_data: {
                "text": str,           # The data content
                "profile": DataProfile, # Data type profile
                "sample_count": int,   # How many samples to analyze
            }
        """
        text = input_data.get("text", "")
        profile = input_data.get("profile")
        data_type = profile.data_type if profile else DataType.TEXT

        # Truncate for analysis if too long
        analysis_text = text[:15000] if len(text) > 15000 else text

        messages = [
            {
                "role": "user",
                "content": f"""Analyze the following {data_type.value} data and extract all discoverable patterns.

Data type: {data_type.value}
Character count: {len(text)}
Line count: {text.count(chr(10)) + 1}

--- DATA START ---
{analysis_text}
--- DATA END ---

Extract structural, semantic, distribution, and style patterns. Return as JSON."""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.3)
            return self.make_result(success=True, data=result, data_type=data_type.value)
        except Exception as e:
            return self.make_result(success=False, error=str(e))

    def analyze_chunks(
        self,
        chunks: list[str],
        data_type: str = "text",
    ) -> AgentResult:
        """
        Analyze multiple chunks to find cross-chunk patterns.
        Good for large datasets that don't fit in one prompt.
        """
        # Sample representative chunks
        if len(chunks) > 10:
            step = len(chunks) // 10
            sampled = [chunks[i] for i in range(0, len(chunks), step)][:10]
        else:
            sampled = chunks

        combined = "\n\n---CHUNK---\n\n".join(sampled[:8])

        messages = [
            {
                "role": "user",
                "content": f"""Analyze these {data_type} data chunks from a larger dataset. 
Find patterns that appear ACROSS chunks (not just within individual chunks).

{combined[:12000]}

Return JSON with:
- "cross_chunk_patterns": patterns that repeat across chunks
- "data_progression": how the data changes/evolves across chunks
- "common_templates": shared structural templates
- "variation_summary": how chunks differ from each other"""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.3)
            return self.make_result(success=True, data=result)
        except Exception as e:
            return self.make_result(success=False, error=str(e))
