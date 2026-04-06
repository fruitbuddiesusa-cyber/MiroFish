"""
Text Agent — generates synthetic text data (NLP, summarization, classification, etc.)
General-purpose text generation with style matching
"""

import json
from typing import Any

from .base_agent import BaseAgent, AgentResult


class TextAgent(BaseAgent):
    """
    Generates synthetic text data:
    1. Topic-preserving text variations
    2. Paraphrases and rewrites
    3. Classification label pairs (text + label)
    4. Summarization pairs (long → short)
    5. Named entity examples
    6. Sentiment/style transfers
    """

    name = "text_agent"
    description = "Generates synthetic text samples"

    def get_system_prompt(self) -> str:
        return """You are a synthetic text data generator. You create realistic text samples that match the style, tone, and domain of existing data.

Key principles:
- Preserve the statistical properties of the original data (length distribution, vocabulary, complexity)
- Generate DIVERSE content — not slight rewordings of the same thing
- Maintain factual plausibility (don't create contradictory claims)
- Match formality level, technical depth, and domain vocabulary
- For labeled data, ensure labels are accurate

Output JSON:
{
  "samples": [
    {
      "text": "...",
      "label": "..." (if applicable),
      "metadata": {"length": N, "complexity": "..."}
    }
  ]
}"""

    def process(self, input_data: dict[str, Any], **kwargs) -> AgentResult:
        """
        Generate synthetic text samples.

        Args:
            input_data: {
                "patterns": dict,
                "task_type": str,  # paraphrase, classify, summarize, generate
                "count": int,
                "reference_data": str,
                "labels": list,    # For classification tasks
            }
        """
        patterns = input_data.get("patterns", {})
        task_type = input_data.get("task_type", "generate")
        count = input_data.get("count", 10)
        reference = input_data.get("reference_data", "")[:8000]
        labels = input_data.get("labels", [])

        label_instruction = ""
        if labels:
            label_instruction = f"\nUse these labels: {labels}. Generate balanced samples across all labels."

        messages = [
            {
                "role": "user",
                "content": f"""Generate {count} synthetic text samples. Task: {task_type}

Style patterns:
{json.dumps(patterns, indent=2, ensure_ascii=False)[:3000]}

Reference data:
{reference}
{label_instruction}

Match the style, vocabulary, and complexity of the reference. Make each sample unique and diverse.
Return as JSON with a "samples" array."""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.7)
            samples = result.get("samples", []) if isinstance(result, dict) else result
            return self.make_result(
                success=True,
                data=samples,
                task_type=task_type,
            )
        except Exception as e:
            return self.make_result(success=False, error=str(e))

    def generate_classification_pairs(
        self,
        categories: list[str],
        reference_text: str,
        count_per_category: int = 5,
    ) -> AgentResult:
        """Generate (text, label) pairs for classification training"""
        messages = [
            {
                "role": "user",
                "content": f"""Generate {count_per_category} text samples for EACH of these categories: {categories}

Reference text (for style):
{reference_text[:4000]}

Each sample should be clearly belonging to its category while matching the reference style.
Return JSON: {{"samples": [{{"text": "...", "label": "category_name"}}]}}"""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.6)
            samples = result.get("samples", []) if isinstance(result, dict) else result
            return self.make_result(success=True, data=samples)
        except Exception as e:
            return self.make_result(success=False, error=str(e))

    def generate_summarization_pairs(
        self,
        reference_text: str,
        count: int = 5,
        ratio: float = 0.3,
    ) -> AgentResult:
        """Generate (long_text, summary) pairs"""
        messages = [
            {
                "role": "user",
                "content": f"""Generate {count} (document, summary) pairs.

Style reference:
{reference_text[:3000]}

Each pair should have:
- A document (300-800 words)
- A summary (~{int(ratio*100)}% of the document length)

Return JSON: {{"pairs": [{{"document": "...", "summary": "..."}}]}}"""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.6)
            pairs = result.get("pairs", []) if isinstance(result, dict) else result
            return self.make_result(success=True, data=pairs)
        except Exception as e:
            return self.make_result(success=False, error=str(e))
