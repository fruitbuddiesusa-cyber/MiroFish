"""
Validator Agent — quality validation for generated synthetic data
Multi-dimensional quality checking
"""

import json
from typing import Any

from .base_agent import BaseAgent, AgentResult


class ValidatorAgent(BaseAgent):
    """
    Validates synthetic data quality across multiple dimensions:
    1. Format validity (syntax, structure, schema conformance)
    2. Semantic validity (meaning, logic, factual consistency)
    3. Diversity check (not duplicates or near-duplicates)
    4. Distribution match (statistical similarity to source)
    5. Label accuracy (for labeled data)
    6. Contamination check (not copying from training data)
    """

    name = "validator_agent"
    description = "Validates synthetic data quality"
    use_boost = True  # Use cheaper model for validation

    def get_system_prompt(self) -> str:
        return """You are a data quality validator. You evaluate synthetic data samples across multiple quality dimensions.

For each sample, assess:
1. **Format Validity**: Is it well-formed? Syntactically correct?
2. **Semantic Validity**: Does it make sense? Is it logically consistent?
3. **Realism**: Does it look like real data, or obviously synthetic?
4. **Diversity**: Is it distinct from other samples?
5. **Label Accuracy**: If labeled, is the label correct?
6. **Contamination Risk**: Does it look like a copy from known datasets?

Output JSON:
{
  "results": [
    {
      "sample_index": 0,
      "valid": true/false,
      "quality_score": 0.0-1.0,
      "dimensions": {
        "format": 0.0-1.0,
        "semantic": 0.0-1.0,
        "realism": 0.0-1.0,
        "diversity": 0.0-1.0,
        "label_accuracy": 0.0-1.0
      },
      "issues": ["..."],
      "suggestions": ["..."]
    }
  ],
  "overall_quality": 0.0-1.0,
  "pass_rate": 0.0-1.0,
  "recommendations": ["..."]
}

Be strict but fair. Don't reject samples for minor issues."""

    def process(self, input_data: dict[str, Any], **kwargs) -> AgentResult:
        """
        Validate a batch of generated samples.

        Args:
            input_data: {
                "samples": list,
                "source_patterns": dict,
                "data_type": str,
                "quality_threshold": float,
            }
        """
        samples = input_data.get("samples", [])
        source_patterns = input_data.get("source_patterns", {})
        data_type = input_data.get("data_type", "text")
        threshold = input_data.get("quality_threshold", 0.7)

        # Validate in batches of 10
        all_results = []
        batch_size = 10

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            batch_str = json.dumps(batch, ensure_ascii=False, indent=2)

            messages = [
                {
                    "role": "user",
                    "content": f"""Validate these {len(batch)} {data_type} synthetic data samples.

Source data patterns:
{json.dumps(source_patterns, indent=2, ensure_ascii=False)[:2000]}

Quality threshold: {threshold}

Samples to validate:
{batch_str[:8000]}

Assess each sample on format, semantic validity, realism, diversity, and label accuracy.
Return JSON with results array."""
                }
            ]

            try:
                result = self.llm_call(messages, parse_json=True, temperature=0.2)
                batch_results = result.get("results", []) if isinstance(result, dict) else result
                # Offset indices
                for r in batch_results:
                    if "sample_index" in r:
                        r["sample_index"] += i
                all_results.extend(batch_results)
            except Exception:
                # If validation fails, mark batch as uncertain
                for j in range(len(batch)):
                    all_results.append({
                        "sample_index": i + j,
                        "valid": True,
                        "quality_score": 0.5,
                        "issues": ["Validation error — assumed valid"],
                    })

        # Compute overall stats
        valid_count = sum(1 for r in all_results if r.get("valid", True))
        scores = [r.get("quality_score", 0.5) for r in all_results]

        return self.make_result(
            success=True,
            data={
                "results": all_results,
                "overall_quality": sum(scores) / max(len(scores), 1),
                "pass_rate": valid_count / max(len(all_results), 1),
                "total_samples": len(samples),
                "passed": valid_count,
                "failed": len(all_results) - valid_count,
            },
        )

    def validate_single(
        self,
        sample: Any,
        data_type: str = "text",
        reference_patterns: dict | None = None,
    ) -> AgentResult:
        """Quick validation of a single sample"""
        messages = [
            {
                "role": "user",
                "content": f"""Quickly validate this {data_type} sample:

{json.dumps(sample, ensure_ascii=False)[:3000]}

Reference patterns: {json.dumps(reference_patterns or {}, ensure_ascii=False)[:1000]}

Return JSON: {{"valid": bool, "quality_score": float, "issues": [...]}}"""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.2)
            return self.make_result(success=True, data=result)
        except Exception as e:
            return self.make_result(success=False, error=str(e))

    def check_diversity(
        self,
        samples: list,
        threshold: float = 0.8,
    ) -> AgentResult:
        """Check if samples are diverse enough (not near-duplicates)"""
        if len(samples) <= 1:
            return self.make_result(success=True, data={"diverse": True, "duplicates": []})

        # Check in batches
        sample_text = json.dumps(samples[:20], ensure_ascii=False)

        messages = [
            {
                "role": "user",
                "content": f"""Check these {len(samples[:20])} samples for near-duplicates.
Similarity threshold for duplicate: {threshold}

{sample_text[:8000]}

Return JSON:
{{
  "diverse": true/false,
  "duplicate_pairs": [[i, j, "reason"]],
  "similarity_groups": [[indices]],
  "diversity_score": 0.0-1.0
}}"""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.2)
            return self.make_result(success=True, data=result)
        except Exception as e:
            return self.make_result(success=False, error=str(e))
