"""
Novelty Agent — detects and scores true novelty in synthetic data
Ensures generated data adds value beyond simple copying
"""

import json
from typing import Any

from .base_agent import BaseAgent, AgentResult


class NoveltyAgent(BaseAgent):
    """
    Scores novelty of synthetic data:
    1. Novelty vs source data (not a copy)
    2. Novelty vs existing generated data (not redundant)
    3. Semantic novelty (new combinations, concepts)
    4. Structural novelty (new formats, patterns)
    5. Usefulness score (is the novelty valuable?)
    """

    name = "novelty_agent"
    description = "Scores and ensures true novelty"
    use_boost = True

    def get_system_prompt(self) -> str:
        return """You are a novelty assessment expert. You evaluate whether synthetic data samples are TRULY novel — adding new information, patterns, or coverage beyond the source data.

Novelty dimensions:
1. **Lexical Novelty**: New vocabulary, phrases, or expressions not in source
2. **Semantic Novelty**: New concepts, relationships, or domain knowledge
3. **Structural Novelty**: New formats, patterns, or organizational styles
4. **Combinatorial Novelty**: New combinations of existing elements
5. **Edge Novelty**: Covers previously uncovered edge cases or boundary conditions

Scoring:
- 0.0-0.3: Low novelty (essentially a paraphrase or copy)
- 0.3-0.6: Moderate novelty (meaningful variation)
- 0.6-0.8: High novelty (new ground covered)
- 0.8-1.0: Exceptional novelty (genuinely new territory)

Output JSON:
{
  "scores": [
    {
      "sample_index": 0,
      "overall_novelty": 0.0-1.0,
      "dimensions": {
        "lexical": 0.0-1.0,
        "semantic": 0.0-1.0,
        "structural": 0.0-1.0,
        "combinatorial": 0.0-1.0,
        "edge": 0.0-1.0
      },
      "is_novel": true/false,
      "novelty_type": "...",
      "explanation": "..."
    }
  ],
  "average_novelty": 0.0-1.0,
  "novel_count": N,
  "recommendations": ["..."]
}"""

    def process(self, input_data: dict[str, Any], **kwargs) -> AgentResult:
        """
        Score novelty of generated samples against source.

        Args:
            input_data: {
                "generated_samples": list,
                "source_samples": list,
                "novelty_threshold": float,
            }
        """
        generated = input_data.get("generated_samples", [])
        source = input_data.get("source_samples", [])
        threshold = input_data.get("novelty_threshold", 0.6)

        # Process in batches
        all_scores = []
        batch_size = 8

        for i in range(0, len(generated), batch_size):
            batch = generated[i:i + batch_size]

            messages = [
                {
                    "role": "user",
                    "content": f"""Score the novelty of these generated samples against the source data.

Source samples (for comparison):
{json.dumps(source[:10], ensure_ascii=False)[:4000]}

Generated samples to score:
{json.dumps(batch, ensure_ascii=False)[:6000]}

Novelty threshold: {threshold}

For each generated sample, score how novel it is compared to the source.
Return JSON with scores array."""
                }
            ]

            try:
                result = self.llm_call(messages, parse_json=True, temperature=0.3)
                batch_scores = result.get("scores", []) if isinstance(result, dict) else result
                for s in batch_scores:
                    if "sample_index" in s:
                        s["sample_index"] += i
                all_scores.extend(batch_scores)
            except Exception:
                continue

        novel_count = sum(1 for s in all_scores if s.get("overall_novelty", 0) >= threshold)
        avg_novelty = sum(s.get("overall_novelty", 0) for s in all_scores) / max(len(all_scores), 1)

        return self.make_result(
            success=True,
            data={
                "scores": all_scores,
                "average_novelty": round(avg_novelty, 3),
                "novel_count": novel_count,
                "total_scored": len(all_scores),
                "novel_rate": novel_count / max(len(all_scores), 1),
            },
        )

    def filter_novel(
        self,
        samples: list,
        source_samples: list,
        threshold: float = 0.6,
    ) -> AgentResult:
        """Filter samples to keep only novel ones"""
        result = self.process({
            "generated_samples": samples,
            "source_samples": source_samples,
            "novelty_threshold": threshold,
        })

        if not result.success:
            return result

        scores = result.data.get("scores", [])
        novel_indices = {
            s["sample_index"]
            for s in scores
            if s.get("overall_novelty", 0) >= threshold
        }

        novel_samples = [s for i, s in enumerate(samples) if i in novel_indices]

        return self.make_result(
            success=True,
            data={
                "novel_samples": novel_samples,
                "kept": len(novel_samples),
                "filtered": len(samples) - len(novel_samples),
            },
        )

    def intra_batch_novelty(
        self,
        samples: list,
        threshold: float = 0.5,
    ) -> AgentResult:
        """Check novelty WITHIN a batch (are samples diverse from each other?)"""
        if len(samples) <= 1:
            return self.make_result(success=True, data={"diverse": True, "redundant_pairs": []})

        messages = [
            {
                "role": "user",
                "content": f"""Check if these {len(samples[:15])} samples are diverse from EACH OTHER (not just from source).

Samples:
{json.dumps(samples[:15], ensure_ascii=False)[:8000]}

Find redundant/near-duplicate pairs within this batch.
Return JSON:
{{
  "diverse": true/false,
  "redundant_pairs": [[i, j, similarity_score, "reason"]],
  "unique_count": N,
  "intra_diversity_score": 0.0-1.0
}}"""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.2)
            return self.make_result(success=True, data=result)
        except Exception as e:
            return self.make_result(success=False, error=str(e))
