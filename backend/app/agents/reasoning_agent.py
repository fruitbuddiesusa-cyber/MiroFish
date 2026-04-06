"""
Reasoning Agent — generates synthetic reasoning chain data
For math, logic, multi-step problem solving
"""

import json
from typing import Any

from .base_agent import BaseAgent, AgentResult


class ReasoningAgent(BaseAgent):
    """
    Generates synthetic reasoning data:
    1. Step-by-step problem solutions
    2. Chain-of-thought reasoning traces
    3. Logical deduction chains
    4. Multi-hop reasoning examples
    5. Error-correction reasoning
    """

    name = "reasoning_agent"
    description = "Generates synthetic reasoning chains"

    def get_system_prompt(self) -> str:
        return """You are a reasoning chain generator. You produce synthetic data showing HOW to solve problems step by step.

Your reasoning chains must:
- Show explicit intermediate steps (not jump to conclusions)
- Include self-checks and verification steps
- Handle uncertainty honestly ("I'm not sure, but...")
- Show backtracking when a path doesn't work
- Use diverse reasoning strategies (algebraic, geometric, logical, analogical)

Output JSON:
{
  "samples": [
    {
      "problem": "...",
      "reasoning_steps": ["step1", "step2", ...],
      "answer": "...",
      "strategy": "algebraic|logical|analogical|exhaustive|heuristic",
      "difficulty": "easy|medium|hard",
      "self_verification": "How to check the answer"
    }
  ]
}"""

    def process(self, input_data: dict[str, Any], **kwargs) -> AgentResult:
        """
        Generate reasoning chain samples.

        Args:
            input_data: {
                "domain": str,        # math, logic, science, etc.
                "difficulty": str,     # easy, medium, hard
                "count": int,
                "patterns": dict,
                "reference_data": str,
            }
        """
        domain = input_data.get("domain", "general")
        difficulty = input_data.get("difficulty", "medium")
        count = input_data.get("count", 5)
        patterns = input_data.get("patterns", {})
        reference = input_data.get("reference_data", "")[:6000]

        messages = [
            {
                "role": "user",
                "content": f"""Generate {count} synthetic reasoning chain samples for the "{domain}" domain at "{difficulty}" difficulty.

Patterns from existing data:
{json.dumps(patterns, indent=2, ensure_ascii=False)[:2000]}

Reference data (for style/topic matching):
{reference}

Generate {count} diverse problems with full step-by-step reasoning. Include self-verification.
Return as JSON with a "samples" array."""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.6)
            samples = result.get("samples", []) if isinstance(result, dict) else result
            return self.make_result(
                success=True,
                data=samples,
                domain=domain,
                difficulty=difficulty,
            )
        except Exception as e:
            return self.make_result(success=False, error=str(e))

    def generate_cot_variations(
        self,
        problem: str,
        correct_answer: str,
        count: int = 3,
    ) -> AgentResult:
        """
        Generate multiple reasoning paths for the same problem.
        Useful for training robust reasoning models.
        """
        messages = [
            {
                "role": "user",
                "content": f"""Given this problem and correct answer, generate {count} DIFFERENT reasoning paths that all arrive at the same answer.

Problem: {problem}
Correct Answer: {correct_answer}

Each path should use a DIFFERENT strategy:
- Path 1: Direct/step-by-step
- Path 2: Work backwards from the answer
- Path 3: Analogical or creative approach

Return JSON:
{{
  "paths": [
    {{
      "strategy": "...",
      "steps": ["..."],
      "arrives_at_correct_answer": true
    }}
  ]
}}"""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.7)
            paths = result.get("paths", []) if isinstance(result, dict) else result
            return self.make_result(success=True, data=paths)
        except Exception as e:
            return self.make_result(success=False, error=str(e))

    def generate_with_distractors(
        self,
        domain: str,
        count: int = 5,
    ) -> AgentResult:
        """Generate reasoning problems with plausible wrong answers"""
        messages = [
            {
                "role": "user",
                "content": f"""Generate {count} reasoning problems in the "{domain}" domain, each with:
1. The correct reasoning chain and answer
2. 2-3 plausible but INCORRECT reasoning chains (distractors)

The distractors should be tempting — common mistakes that look reasonable.

Return JSON:
{{
  "samples": [
    {{
      "problem": "...",
      "correct_reasoning": ["step1", ...],
      "correct_answer": "...",
      "distractors": [
        {{"reasoning": ["..."], "answer": "...", "error_type": "misconception|calculation|assumption"}}
      ]
    }}
  ]
}}"""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.6)
            samples = result.get("samples", []) if isinstance(result, dict) else result
            return self.make_result(success=True, data=samples)
        except Exception as e:
            return self.make_result(success=False, error=str(e))
