"""
Code Agent — generates synthetic code data (functions, classes, snippets, tests)
Specialized for programming language understanding
"""

import json
from typing import Any

from .base_agent import BaseAgent, AgentResult


class CodeAgent(BaseAgent):
    """
    Generates synthetic code data:
    1. Function implementations
    2. Class definitions
    3. Test cases
    4. Bug-fix pairs
    5. Code completion examples
    6. Refactoring examples
    """

    name = "code_agent"
    description = "Generates synthetic code samples"

    def get_system_prompt(self) -> str:
        return """You are a code synthesis expert. You generate realistic, runnable code samples that match the style, patterns, and conventions of existing code.

Rules:
- Code MUST be syntactically valid
- Follow the same naming conventions, style, and patterns as the input
- Include realistic variable names, not placeholders
- Match the complexity level of the source data
- If generating bug-fix pairs, bugs should be subtle and realistic (not obvious syntax errors)
- For tests, follow the same testing framework and patterns

Output JSON:
{
  "samples": [
    {
      "code": "...",
      "language": "...",
      "type": "function|class|test|bug_fix|completion|refactor",
      "description": "What this code does",
      "complexity": "simple|medium|complex"
    }
  ]
}"""

    def process(self, input_data: dict[str, Any], **kwargs) -> AgentResult:
        """
        Generate synthetic code samples.

        Args:
            input_data: {
                "language": str,
                "patterns": dict,
                "count": int,
                "code_type": str,  # function, class, test, bug_fix, etc.
                "reference_code": str,
            }
        """
        language = input_data.get("language", "python")
        patterns = input_data.get("patterns", {})
        count = input_data.get("count", 5)
        code_type = input_data.get("code_type", "function")
        reference_code = input_data.get("reference_code", "")[:6000]

        messages = [
            {
                "role": "user",
                "content": f"""Generate {count} synthetic {language} code samples of type "{code_type}".

Style patterns from existing code:
{json.dumps(patterns, indent=2, ensure_ascii=False)[:3000]}

Reference code (for style matching):
```
{reference_code}
```

Generate {count} diverse {code_type} samples. Each should be realistic, runnable, and match the reference style.
Return as JSON with a "samples" array."""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.6)
            samples = result.get("samples", []) if isinstance(result, dict) else result
            return self.make_result(
                success=True,
                data=samples,
                language=language,
                code_type=code_type,
            )
        except Exception as e:
            return self.make_result(success=False, error=str(e))

    def generate_test_pairs(
        self,
        language: str,
        reference_code: str,
        count: int = 5,
    ) -> AgentResult:
        """Generate code + corresponding test case pairs"""
        messages = [
            {
                "role": "user",
                "content": f"""Generate {count} {language} code + test pairs.

For each pair:
1. Write a function/class
2. Write the corresponding unit tests

Reference style:
```
{reference_code[:4000]}
```

Return JSON:
{{
  "pairs": [
    {{
      "code": "...",
      "test": "...",
      "description": "..."
    }}
  ]
}}"""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.5)
            pairs = result.get("pairs", []) if isinstance(result, dict) else result
            return self.make_result(success=True, data=pairs)
        except Exception as e:
            return self.make_result(success=False, error=str(e))

    def generate_bug_fix_pairs(
        self,
        language: str,
        reference_code: str,
        count: int = 5,
    ) -> AgentResult:
        """Generate buggy code + fix pairs"""
        messages = [
            {
                "role": "user",
                "content": f"""Generate {count} realistic bug-fix pairs for {language}.

Rules:
- Bugs must be SUBTLE (logic errors, off-by-one, wrong operator, missing edge case)
- NOT syntax errors or obvious typos
- The fix should be minimal (1-3 lines changed)
- Include a brief explanation of the bug

Reference style:
```
{reference_code[:4000]}
```

Return JSON:
{{
  "pairs": [
    {{
      "buggy_code": "...",
      "fixed_code": "...",
      "bug_description": "What's wrong",
      "fix_description": "What changed",
      "bug_type": "logic|off_by_one|type|edge_case|concurrency"
    }}
  ]
}}"""
            }
        ]

        try:
            result = self.llm_call(messages, parse_json=True, temperature=0.5)
            pairs = result.get("pairs", []) if isinstance(result, dict) else result
            return self.make_result(success=True, data=pairs)
        except Exception as e:
            return self.make_result(success=False, error=str(e))
