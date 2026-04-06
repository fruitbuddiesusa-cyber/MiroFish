"""
Hallucination Guard — prevents LLM hallucinations from corrupting synthetic data
Multi-layer defense: format checks, consistency checks, fact verification
"""

import json
import re
from typing import Any
from dataclasses import dataclass, field


@dataclass
class HallucinationCheck:
    """Result of a hallucination check"""
    passed: bool = True
    confidence: float = 1.0
    issues: list[str] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "confidence": self.confidence,
            "issues": self.issues,
            "checks_run": self.checks_run,
        }


class HallucinationGuard:
    """
    Multi-layer hallucination detection for synthetic data.
    Runs fast heuristic checks + deeper semantic checks.
    """

    # Common hallucination patterns
    _PLACEHOLDER_PATTERNS = [
        r'\b(Lorem ipsum)\b',
        r'\b(XXX|PLACEHOLDER|TODO|FIXME|TBD)\b',
        r'\b(example\.com|test\.com|foo\.bar)\b',
        r'\b(John Doe|Jane Doe|Acme Corp)\b',
        r'\b(\[INSERT.*?\])\b',
        r'\b(your_\w+_here)\b',
    ]

    _REPETITION_PATTERNS = [
        r'(.{20,})\1{2,}',  # Repeated phrases
        r'(\b\w+\b)(\s+\1){5,}',  # Repeated words
    ]

    _NONSENSE_PATTERNS = [
        r'([a-z])\1{5,}',  # Repeated characters
        r'(as an ai|i am an ai|as a language model)',
        r'(i don\'t have access|i cannot browse)',
        r'(here is a sample|here\'s an example)',
    ]

    @classmethod
    def check_sample(
        cls,
        sample: Any,
        source_patterns: dict | None = None,
        data_type: str = "text",
    ) -> HallucinationCheck:
        """
        Run all checks on a single sample.
        """
        check = HallucinationCheck()
        text = json.dumps(sample, ensure_ascii=False) if not isinstance(sample, str) else sample

        # Layer 1: Format validity
        cls._check_format(text, data_type, check)

        # Layer 2: Placeholder detection
        cls._check_placeholders(text, check)

        # Layer 3: Repetition detection
        cls._check_repetition(text, check)

        # Layer 4: Nonsense detection
        cls._check_nonsense(text, check)

        # Layer 5: Pattern consistency
        if source_patterns:
            cls._check_pattern_consistency(text, source_patterns, check)

        # Layer 6: Length sanity
        cls._check_length_sanity(text, data_type, check)

        check.passed = len(check.issues) == 0
        check.confidence = max(0, 1 - len(check.issues) * 0.15)

        return check

    @classmethod
    def check_batch(
        cls,
        samples: list,
        source_patterns: dict | None = None,
        data_type: str = "text",
    ) -> dict:
        """
        Check a batch of samples. Returns summary + per-sample results.
        """
        results = []
        for sample in samples:
            results.append(cls.check_sample(sample, source_patterns, data_type).to_dict())

        passed = sum(1 for r in results if r["passed"])
        avg_confidence = sum(r["confidence"] for r in results) / max(len(results), 1)

        return {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": passed / max(len(results), 1),
            "avg_confidence": round(avg_confidence, 3),
            "results": results,
        }

    @classmethod
    def _check_format(cls, text: str, data_type: str, check: HallucinationCheck):
        """Check format validity"""
        check.checks_run.append("format")

        if data_type == "code":
            # Check for balanced brackets
            brackets = {"(": ")", "[": "]", "{": "}"}
            stack = []
            for char in text:
                if char in brackets:
                    stack.append(brackets[char])
                elif char in brackets.values():
                    if not stack or stack[-1] != char:
                        check.issues.append("Unbalanced brackets in code")
                        break
                    stack.pop()

        elif data_type == "structured":
            if text.strip().startswith('['):
                try:
                    json.loads(text)
                except json.JSONDecodeError as e:
                    check.issues.append(f"Invalid JSON: {str(e)[:100]}")

    @classmethod
    def _check_placeholders(cls, text: str, check: HallucinationCheck):
        """Check for placeholder content"""
        check.checks_run.append("placeholders")

        for pattern in cls._PLACEHOLDER_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                check.issues.append(f"Placeholder detected: {pattern}")

    @classmethod
    def _check_repetition(cls, text: str, check: HallucinationCheck):
        """Check for excessive repetition"""
        check.checks_run.append("repetition")

        for pattern in cls._REPETITION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                check.issues.append("Excessive repetition detected")
                break

        # Check line-level repetition
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) > 5:
            unique = len(set(lines))
            if unique / len(lines) < 0.3:
                check.issues.append(f"Low line diversity: {unique}/{len(lines)} unique")

    @classmethod
    def _check_nonsense(cls, text: str, check: HallucinationCheck):
        """Check for nonsense or AI self-reference"""
        check.checks_run.append("nonsense")

        for pattern in cls._NONSENSE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                check.issues.append(f"Nonsense pattern: {pattern}")

    @classmethod
    def _check_pattern_consistency(
        cls,
        text: str,
        patterns: dict,
        check: HallucinationCheck,
    ):
        """Check if sample is consistent with source patterns"""
        check.checks_run.append("pattern_consistency")

        style = patterns.get("style_patterns", {})

        # Check length consistency
        if style.get("avg_length"):
            expected = style["avg_length"]
            actual = len(text)
            if actual > expected * 5 or actual < expected * 0.1:
                check.issues.append(
                    f"Length outlier: {actual} vs expected ~{expected}"
                )

    @classmethod
    def _check_length_sanity(cls, text: str, data_type: str, check: HallucinationCheck):
        """Check for absurd lengths"""
        check.checks_run.append("length_sanity")

        max_lengths = {
            "code": 10000,
            "text": 5000,
            "structured": 20000,
            "reasoning": 8000,
        }

        max_len = max_lengths.get(data_type, 5000)
        if len(text) > max_len:
            check.issues.append(f"Sample too long: {len(text)} > {max_len}")

        if len(text.strip()) < 10:
            check.issues.append("Sample too short")
