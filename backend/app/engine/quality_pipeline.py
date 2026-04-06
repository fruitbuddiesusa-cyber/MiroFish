"""
Quality Pipeline — multi-stage quality filtering for generated data
Runs: format check → hallucination guard → diversity check → novelty scoring → final filter
"""

import json
from typing import Any, Callable
from dataclasses import dataclass, field

from ..engine.hallucination_guard import HallucinationGuard
from ..engine.novelty_detector import NoveltyDetector
from ..agents.validator_agent import ValidatorAgent
from ..config.settings import Settings


@dataclass
class QualityReport:
    """Quality report for a pipeline run"""
    total_input: int = 0
    passed_format: int = 0
    passed_hallucination: int = 0
    passed_diversity: int = 0
    passed_novelty: int = 0
    passed_llm_validation: int = 0
    final_output: int = 0
    stage_reports: dict = field(default_factory=dict)
    cost_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_input": self.total_input,
            "passed_format": self.passed_format,
            "passed_hallucination": self.passed_hallucination,
            "passed_diversity": self.passed_diversity,
            "passed_novelty": self.passed_novelty,
            "passed_llm_validation": self.passed_llm_validation,
            "final_output": self.final_output,
            "pass_rate": self.final_output / max(self.total_input, 1),
            "stage_reports": self.stage_reports,
            "cost_summary": self.cost_summary,
        }


class QualityPipeline:
    """
    Multi-stage quality pipeline.

    Stages:
    1. Format Check — fast heuristic format validation
    2. Hallucination Guard — detect placeholders, nonsense, repetition
    3. Diversity Check — remove near-duplicates
    4. Novelty Scoring — score novelty vs source data
    5. LLM Validation — deep quality check (optional, expensive)
    6. Final Filter — apply thresholds and output
    """

    def __init__(self):
        self.validator = ValidatorAgent()

    def run(
        self,
        samples: list[Any],
        source_samples: list[str] | None = None,
        source_patterns: dict | None = None,
        data_type: str = "text",
        quality_threshold: float | None = None,
        novelty_threshold: float | None = None,
        use_llm_validation: bool = True,
        progress_callback: Callable | None = None,
    ) -> dict:
        """
        Run the full quality pipeline.

        Args:
            samples: Generated samples to filter
            source_samples: Original data samples for novelty comparison
            source_patterns: Discovered patterns for validation
            data_type: Data type (code, text, structured, etc.)
            quality_threshold: Min quality score
            novelty_threshold: Min novelty score
            use_llm_validation: Whether to use expensive LLM validation
            progress_callback: fn(stage, progress, message)

        Returns:
            {"samples": [...], "report": QualityReport}
        """
        q_thresh = quality_threshold or Settings.QUALITY_THRESHOLD
        n_thresh = novelty_threshold or Settings.NOVELTY_THRESHOLD

        report = QualityReport(total_input=len(samples))
        current = list(samples)

        # ========== Stage 1: Format Check ==========
        self._notify(progress_callback, "format", 0.1, "Running format checks...")

        format_passed = []
        for sample in current:
            text = json.dumps(sample) if not isinstance(sample, str) else sample
            if len(text.strip()) >= 10:  # Minimum length
                format_passed.append(sample)

        report.passed_format = len(format_passed)
        report.stage_reports["format"] = {
            "input": len(current),
            "output": len(format_passed),
            "rejected": len(current) - len(format_passed),
        }
        current = format_passed

        # ========== Stage 2: Hallucination Guard ==========
        self._notify(progress_callback, "hallucination", 0.2, "Checking for hallucinations...")

        hallo_result = HallucinationGuard.check_batch(
            current,
            source_patterns=source_patterns,
            data_type=data_type,
        )
        hallo_passed = [
            s for s, r in zip(current, hallo_result["results"])
            if r["passed"]
        ]

        report.passed_hallucination = len(hallo_passed)
        report.stage_reports["hallucination"] = {
            "input": len(current),
            "output": len(hallo_passed),
            "avg_confidence": hallo_result["avg_confidence"],
        }
        current = hallo_passed

        # ========== Stage 3: Diversity (Dedup) ==========
        self._notify(progress_callback, "diversity", 0.3, "Removing duplicates...")

        if current and isinstance(current[0], str):
            deduped = NoveltyDetector.dedup(current, threshold=0.85)
        else:
            texts = [json.dumps(s) if not isinstance(s, str) else s for s in current]
            deduped_texts = NoveltyDetector.dedup(texts, threshold=0.85)
            deduped_set = set(deduped_texts)
            deduped = [s for s in current if (json.dumps(s) if not isinstance(s, str) else s) in deduped_set]

        report.passed_diversity = len(deduped)
        report.stage_reports["diversity"] = {
            "input": len(current),
            "output": len(deduped),
            "duplicates_removed": len(current) - len(deduped),
        }
        current = deduped

        # ========== Stage 4: Novelty Scoring ==========
        self._notify(progress_callback, "novelty", 0.5, "Scoring novelty...")

        if source_samples:
            novelty_results = NoveltyDetector.filter_novel(
                candidates=[json.dumps(s) if not isinstance(s, str) else s for s in current],
                source_samples=source_samples,
                threshold=n_thresh,
            )
            novel_passed = [
                s for s, r in zip(current, novelty_results)
                if r["kept"]
            ]
            avg_novelty = sum(r["overall_novelty"] for r in novelty_results) / max(len(novelty_results), 1)
        else:
            novel_passed = current
            avg_novelty = 1.0

        report.passed_novelty = len(novel_passed)
        report.stage_reports["novelty"] = {
            "input": len(current),
            "output": len(novel_passed),
            "avg_novelty": round(avg_novelty, 3),
            "threshold": n_thresh,
        }
        current = novel_passed

        # ========== Stage 5: LLM Validation (optional) ==========
        if use_llm_validation and current:
            self._notify(progress_callback, "llm_validation", 0.7, "Running LLM validation...")

            val_result = self.validator.process({
                "samples": current,
                "source_patterns": source_patterns or {},
                "data_type": data_type,
                "quality_threshold": q_thresh,
            })

            if val_result.success:
                val_data = val_result.data
                valid_indices = set()
                for r in val_data.get("results", []):
                    if r.get("valid", True) and r.get("quality_score", 0) >= q_thresh:
                        valid_indices.add(r.get("sample_index", -1))

                llm_passed = [s for i, s in enumerate(current) if i in valid_indices]
                report.stage_reports["llm_validation"] = {
                    "input": len(current),
                    "output": len(llm_passed),
                    "pass_rate": val_data.get("pass_rate", 0),
                    "avg_quality": val_data.get("overall_quality", 0),
                }
            else:
                llm_passed = current
                report.stage_reports["llm_validation"] = {
                    "input": len(current),
                    "output": len(current),
                    "error": "Validation failed, keeping all",
                }

            report.passed_llm_validation = len(llm_passed)
            current = llm_passed
        else:
            report.passed_llm_validation = len(current)

        # ========== Final ==========
        report.final_output = len(current)
        self._notify(progress_callback, "done", 1.0, f"Quality pipeline complete: {len(current)} samples")

        return {
            "samples": current,
            "report": report.to_dict(),
        }

    @staticmethod
    def _notify(callback, stage, progress, message):
        if callback:
            callback(stage, progress, message)
