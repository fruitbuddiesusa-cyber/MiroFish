"""
Orchestrator — multi-agent coordinator for synthetic data generation
Manages the full pipeline: analyze → explore → generate → validate → filter
"""

import json
import time
from typing import Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_agent import AgentResult
from .pattern_agent import PatternAgent
from .gap_agent import GapAgent
from .edge_agent import EdgeAgent
from .code_agent import CodeAgent
from .reasoning_agent import ReasoningAgent
from .text_agent import TextAgent
from .validator_agent import ValidatorAgent
from .novelty_agent import NoveltyAgent

from ..config.settings import Settings
from ..utils.data_detector import DataDetector, DataType, DataProfile
from ..utils.cost_tracker import CostTracker


class PipelineStage(str, Enum):
    ANALYZE = "analyze"
    EXPLORE = "explore"
    GENERATE = "generate"
    VALIDATE = "validate"
    FILTER = "filter"
    EXPORT = "export"


@dataclass
class PipelineState:
    """State of the generation pipeline"""
    stage: PipelineStage = PipelineStage.ANALYZE
    progress: float = 0.0
    message: str = ""
    data_profile: DataProfile | None = None
    patterns: dict = field(default_factory=dict)
    gaps: dict = field(default_factory=dict)
    generated_samples: list = field(default_factory=list)
    validated_samples: list = field(default_factory=list)
    novel_samples: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    cost_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "progress": self.progress,
            "message": self.message,
            "data_type": self.data_profile.data_type.value if self.data_profile else None,
            "patterns_count": len(self.patterns),
            "gaps_count": len(self.gaps.get("coverage_gaps", [])),
            "generated_count": len(self.generated_samples),
            "validated_count": len(self.validated_samples),
            "novel_count": len(self.novel_samples),
            "errors": self.errors,
            "cost_summary": self.cost_summary,
        }


class Orchestrator:
    """
    Multi-agent coordinator.
    Runs the full synthetic data generation pipeline.

    Pipeline:
    1. ANALYZE: Detect data type → extract patterns → find gaps
    2. EXPLORE: Plan what to generate (gap prioritization)
    3. GENERATE: Dispatch to specialized agents (parallel)
    4. VALIDATE: Quality check all generated data
    5. FILTER: Remove duplicates, low-quality, non-novel
    6. EXPORT: Format output
    """

    def __init__(self):
        # Initialize agents
        self.pattern_agent = PatternAgent()
        self.gap_agent = GapAgent()
        self.edge_agent = EdgeAgent()
        self.code_agent = CodeAgent()
        self.reasoning_agent = ReasoningAgent()
        self.text_agent = TextAgent()
        self.validator_agent = ValidatorAgent()
        self.novelty_agent = NoveltyAgent()

        self._state = PipelineState()

    @property
    def state(self) -> PipelineState:
        return self._state

    def run_pipeline(
        self,
        input_text: str,
        filename: str | None = None,
        target_count: int = 50,
        quality_threshold: float | None = None,
        novelty_threshold: float | None = None,
        include_edge_cases: bool = True,
        edge_case_ratio: float = 0.2,
        progress_callback: Callable | None = None,
    ) -> dict:
        """
        Run the full synthetic data generation pipeline.

        Args:
            input_text: Source data text
            filename: Optional filename for type detection
            target_count: How many samples to generate
            quality_threshold: Min quality score (0-1)
            novelty_threshold: Min novelty score (0-1)
            include_edge_cases: Whether to generate edge cases
            edge_case_ratio: Fraction of target for edge cases
            progress_callback: fn(stage, progress, message)

        Returns:
            Dict with generated samples and metadata
        """
        quality_thresh = quality_threshold or Settings.QUALITY_THRESHOLD
        novelty_thresh = novelty_threshold or Settings.NOVELTY_THRESHOLD

        CostTracker.reset()

        try:
            # ========== STAGE 1: ANALYZE ==========
            self._update(PipelineStage.ANALYZE, 0.1, "Detecting data type...", progress_callback)

            profile = DataDetector.detect(input_text, filename)
            self._state.data_profile = profile

            self._update(PipelineStage.ANALYZE, 0.2, f"Detected: {profile.data_type.value}", progress_callback)

            # Extract patterns
            self._update(PipelineStage.ANALYZE, 0.3, "Extracting patterns...", progress_callback)
            pattern_result = self.pattern_agent.process({
                "text": input_text,
                "profile": profile,
            })
            if pattern_result.success:
                self._state.patterns = pattern_result.data
            else:
                self._state.errors.append(f"Pattern analysis failed: {pattern_result.error}")

            # Cross-chunk patterns for large data
            if len(input_text) > 10000:
                self._update(PipelineStage.ANALYZE, 0.4, "Analyzing cross-chunk patterns...", progress_callback)
                from ..utils.token_counter import TokenCounter
                chunks = TokenCounter.split_by_tokens(input_text, chunk_size=2000)
                chunk_result = self.pattern_agent.analyze_chunks(chunks, profile.data_type.value)
                if chunk_result.success:
                    self._state.patterns["cross_chunk"] = chunk_result.data

            # ========== STAGE 2: EXPLORE ==========
            self._update(PipelineStage.EXPLORE, 0.5, "Identifying data gaps...", progress_callback)

            gap_result = self.gap_agent.process({
                "text": input_text,
                "patterns": self._state.patterns,
                "existing_count": profile.row_count,
            })
            if gap_result.success:
                self._state.gaps = gap_result.data
            else:
                self._state.errors.append(f"Gap analysis failed: {gap_result.error}")

            # Prioritize gaps
            self._update(PipelineStage.EXPLORE, 0.6, "Prioritizing generation targets...", progress_callback)

            edge_count = int(target_count * edge_case_ratio) if include_edge_cases else 0
            main_count = target_count - edge_count

            priority_result = self.gap_agent.prioritize_gaps(
                self._state.gaps,
                target_count=main_count,
            )
            fill_plan = priority_result.data if priority_result.success else []

            # ========== STAGE 3: GENERATE ==========
            self._update(PipelineStage.GENERATE, 0.65, "Generating samples...", progress_callback)

            all_samples = []

            # Dispatch to specialized agent based on data type
            if profile.data_type == DataType.CODE:
                code_result = self.code_agent.process({
                    "language": profile.language.value if profile.language else "python",
                    "patterns": self._state.patterns,
                    "count": main_count,
                    "code_type": "function",
                    "reference_code": input_text[:6000],
                })
                if code_result.success:
                    all_samples.extend(code_result.data)

            elif profile.data_type == DataType.REASONING:
                reason_result = self.reasoning_agent.process({
                    "domain": "general",
                    "difficulty": "medium",
                    "count": main_count,
                    "patterns": self._state.patterns,
                    "reference_data": input_text[:6000],
                })
                if reason_result.success:
                    all_samples.extend(reason_result.data)

            else:
                # General text generation
                text_result = self.text_agent.process({
                    "patterns": self._state.patterns,
                    "task_type": "generate",
                    "count": main_count,
                    "reference_data": input_text[:8000],
                })
                if text_result.success:
                    all_samples.extend(text_result.data)

            self._update(
                PipelineStage.GENERATE, 0.75,
                f"Generated {len(all_samples)} samples",
                progress_callback,
            )

            # Edge cases
            if include_edge_cases and edge_count > 0:
                self._update(PipelineStage.GENERATE, 0.8, "Generating edge cases...", progress_callback)

                if profile.data_type == DataType.CODE:
                    edge_result = self.edge_agent.generate_for_code(
                        language=profile.language.value if profile.language else "python",
                        patterns=self._state.patterns,
                        count=edge_count,
                    )
                else:
                    edge_result = self.edge_agent.process({
                        "patterns": self._state.patterns,
                        "data_type": profile.data_type.value,
                        "count": edge_count,
                    })

                if edge_result.success:
                    all_samples.extend(edge_result.data)

            self._state.generated_samples = all_samples

            # ========== STAGE 4: VALIDATE ==========
            self._update(PipelineStage.VALIDATE, 0.85, "Validating quality...", progress_callback)

            val_result = self.validator_agent.process({
                "samples": all_samples,
                "source_patterns": self._state.patterns,
                "data_type": profile.data_type.value,
                "quality_threshold": quality_thresh,
            })

            # Filter by quality
            if val_result.success:
                val_data = val_result.data
                valid_indices = set()
                for r in val_data.get("results", []):
                    if r.get("valid", True) and r.get("quality_score", 0) >= quality_thresh:
                        valid_indices.add(r.get("sample_index", -1))

                validated = [s for i, s in enumerate(all_samples) if i in valid_indices]
            else:
                validated = all_samples  # Keep all if validation fails
                self._state.errors.append(f"Validation partial failure: {val_result.error}")

            self._state.validated_samples = validated

            # ========== STAGE 5: FILTER (NOVELTY) ==========
            self._update(PipelineStage.FILTER, 0.9, "Checking novelty...", progress_callback)

            novelty_result = self.novelty_agent.filter_novel(
                samples=validated,
                source_samples=self._extract_source_samples(input_text),
                threshold=novelty_thresh,
            )

            if novelty_result.success:
                novel = novelty_result.data.get("novel_samples", validated)
            else:
                novel = validated

            self._state.novel_samples = novel

            # ========== DONE ==========
            self._state.cost_summary = CostTracker.get_summary()
            self._update(PipelineStage.EXPORT, 1.0, f"Complete! {len(novel)} samples generated", progress_callback)

            return {
                "success": True,
                "samples": novel,
                "total_generated": len(all_samples),
                "quality_passed": len(validated),
                "novel_passed": len(novel),
                "data_type": profile.data_type.value,
                "patterns": self._state.patterns,
                "gaps": self._state.gaps,
                "cost_summary": self._state.cost_summary,
                "errors": self._state.errors,
            }

        except Exception as e:
            self._state.errors.append(str(e))
            return {
                "success": False,
                "error": str(e),
                "samples": self._state.novel_samples or self._state.generated_samples,
                "cost_summary": CostTracker.get_summary(),
            }

    def _update(
        self,
        stage: PipelineStage,
        progress: float,
        message: str,
        callback: Callable | None,
    ):
        """Update pipeline state and notify callback"""
        self._state.stage = stage
        self._state.progress = progress
        self._state.message = message
        if callback:
            callback(stage.value, progress, message)

    def _extract_source_samples(self, text: str, max_samples: int = 15) -> list[str]:
        """Extract representative samples from source for novelty comparison"""
        from ..utils.token_counter import TokenCounter
        chunks = TokenCounter.split_by_tokens(text, chunk_size=500)
        if len(chunks) <= max_samples:
            return chunks
        step = len(chunks) // max_samples
        return [chunks[i] for i in range(0, len(chunks), step)][:max_samples]
