"""
Novelty Detector — fast heuristic novelty scoring
Complements the LLM-based NoveltyAgent with speed-optimized checks
"""

import re
from collections import Counter
from typing import Any


class NoveltyDetector:
    """
    Fast heuristic novelty detection.
    Works WITHOUT LLM calls for speed.
    """

    @classmethod
    def compute_novelty_score(
        cls,
        candidate: str,
        source_samples: list[str],
        generated_samples: list[str] | None = None,
    ) -> dict:
        """
        Compute novelty score of a candidate vs source and existing generated.
        Returns dict with scores and breakdown.
        """
        source_sim = cls._max_similarity(candidate, source_samples)
        gen_sim = 0.0
        if generated_samples:
            gen_sim = cls._max_similarity(candidate, generated_samples)

        # Novelty = 1 - max similarity
        source_novelty = 1 - source_sim
        gen_novelty = 1 - gen_sim

        # Vocabulary novelty
        vocab_novelty = cls._vocab_novelty(candidate, source_samples)

        # Structure novelty
        struct_novelty = cls._structure_novelty(candidate, source_samples)

        overall = (
            source_novelty * 0.4 +
            gen_novelty * 0.2 +
            vocab_novelty * 0.2 +
            struct_novelty * 0.2
        )

        return {
            "overall_novelty": round(overall, 3),
            "source_novelty": round(source_novelty, 3),
            "generation_novelty": round(gen_novelty, 3),
            "vocab_novelty": round(vocab_novelty, 3),
            "structure_novelty": round(struct_novelty, 3),
            "max_source_similarity": round(source_sim, 3),
        }

    @classmethod
    def filter_novel(
        cls,
        candidates: list[str],
        source_samples: list[str],
        threshold: float = 0.6,
    ) -> list[dict]:
        """
        Filter candidates to keep only novel ones.
        Returns list of {sample, novelty_score, ...}
        """
        results = []
        generated_so_far = []

        for candidate in candidates:
            score = cls.compute_novelty_score(candidate, source_samples, generated_so_far)
            if score["overall_novelty"] >= threshold:
                results.append({
                    "sample": candidate,
                    **score,
                    "kept": True,
                })
                generated_so_far.append(candidate)
            else:
                results.append({
                    "sample": candidate,
                    **score,
                    "kept": False,
                })

        return results

    @classmethod
    def _max_similarity(cls, text: str, references: list[str]) -> float:
        """
        Compute max similarity to any reference.
        Uses Jaccard similarity on word n-grams (fast).
        """
        if not references:
            return 0.0

        text_ngrams = cls._word_ngrams(text, 3)
        if not text_ngrams:
            return 0.0

        max_sim = 0.0
        for ref in references:
            ref_ngrams = cls._word_ngrams(ref, 3)
            if not ref_ngrams:
                continue

            intersection = len(text_ngrams & ref_ngrams)
            union = len(text_ngrams | ref_ngrams)
            sim = intersection / union if union > 0 else 0
            max_sim = max(max_sim, sim)

        return max_sim

    @classmethod
    def _word_ngrams(cls, text: str, n: int = 3) -> set:
        """Extract word-level n-grams"""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < n:
            return set(words)
        return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}

    @classmethod
    def _vocab_novelty(cls, candidate: str, references: list[str]) -> float:
        """
        Score based on unique vocabulary not in references.
        """
        cand_words = set(re.findall(r'\b\w+\b', candidate.lower()))
        if not cand_words:
            return 0.0

        ref_words = set()
        for ref in references:
            ref_words.update(re.findall(r'\b\w+\b', ref.lower()))

        new_words = cand_words - ref_words
        return len(new_words) / len(cand_words)

    @classmethod
    def _structure_novelty(cls, candidate: str, references: list[str]) -> float:
        """
        Score based on structural differences.
        Compares: line count, indent pattern, punctuation mix, etc.
        """
        if not references:
            return 1.0

        def struct_features(text: str) -> dict:
            lines = text.split('\n')
            return {
                "line_count": len(lines),
                "avg_line_len": sum(len(l) for l in lines) / max(len(lines), 1),
                "indent_ratio": sum(1 for l in lines if l.startswith((' ', '\t'))) / max(len(lines), 1),
                "comma_ratio": text.count(',') / max(len(text), 1),
                "paren_ratio": text.count('(') / max(len(text), 1),
            }

        cand_feat = struct_features(candidate)
        ref_feats = [struct_features(r) for r in references[:10]]

        # Average structural distance
        distances = []
        for rf in ref_feats:
            dist = sum(
                abs(cand_feat[k] - rf[k]) / max(abs(cand_feat[k]) + abs(rf[k]), 1)
                for k in cand_feat
            ) / len(cand_feat)
            distances.append(dist)

        avg_dist = sum(distances) / len(distances)
        return min(avg_dist * 3, 1.0)  # Scale up

    @classmethod
    def dedup(
        cls,
        samples: list[str],
        threshold: float = 0.85,
    ) -> list[str]:
        """
        Remove near-duplicates from a list.
        """
        if len(samples) <= 1:
            return samples

        kept = [samples[0]]
        for sample in samples[1:]:
            max_sim = max(
                cls._jaccard(sample, k) for k in kept
            )
            if max_sim < threshold:
                kept.append(sample)

        return kept

    @classmethod
    def _jaccard(cls, a: str, b: str) -> float:
        """Word-level Jaccard similarity"""
        a_words = set(re.findall(r'\b\w+\b', a.lower()))
        b_words = set(re.findall(r'\b\w+\b', b.lower()))
        if not a_words or not b_words:
            return 0.0
        return len(a_words & b_words) / len(a_words | b_words)
