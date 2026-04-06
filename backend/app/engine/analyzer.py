"""
Analyzer — deep data analysis for understanding input characteristics
Combines statistical analysis with LLM-powered semantic analysis
"""

import json
import re
from collections import Counter
from typing import Any
from dataclasses import dataclass, field

from ..utils.data_detector import DataDetector, DataProfile, DataType
from ..utils.token_counter import TokenCounter


@dataclass
class DataAnalysis:
    """Complete analysis result"""
    profile: DataProfile
    statistics: dict[str, Any] = field(default_factory=dict)
    vocabulary: dict[str, Any] = field(default_factory=dict)
    structure: dict[str, Any] = field(default_factory=dict)
    complexity: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "profile": {
                "data_type": self.profile.data_type.value,
                "language": self.profile.language.value if self.profile.language else None,
                "row_count": self.profile.row_count,
                "char_count": self.profile.char_count,
            },
            "statistics": self.statistics,
            "vocabulary": self.vocabulary,
            "structure": self.structure,
            "complexity": self.complexity,
        }


class Analyzer:
    """
    Deep data analysis combining statistical + semantic methods.
    """

    @classmethod
    def analyze(cls, text: str, filename: str | None = None) -> DataAnalysis:
        """
        Full analysis of input text.
        """
        profile = DataDetector.detect(text, filename)

        analysis = DataAnalysis(profile=profile)
        analysis.statistics = cls._basic_stats(text)
        analysis.vocabulary = cls._vocab_analysis(text)
        analysis.complexity = cls._complexity_analysis(text, profile)

        if profile.data_type == DataType.CODE:
            analysis.structure = cls._code_structure(text)
        elif profile.data_type == DataType.STRUCTURED:
            analysis.structure = cls._structured_analysis(text)
        else:
            analysis.structure = cls._text_structure(text)

        return analysis

    @classmethod
    def _basic_stats(cls, text: str) -> dict:
        """Basic statistical analysis"""
        lines = text.split('\n')
        words = text.split()
        non_empty = [l for l in lines if l.strip()]

        line_lengths = [len(l) for l in non_empty]
        word_lengths = [len(w) for w in words]

        return {
            "total_chars": len(text),
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty),
            "total_words": len(words),
            "total_tokens_est": TokenCounter.count(text),
            "avg_line_length": sum(line_lengths) / max(len(line_lengths), 1),
            "max_line_length": max(line_lengths) if line_lengths else 0,
            "avg_word_length": sum(word_lengths) / max(len(word_lengths), 1),
            "unique_words": len(set(w.lower() for w in words)),
            "vocabulary_richness": len(set(w.lower() for w in words)) / max(len(words), 1),
        }

    @classmethod
    def _vocab_analysis(cls, text: str) -> dict:
        """Vocabulary and n-gram analysis"""
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)

        # Bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        bigram_freq = Counter(bigrams)

        return {
            "top_words": word_freq.most_common(20),
            "top_bigrams": bigram_freq.most_common(10),
            "hapax_legomena": sum(1 for w, c in word_freq.items() if c == 1),
            "repeated_words": {w: c for w, c in word_freq.most_common(5) if c > 2},
        }

    @classmethod
    def _complexity_analysis(cls, text: str, profile: DataProfile) -> dict:
        """Complexity metrics"""
        lines = [l for l in text.split('\n') if l.strip()]
        words = text.split()

        # Sentence-level (approximate)
        sentences = re.split(r'[.!?。！？]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sent_lengths = [len(s.split()) for s in sentences]

        metrics = {
            "avg_sentence_length": sum(sent_lengths) / max(len(sent_lengths), 1),
            "max_sentence_length": max(sent_lengths) if sent_lengths else 0,
            "avg_line_length_chars": sum(len(l) for l in lines) / max(len(lines), 1),
            "unique_char_count": len(set(text)),
            "special_char_ratio": sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1),
            "digit_ratio": sum(1 for c in text if c.isdigit()) / max(len(text), 1),
        }

        # Code-specific
        if profile.data_type == DataType.CODE:
            metrics["nesting_depth"] = cls._estimate_nesting(text)
            metrics["comment_ratio"] = cls._comment_ratio(text)

        return metrics

    @classmethod
    def _code_structure(cls, text: str) -> dict:
        """Analyze code structure"""
        functions = re.findall(r'(?:def|function|fn|func)\s+(\w+)', text)
        classes = re.findall(r'(?:class|struct|trait|interface|impl)\s+(\w+)', text)
        imports = re.findall(r'(?:import|from|use|require)\s+(\S+)', text)

        return {
            "functions": functions[:20],
            "classes": classes[:10],
            "imports": imports[:20],
            "function_count": len(functions),
            "class_count": len(classes),
            "import_count": len(imports),
            "has_tests": bool(re.search(r'(test_|def test|it\(|describe\(|#\[test)', text)),
            "has_docstrings": bool(re.search(r'("""|\'\'\'|/\*\*)', text)),
            "has_type_hints": bool(re.search(r'(:\s*(str|int|float|bool|list|dict|Optional))|->\s*\w+', text)),
        }

    @classmethod
    def _structured_analysis(cls, text: str) -> dict:
        """Analyze structured data (CSV/JSON)"""
        stripped = text.strip()
        result = {}

        if stripped.startswith('['):
            try:
                data = json.loads(stripped)
                if isinstance(data, list) and data:
                    result["format"] = "json_array"
                    result["item_count"] = len(data)
                    if isinstance(data[0], dict):
                        result["fields"] = list(data[0].keys())
                        result["field_count"] = len(data[0])
                        # Field type inference
                        result["field_types"] = {}
                        for field in list(data[0].keys())[:20]:
                            types = [type(row.get(field)).__name__ for row in data[:20] if field in row]
                            result["field_types"][field] = Counter(types).most_common(1)[0][0] if types else "unknown"
            except json.JSONDecodeError:
                result["format"] = "invalid_json"

        return result

    @classmethod
    def _text_structure(cls, text: str) -> dict:
        """Analyze text structure"""
        headers = re.findall(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
        lists = re.findall(r'^\s*[-*+]\s+', text, re.MULTILINE)
        links = re.findall(r'\[.+?\]\(.+?\)', text)

        return {
            "has_headers": len(headers) > 0,
            "header_count": len(headers),
            "headers": headers[:10],
            "list_items": len(lists),
            "links": len(links),
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
        }

    @classmethod
    def _estimate_nesting(cls, text: str) -> int:
        """Estimate max nesting depth"""
        depth = 0
        max_depth = 0
        for char in text:
            if char in '{([':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char in '})]':
                depth = max(0, depth - 1)
        return max_depth

    @classmethod
    def _comment_ratio(cls, text: str) -> float:
        """Estimate comment ratio for code"""
        lines = text.split('\n')
        comment_lines = sum(1 for l in lines if re.match(r'^\s*(#|//|\*|/\*)', l))
        return comment_lines / max(len(lines), 1)
