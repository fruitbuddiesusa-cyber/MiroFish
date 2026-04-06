"""
Data Detector — auto-detect input data type and characteristics
"""

import json
import re
import csv
import io
from typing import Any
from dataclasses import dataclass, field
from enum import Enum


class DataType(str, Enum):
    """Detected data types"""
    CODE = "code"
    TEXT = "text"
    STRUCTURED = "structured"      # CSV, JSON arrays
    MARKDOWN = "markdown"
    MIXED = "mixed"
    CONVERSATION = "conversation"  # Chat/dialogue
    INSTRUCTION = "instruction"    # Instruction-following data
    QA = "qa"                      # Question-answer pairs
    REASONING = "reasoning"        # Math/logic reasoning chains


class CodeLanguage(str, Enum):
    """Detected programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    CSHARP = "csharp"
    SQL = "sql"
    UNKNOWN = "unknown"


@dataclass
class DataProfile:
    """Analysis profile of input data"""
    data_type: DataType
    language: CodeLanguage | None = None
    row_count: int = 0
    char_count: int = 0
    has_structure: bool = False
    columns: list[str] = field(default_factory=list)
    sample_values: dict[str, list] = field(default_factory=dict)
    quality_signals: dict[str, Any] = field(default_factory=dict)
    suggested_chunk_size: int = 1500
    metadata: dict[str, Any] = field(default_factory=dict)


# File extension to language mapping
_EXT_MAP = {
    ".py": CodeLanguage.PYTHON,
    ".js": CodeLanguage.JAVASCRIPT,
    ".ts": CodeLanguage.TYPESCRIPT,
    ".tsx": CodeLanguage.TYPESCRIPT,
    ".java": CodeLanguage.JAVA,
    ".go": CodeLanguage.GO,
    ".rs": CodeLanguage.RUST,
    ".cpp": CodeLanguage.CPP,
    ".c": CodeLanguage.CPP,
    ".cs": CodeLanguage.CSHARP,
    ".sql": CodeLanguage.SQL,
}

# Language detection patterns
_CODE_PATTERNS = {
    CodeLanguage.PYTHON: [
        r'^\s*(def |class |import |from |if __name__|print\()',
        r'^\s*#.*coding[:=]',
        r':\s*$',
    ],
    CodeLanguage.JAVASCRIPT: [
        r'^\s*(const |let |var |function |export |import )',
        r'=>\s*{',
        r'\bconsole\.(log|error|warn)\(',
    ],
    CodeLanguage.TYPESCRIPT: [
        r'^\s*(interface |type |enum |declare )',
        r':\s*(string|number|boolean|any)\b',
    ],
    CodeLanguage.GO: [
        r'^\s*package\s+\w+',
        r'^\s*func\s+\w+\(',
        r':=',
    ],
    CodeLanguage.RUST: [
        r'^\s*fn\s+\w+\(',
        r'^\s*(let|mut|impl|trait|struct|enum)\s+',
        r'->.*{',
    ],
}


class DataDetector:
    """Auto-detect data type and characteristics"""

    @classmethod
    def detect(cls, text: str, filename: str | None = None) -> DataProfile:
        """
        Analyze text and return a DataProfile.
        Optionally use filename for extension-based hints.
        """
        char_count = len(text)
        lines = text.strip().split('\n')
        row_count = len(lines)

        # Try extension-based detection first
        ext_hint = None
        if filename:
            for ext, lang in _EXT_MAP.items():
                if filename.endswith(ext):
                    ext_hint = lang
                    break

        # Detect data type
        if ext_hint:
            data_type = DataType.CODE
            language = ext_hint
        elif cls._is_code(text):
            data_type = DataType.CODE
            language = cls._detect_language(text)
        elif cls._is_structured(text):
            data_type = DataType.STRUCTURED
            language = None
        elif cls._is_markdown(text):
            data_type = DataType.MARKDOWN
            language = None
        elif cls._is_conversation(text):
            data_type = DataType.CONVERSATION
            language = None
        elif cls._is_qa(text):
            data_type = DataType.QA
            language = None
        elif cls._is_reasoning(text):
            data_type = DataType.REASONING
            language = None
        else:
            data_type = DataType.TEXT
            language = None

        # Build profile
        profile = DataProfile(
            data_type=data_type,
            language=language,
            row_count=row_count,
            char_count=char_count,
        )

        # Structured data analysis
        if data_type == DataType.STRUCTURED:
            profile = cls._analyze_structured(text, profile)

        # Quality signals
        profile.quality_signals = cls._assess_quality(text, data_type)

        # Suggested chunk size
        profile.suggested_chunk_size = cls._suggest_chunk_size(data_type, char_count)

        return profile

    @classmethod
    def _is_code(cls, text: str) -> bool:
        """Check if text is code"""
        indicators = 0
        lines = text.strip().split('\n')[:50]

        code_markers = [
            r'^\s*(def |class |function |import |from |const |let |var |package |fn |func )',
            r'[{};]\s*$',
            r'^\s*(if |for |while |return |switch )',
            r'//.*$', r'#.*$', r'/\*.*\*/',
            r'=\s*\[', r'=\s*{',
        ]

        for line in lines:
            if not line.strip():
                continue
            for pattern in code_markers:
                if re.search(pattern, line):
                    indicators += 1
                    break

        # If >30% of non-empty lines have code markers
        non_empty = sum(1 for l in lines if l.strip())
        return non_empty > 0 and (indicators / non_empty) > 0.3

    @classmethod
    def _detect_language(cls, text: str) -> CodeLanguage:
        """Detect programming language"""
        scores: dict[CodeLanguage, int] = {lang: 0 for lang in CodeLanguage}

        for lang, patterns in _CODE_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.MULTILINE)
                scores[lang] += len(matches)

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else CodeLanguage.UNKNOWN

    @classmethod
    def _is_structured(cls, text: str) -> bool:
        """Check for CSV/JSON structure"""
        stripped = text.strip()
        # JSON
        if (stripped.startswith('[') and stripped.endswith(']')) or \
           (stripped.startswith('{') and stripped.endswith('}')):
            try:
                json.loads(stripped)
                return True
            except json.JSONDecodeError:
                pass
        # CSV — check consistent column counts
        lines = stripped.split('\n')[:20]
        if len(lines) >= 3:
            try:
                reader = csv.reader(io.StringIO(stripped))
                rows = [row for i, row in enumerate(reader) if i < 20]
                if len(rows) >= 3:
                    col_counts = [len(r) for r in rows]
                    if len(set(col_counts)) == 1 and col_counts[0] > 1:
                        return True
            except csv.Error:
                pass
        return False

    @classmethod
    def _is_markdown(cls, text: str) -> bool:
        """Check for markdown"""
        md_indicators = [
            r'^#{1,6}\s+',        # Headers
            r'^\s*[-*+]\s+',      # Lists
            r'^\s*\d+\.\s+',      # Numbered lists
            r'\[.*\]\(.*\)',      # Links
            r'^```',               # Code blocks
            r'^\s*\|.*\|.*\|',    # Tables
        ]
        count = 0
        for line in text.split('\n')[:30]:
            for pattern in md_indicators:
                if re.search(pattern, line):
                    count += 1
                    break
        return count >= 3

    @classmethod
    def _is_conversation(cls, text: str) -> bool:
        """Check for chat/dialogue format"""
        patterns = [
            r'^(User|Assistant|Human|AI|Bot|System|Speaker\s*\w*)\s*[:：]',
            r'^\[(User|Assistant|Human|System)\]',
            r'^<(im_start|im_end)>',
        ]
        count = 0
        for line in text.split('\n')[:30]:
            for p in patterns:
                if re.search(p, line, re.IGNORECASE):
                    count += 1
                    break
        return count >= 2

    @classmethod
    def _is_qa(cls, text: str) -> bool:
        """Check for Q&A format"""
        qa_patterns = [
            r'^(Q|Question|问)\s*[:：.]',
            r'^(A|Answer|答)\s*[:：.]',
        ]
        count = 0
        for line in text.split('\n')[:30]:
            for p in qa_patterns:
                if re.search(p, line, re.IGNORECASE):
                    count += 1
                    break
        return count >= 2

    @classmethod
    def _is_reasoning(cls, text: str) -> bool:
        """Check for reasoning chain format"""
        reasoning_markers = [
            r'(step|步骤)\s*\d+',
            r'(therefore|so|thus|hence|所以|因此|于是)',
            r'(first|second|third|finally|首先|其次|最后)',
            r'(because|since|由于|因为)',
            r'(let\'s think|让我们思考|分析如下)',
        ]
        count = 0
        for line in text.split('\n')[:50]:
            for p in reasoning_markers:
                if re.search(p, line, re.IGNORECASE):
                    count += 1
                    break
        return count >= 3

    @classmethod
    def _analyze_structured(cls, text: str, profile: DataProfile) -> DataProfile:
        """Analyze structured data for columns and samples"""
        stripped = text.strip()

        # JSON analysis
        if stripped.startswith('['):
            try:
                data = json.loads(stripped)
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    profile.columns = list(data[0].keys())
                    profile.row_count = len(data)
                    profile.has_structure = True
                    # Sample values
                    for col in profile.columns[:10]:
                        vals = [row.get(col) for row in data[:5] if col in row]
                        profile.sample_values[col] = [str(v)[:100] for v in vals]
            except json.JSONDecodeError:
                pass

        # CSV analysis
        elif ',' in stripped.split('\n')[0]:
            try:
                reader = csv.reader(io.StringIO(stripped))
                rows = list(reader)
                if rows:
                    profile.columns = rows[0]
                    profile.row_count = len(rows) - 1
                    profile.has_structure = True
                    if len(rows) > 1:
                        for i, col in enumerate(profile.columns[:10]):
                            vals = [row[i] for row in rows[1:6] if i < len(row)]
                            profile.sample_values[col] = vals
            except csv.Error:
                pass

        return profile

    @classmethod
    def _assess_quality(cls, text: str, data_type: DataType) -> dict:
        """Quick quality assessment"""
        lines = text.split('\n')
        non_empty = [l for l in lines if l.strip()]

        signals = {
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty),
            "avg_line_length": sum(len(l) for l in non_empty) / max(len(non_empty), 1),
            "empty_ratio": 1 - (len(non_empty) / max(len(lines), 1)),
            "has_unicode": bool(re.search(r'[^\x00-\x7F]', text)),
            "repetition_score": cls._repetition_score(text),
        }

        # Code-specific
        if data_type == DataType.CODE:
            signals["has_comments"] = bool(re.search(r'(#|//|/\*)', text))
            signals["has_docstrings"] = bool(re.search(r'("""|\'\'\')', text))

        return signals

    @classmethod
    def _repetition_score(cls, text: str) -> float:
        """Score 0-1 indicating repetition level"""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) < 10:
            return 0.0
        unique = len(set(lines))
        return 1 - (unique / len(lines))

    @classmethod
    def _suggest_chunk_size(cls, data_type: DataType, char_count: int) -> int:
        """Suggest optimal chunk size based on data type"""
        base_sizes = {
            DataType.CODE: 1000,
            DataType.TEXT: 1500,
            DataType.STRUCTURED: 2000,
            DataType.MARKDOWN: 1500,
            DataType.CONVERSATION: 2000,
            DataType.INSTRUCTION: 1500,
            DataType.QA: 1500,
            DataType.REASONING: 2000,
            DataType.MIXED: 1500,
        }
        return base_sizes.get(data_type, 1500)
