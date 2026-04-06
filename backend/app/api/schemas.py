"""
API Schemas — request/response models for the REST API
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerateRequest:
    """POST /api/generate request"""
    text: str = ""
    filename: str | None = None
    target_count: int = 50
    quality_threshold: float = 0.7
    novelty_threshold: float = 0.6
    include_edge_cases: bool = True
    edge_case_ratio: float = 0.2
    export_format: str = "json"
    use_llm_validation: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "GenerateRequest":
        return cls(
            text=data.get("text", ""),
            filename=data.get("filename"),
            target_count=int(data.get("target_count", 50)),
            quality_threshold=float(data.get("quality_threshold", 0.7)),
            novelty_threshold=float(data.get("novelty_threshold", 0.6)),
            include_edge_cases=bool(data.get("include_edge_cases", True)),
            edge_case_ratio=float(data.get("edge_case_ratio", 0.2)),
            export_format=data.get("export_format", "json"),
            use_llm_validation=bool(data.get("use_llm_validation", True)),
        )


@dataclass
class AnalyzeRequest:
    """POST /api/analyze request"""
    text: str = ""
    filename: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "AnalyzeRequest":
        return cls(
            text=data.get("text", ""),
            filename=data.get("filename"),
        )


@dataclass
class ValidateRequest:
    """POST /api/validate request"""
    samples: list = field(default_factory=list)
    source_text: str = ""
    data_type: str = "text"
    quality_threshold: float = 0.7

    @classmethod
    def from_dict(cls, data: dict) -> "ValidateRequest":
        return cls(
            samples=data.get("samples", []),
            source_text=data.get("source_text", ""),
            data_type=data.get("data_type", "text"),
            quality_threshold=float(data.get("quality_threshold", 0.7)),
        )


@dataclass
class ExportRequest:
    """POST /api/export request"""
    samples: list = field(default_factory=list)
    format: str = "json"
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "ExportRequest":
        return cls(
            samples=data.get("samples", []),
            format=data.get("format", "json"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ApiResponse:
    """Standard API response"""
    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        resp = {"success": self.success}
        if self.data is not None:
            resp["data"] = self.data
        if self.error:
            resp["error"] = self.error
        if self.metadata:
            resp["metadata"] = self.metadata
        return resp
