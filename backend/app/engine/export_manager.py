"""
Export Manager — formats and exports synthetic data in multiple formats
Supports JSON, CSV, JSONL, and Parquet
"""

import json
import csv
import io
import os
from typing import Any
from datetime import datetime


class ExportManager:
    """
    Export synthetic data in various formats.
    """

    @classmethod
    def export(
        cls,
        samples: list[Any],
        format: str = "json",
        filepath: str | None = None,
        metadata: dict | None = None,
    ) -> str | bytes:
        """
        Export samples in the specified format.

        Args:
            samples: List of data samples
            format: Output format (json, csv, jsonl, parquet)
            filepath: Optional file path to write to
            metadata: Optional metadata to include

        Returns:
            Formatted string or bytes
        """
        format = format.lower()

        if format == "json":
            return cls._export_json(samples, filepath, metadata)
        elif format == "csv":
            return cls._export_csv(samples, filepath)
        elif format == "jsonl":
            return cls._export_jsonl(samples, filepath)
        elif format == "parquet":
            return cls._export_parquet(samples, filepath, metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def _export_json(
        cls,
        samples: list[Any],
        filepath: str | None,
        metadata: dict | None,
    ) -> str:
        """Export as JSON"""
        output = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "total_samples": len(samples),
            "metadata": metadata or {},
            "samples": samples,
        }
        text = json.dumps(output, ensure_ascii=False, indent=2)

        if filepath:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)

        return text

    @classmethod
    def _export_csv(cls, samples: list[Any], filepath: str | None) -> str:
        """Export as CSV — samples must be dicts or strings"""
        output = io.StringIO()

        if not samples:
            return ""

        # Normalize samples to dicts
        if isinstance(samples[0], dict):
            fieldnames = list(samples[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for sample in samples:
                if isinstance(sample, dict):
                    # Flatten nested structures
                    row = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                           for k, v in sample.items()}
                    writer.writerow(row)
        else:
            writer = csv.writer(output)
            writer.writerow(["text"])
            for sample in samples:
                writer.writerow([str(sample)])

        text = output.getvalue()

        if filepath:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                f.write(text)

        return text

    @classmethod
    def _export_jsonl(cls, samples: list[Any], filepath: str | None) -> str:
        """Export as JSONL (one JSON per line)"""
        lines = []
        for sample in samples:
            if isinstance(sample, str):
                lines.append(json.dumps({"text": sample}, ensure_ascii=False))
            else:
                lines.append(json.dumps(sample, ensure_ascii=False))

        text = '\n'.join(lines)

        if filepath:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)

        return text

    @classmethod
    def _export_parquet(
        cls,
        samples: list[Any],
        filepath: str | None,
        metadata: dict | None,
    ) -> bytes:
        """Export as Parquet — requires pyarrow"""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for Parquet export: pip install pyarrow")

        # Normalize to dicts
        if samples and isinstance(samples[0], dict):
            data = samples
        else:
            data = [{"text": str(s)} for s in samples]

        table = pa.Table.from_pylist(data)

        if metadata:
            table = table.replace_schema_metadata({
                "generated_at": datetime.now().isoformat(),
                **{k: str(v) for k, v in metadata.items()},
            })

        if filepath:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            pq.write_table(table, filepath)

        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        return buffer.getvalue()

    @classmethod
    def get_supported_formats(cls) -> list[str]:
        """Get list of supported export formats"""
        return ["json", "csv", "jsonl", "parquet"]

    @classmethod
    def estimate_size(cls, samples: list[Any], format: str = "json") -> dict:
        """Estimate export file size"""
        if format == "json":
            content = cls._export_json(samples, None, None)
        elif format == "csv":
            content = cls._export_csv(samples, None)
        elif format == "jsonl":
            content = cls._export_jsonl(samples, None)
        else:
            content = str(samples)

        size_bytes = len(content.encode('utf-8'))
        return {
            "format": format,
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 1),
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "sample_count": len(samples),
        }
