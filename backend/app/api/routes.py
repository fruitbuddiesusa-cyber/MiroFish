"""
API Routes — REST endpoints for the Synthetic Data Engine
"""

import json
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, Response

from .schemas import (
    GenerateRequest, AnalyzeRequest, ValidateRequest,
    ExportRequest, ApiResponse,
)
from ..agents.orchestrator import Orchestrator
from ..engine.analyzer import Analyzer
from ..engine.hallucination_guard import HallucinationGuard
from ..engine.export_manager import ExportManager
from ..utils.cost_tracker import CostTracker
from ..utils.file_parser import FileParser

# Blueprint
api_bp = Blueprint('api', __name__)

# In-memory job storage (for async jobs)
_jobs: dict[str, dict] = {}


@api_bp.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({"status": "ok", "service": "Synthetic Data Engine"})


@api_bp.route('/analyze', methods=['POST'])
def analyze():
    """Analyze input data — detect type, extract patterns, find gaps"""
    data = request.get_json(silent=True) or {}
    req = AnalyzeRequest.from_dict(data)

    if not req.text:
        return jsonify(ApiResponse(success=False, error="No text provided").to_dict()), 400

    try:
        analysis = Analyzer.analyze(req.text, req.filename)
        return jsonify(ApiResponse(
            success=True,
            data=analysis.to_dict(),
        ).to_dict())
    except Exception as e:
        return jsonify(ApiResponse(success=False, error=str(e)).to_dict()), 500


@api_bp.route('/generate', methods=['POST'])
def generate():
    """
    Run the full synthetic data generation pipeline.
    Synchronous — waits for completion.
    For async, use /generate-async + /status/:job_id
    """
    data = request.get_json(silent=True) or {}
    req = GenerateRequest.from_dict(data)

    if not req.text:
        return jsonify(ApiResponse(success=False, error="No text provided").to_dict()), 400

    try:
        orchestrator = Orchestrator()
        result = orchestrator.run_pipeline(
            input_text=req.text,
            filename=req.filename,
            target_count=req.target_count,
            quality_threshold=req.quality_threshold,
            novelty_threshold=req.novelty_threshold,
            include_edge_cases=req.include_edge_cases,
            edge_case_ratio=req.edge_case_ratio,
        )

        # Export if format specified
        if result.get("success") and req.export_format != "none":
            export_data = ExportManager.export(
                samples=result["samples"],
                format=req.export_format,
                metadata={
                    "data_type": result.get("data_type"),
                    "total_generated": result.get("total_generated"),
                },
            )
            result["export"] = export_data[:1000] if isinstance(export_data, str) else "[binary]"

        return jsonify(ApiResponse(success=True, data=result).to_dict())

    except Exception as e:
        return jsonify(ApiResponse(success=False, error=str(e)).to_dict()), 500


@api_bp.route('/generate-async', methods=['POST'])
def generate_async():
    """
    Start async generation. Returns job_id immediately.
    Poll /status/:job_id for progress, or use WebSocket.
    """
    import threading

    data = request.get_json(silent=True) or {}
    job_id = f"job_{uuid.uuid4().hex[:12]}"

    _jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued",
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None,
    }

    def run_job():
        _jobs[job_id]["status"] = "running"

        def progress_cb(stage, progress, message):
            _jobs[job_id]["progress"] = progress
            _jobs[job_id]["message"] = message
            _jobs[job_id]["stage"] = stage

        try:
            req = GenerateRequest.from_dict(data)
            orchestrator = Orchestrator()
            result = orchestrator.run_pipeline(
                input_text=req.text,
                filename=req.filename,
                target_count=req.target_count,
                quality_threshold=req.quality_threshold,
                novelty_threshold=req.novelty_threshold,
                include_edge_cases=req.include_edge_cases,
                edge_case_ratio=req.edge_case_ratio,
                progress_callback=progress_cb,
            )
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result
            _jobs[job_id]["progress"] = 1.0
        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)

    thread = threading.Thread(target=run_job)
    thread.daemon = True
    thread.start()

    return jsonify(ApiResponse(
        success=True,
        data={"job_id": job_id, "status": "queued"},
    ).to_dict()), 202


@api_bp.route('/status/<job_id>', methods=['GET'])
def job_status(job_id: str):
    """Get job status and progress"""
    job = _jobs.get(job_id)
    if not job:
        return jsonify(ApiResponse(success=False, error="Job not found").to_dict()), 404

    return jsonify(ApiResponse(success=True, data=job).to_dict())


@api_bp.route('/validate', methods=['POST'])
def validate():
    """Validate synthetic data samples"""
    data = request.get_json(silent=True) or {}
    req = ValidateRequest.from_dict(data)

    if not req.samples:
        return jsonify(ApiResponse(success=False, error="No samples provided").to_dict()), 400

    try:
        result = HallucinationGuard.check_batch(
            req.samples,
            data_type=req.data_type,
        )
        return jsonify(ApiResponse(success=True, data=result).to_dict())
    except Exception as e:
        return jsonify(ApiResponse(success=False, error=str(e)).to_dict()), 500


@api_bp.route('/export', methods=['POST'])
def export_data():
    """Export samples in specified format"""
    data = request.get_json(silent=True) or {}
    req = ExportRequest.from_dict(data)

    if not req.samples:
        return jsonify(ApiResponse(success=False, error="No samples provided").to_dict()), 400

    try:
        exported = ExportManager.export(
            samples=req.samples,
            format=req.format,
            metadata=req.metadata,
        )
        size_info = ExportManager.estimate_size(req.samples, req.format)

        return jsonify(ApiResponse(
            success=True,
            data={
                "content": exported if req.format != "parquet" else "[binary]",
                "format": req.format,
                "size": size_info,
            },
        ).to_dict())
    except Exception as e:
        return jsonify(ApiResponse(success=False, error=str(e)).to_dict()), 500


@api_bp.route('/cost', methods=['GET'])
def cost_summary():
    """Get API cost summary"""
    return jsonify(ApiResponse(
        success=True,
        data=CostTracker.get_summary(),
    ).to_dict())


@api_bp.route('/upload', methods=['POST'])
def upload_file():
    """Upload a file for analysis/generation"""
    if 'file' not in request.files:
        return jsonify(ApiResponse(success=False, error="No file provided").to_dict()), 400

    file = request.files['file']
    if not file.filename:
        return jsonify(ApiResponse(success=False, error="No filename").to_dict()), 400

    try:
        # Save to temp
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            file.save(tmp.name)
            text = FileParser.extract(tmp.name)
            os.unlink(tmp.name)

        return jsonify(ApiResponse(
            success=True,
            data={
                "filename": file.filename,
                "text_length": len(text),
                "text": text[:50000],  # Cap at 50k chars for response
            },
        ).to_dict())
    except Exception as e:
        return jsonify(ApiResponse(success=False, error=str(e)).to_dict()), 500
