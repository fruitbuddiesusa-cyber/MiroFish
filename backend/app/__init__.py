"""
Synthetic Data Engine — Flask Application Factory
"""

import os
import logging
from flask import Flask, request
from flask_cors import CORS

from .config.settings import Settings
from .api.routes import api_bp


def create_app():
    """Flask application factory"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = Settings.SECRET_KEY
    app.config['JSON_AS_ASCII'] = False

    # JSON encoding for Chinese
    if hasattr(app, 'json') and hasattr(app.json, 'ensure_ascii'):
        app.json.ensure_ascii = False

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if Settings.DEBUG else logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    )
    logger = logging.getLogger('synthetic_engine')

    is_reloader = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    should_log = not Settings.DEBUG or is_reloader

    if should_log:
        logger.info("=" * 50)
        logger.info("Synthetic Data Engine starting...")
        logger.info("=" * 50)

    # CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Request logging
    @app.before_request
    def log_request():
        if Settings.DEBUG:
            logger.debug(f"{request.method} {request.path}")

    # Register API blueprint
    app.register_blueprint(api_bp, url_prefix='/api')

    # Root health check
    @app.route('/')
    def index():
        return {
            "service": "Synthetic Data Engine",
            "version": "0.1.0",
            "status": "running",
            "endpoints": {
                "health": "GET /api/health",
                "analyze": "POST /api/analyze",
                "generate": "POST /api/generate",
                "generate_async": "POST /api/generate-async",
                "status": "GET /api/status/:job_id",
                "validate": "POST /api/validate",
                "export": "POST /api/export",
                "upload": "POST /api/upload",
                "cost": "GET /api/cost",
            },
        }

    if should_log:
        logger.info("Synthetic Data Engine ready")

    return app
