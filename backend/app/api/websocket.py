"""
WebSocket handler — real-time progress streaming for generation jobs
"""

import json
from typing import Any


class WebSocketHandler:
    """
    WebSocket event handler for real-time generation progress.
    Integrates with the orchestrator's progress_callback.

    Note: This is a lightweight handler designed for use with
    flask-sock or simple-websocket. For production, consider
    using Socket.IO or a dedicated WS server.
    """

    # Connected clients (session_id -> send_function)
    _clients: dict[str, Any] = {}

    @classmethod
    def register(cls, session_id: str, send_fn):
        """Register a WebSocket client"""
        cls._clients[session_id] = send_fn

    @classmethod
    def unregister(cls, session_id: str):
        """Unregister a WebSocket client"""
        cls._clients.pop(session_id, None)

    @classmethod
    def broadcast(cls, event: dict):
        """Send event to all connected clients"""
        dead = []
        for sid, send in cls._clients.items():
            try:
                send(json.dumps(event, ensure_ascii=False))
            except Exception:
                dead.append(sid)
        for sid in dead:
            cls._clients.pop(sid, None)

    @classmethod
    def send_to(cls, session_id: str, event: dict):
        """Send event to specific client"""
        send = cls._clients.get(session_id)
        if send:
            try:
                send(json.dumps(event, ensure_ascii=False))
            except Exception:
                cls._clients.pop(session_id, None)

    @classmethod
    def create_progress_callback(cls, session_id: str | None = None):
        """
        Create a progress callback function for the orchestrator.
        If session_id is set, sends to that client only.
        Otherwise broadcasts to all.
        """
        def callback(stage: str, progress: float, message: str):
            event = {
                "type": "progress",
                "stage": stage,
                "progress": round(progress, 3),
                "message": message,
            }
            if session_id:
                cls.send_to(session_id, event)
            else:
                cls.broadcast(event)

        return callback

    @classmethod
    def client_count(cls) -> int:
        """Number of connected clients"""
        return len(cls._clients)
