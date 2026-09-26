"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._pending: dict[str, dict] = {}

    @staticmethod
    def _safe_text(text: str) -> str:
        """Redact known PII/secrets before placing content in an audit file."""
        try:
            from guardrails.output_guardrails import content_filter

            return content_filter(str(text or ""))["redacted"]
        except Exception:
            # Logging must not break the request path. Avoid returning an
            # exception or repr that could accidentally contain user data.
            return "[UNAVAILABLE]"

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store a redacted input and start timestamp for an interaction."""
        key = request_id or f"{user_id}:{uuid.uuid4().hex}"
        self._open[key] = time.perf_counter()
        self._pending[key] = {
            "request_id": key,
            "user_id": user_id,
            "input": self._safe_text(text),
            "started_at": utc_now_iso(),
        }
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store a redacted output, decision layer and measured latency."""
        key = request_id or f"{user_id}:untracked"
        started = self._open.pop(key, time.perf_counter())
        pending = self._pending.pop(key, {})
        finished_at = utc_now_iso()
        entry = {
            "request_id": key,
            "user_id": user_id,
            "input": pending.get("input", ""),
            "text": self._safe_text(text),
            "output": self._safe_text(text),
            "blocked": bool(blocked),
            "layer": layer or "none",
            "started_at": pending.get("started_at", finished_at),
            "finished_at": finished_at,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
