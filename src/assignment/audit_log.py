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

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Start a request record while redacting PII and protected values."""
        key = request_id or str(uuid.uuid4())
        try:
            from guardrails.output_guardrails import content_filter
            safe_text = content_filter(text)["redacted"]
        except Exception:
            safe_text = "[REDACTED]"
        self._open[key] = time.monotonic()
        self._pending = getattr(self, "_pending", {})
        self._pending[key] = {
            "request_id": key,
            "user_id": user_id,
            "input": safe_text[:2000],
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
        """Finish and append a request record with decision and latency."""
        key = request_id or user_id
        pending = getattr(self, "_pending", {}).pop(key, None)
        started = self._open.pop(key, None)
        try:
            from guardrails.output_guardrails import content_filter
            safe_text = content_filter(text)["redacted"]
        except Exception:
            safe_text = "[REDACTED]"
        row = pending or {
            "request_id": key,
            "user_id": user_id,
            "started_at": utc_now_iso(),
        }
        row.update({
            "finished_at": utc_now_iso(),
            "latency_ms": round((time.monotonic() - started) * 1000, 2) if started is not None else None,
            "output": safe_text[:2000],
            "blocked": bool(blocked),
            "layer": layer,
        })
        self.logs.append(row)
        return row

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
