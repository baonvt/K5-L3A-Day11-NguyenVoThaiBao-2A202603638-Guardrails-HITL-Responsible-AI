"""Local HTTP service for the Blue VinBank assistant.

Run from the repository root with:
    python -m uvicorn server:app --app-dir src --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from assignment.pipeline import build_observability, build_production_plugins
from assignment.rate_limiter import RateLimitPlugin
from core.config import (
    blue_provider_label,
    get_blue_provider,
    PROVIDER_LIQUID,
    get_openrouter_api_key,
)
from core.utils import chat_with_agent
from agents.agent import create_blue_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    if get_blue_provider() == PROVIDER_LIQUID:
        import os
        has_key = bool(os.environ.get("LIQUID_API_KEY", "").strip())
    else:
        has_key = bool(get_openrouter_api_key())
    if not has_key:
        raise RuntimeError("Blue provider API key is missing from the process environment.")
    plugins = build_production_plugins(use_llm_judge=False)
    audit, monitor = build_observability()
    agent, runner = create_blue_agent(plugins)
    app.state.plugins = plugins
    app.state.agent = agent
    app.state.runner = runner
    app.state.audit = audit
    app.state.monitor = monitor
    yield


app = FastAPI(
    title="VinBank Guardrails Lab",
    description="Blue assistant API with input/output guardrails, rate limiting, and audit logging.",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    user_id: str = Field(default="local-user", min_length=1, max_length=128)


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "Blue", "provider": blue_provider_label()}


@app.post("/chat")
async def chat(request: ChatRequest):
    plugins = app.state.plugins
    runner = app.state.runner
    limiter = next(p for p in plugins if isinstance(p, RateLimitPlugin))
    input_guard = next(p for p in plugins if p.name == "input_guardrail")
    output_guard = next(p for p in plugins if p.name == "output_guardrail")
    request_id = str(uuid.uuid4())
    app.state.audit.record_input(
        user_id=request.user_id, text=request.message, request_id=request_id
    )
    before = (limiter.blocked_count, input_guard.blocked_count, output_guard.redacted_count)
    app.state.monitor.total_requests += 1
    runner.user_id = request.user_id
    try:
        response, _ = await chat_with_agent(app.state.agent, runner, request.message)
        response = response or ""
    except Exception as exc:
        app.state.monitor.blocked_requests += 1
        app.state.audit.record_output(
            user_id=request.user_id,
            text="Request blocked because the model service is unavailable.",
            blocked=True,
            layer="model_error",
            request_id=request_id,
        )
        app.state.audit.export_json()
        app.state.monitor.export_json()
        raise HTTPException(
            status_code=503,
            detail="The assistant service is temporarily unavailable; the request was blocked.",
        ) from exc

    rate_block = limiter.blocked_count > before[0]
    input_block = input_guard.blocked_count > before[1]
    redacted = output_guard.redacted_count > before[2]
    blocked = rate_block or input_block
    layer = (
        "rate_limit" if rate_block else
        "input_guardrail" if input_block else
        "output_redaction" if redacted else None
    )
    if blocked:
        app.state.monitor.blocked_requests += 1
    if rate_block:
        app.state.monitor.rate_limit_hits += 1
    app.state.audit.record_output(
        user_id=request.user_id,
        text=response,
        blocked=blocked,
        layer=layer,
        request_id=request_id,
    )
    app.state.audit.export_json()
    app.state.monitor.export_json()
    return {
        "request_id": request_id,
        "agent": "Blue",
        "response": response,
        "blocked": blocked,
        "redacted": redacted,
        "layer": layer,
    }
