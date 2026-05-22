"""
aaizaql.api.health
──────────────────
T5.1 — GET /health endpoint for load balancer probes.

Mount with FastAPI:
    from aaizaql.api.health import router as health_router
    app.include_router(health_router)
"""

from __future__ import annotations

import time
from typing import Any

try:
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse

    router = APIRouter()

    @router.get("/health")
    async def health(engine: Any = None) -> JSONResponse:
        result = {"status": "ok", "checks": {}}
        if engine is not None:
            result["checks"] = engine.health_check()
        return JSONResponse(content=result)

except ImportError:
    router = None  # type: ignore[assignment]


def run_health_check(engine: Any) -> dict:
    """
    Standalone health check callable — usable without FastAPI.
    engine must expose _connector, _vector_store, _llm attributes.
    """
    t0 = time.monotonic()
    checks: dict[str, Any] = {}

    # DB connectivity
    try:
        engine._connector.test_connection()
        checks["db"] = "ok"
    except Exception as exc:
        checks["db"] = f"error: {exc}"

    # Vector store
    try:
        engine._vector_store.count()
        checks["vector_store"] = "ok"
    except Exception as exc:
        checks["vector_store"] = f"error: {exc}"

    # LLM ping (cheapest possible call)
    try:
        engine._llm.complete("ping", timeout=5)
        checks["llm"] = "ok"
    except Exception as exc:
        checks["llm"] = f"error: {exc}"

    checks["latency_ms"] = int((time.monotonic() - t0) * 1000)
    return checks
