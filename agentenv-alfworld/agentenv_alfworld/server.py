"""FastAPI application for the ALFWorld TextWorld service."""

from __future__ import annotations

import importlib.metadata
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from .model import ResetRequestBody, SessionRequestBody, StepRequestBody
from .service import ALFWorldService, ServiceError, official_environment_factory

logger = logging.getLogger(__name__)


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set explicitly")
    return Path(value).resolve()


def create_app() -> FastAPI:
    data_root = _required_path("ALFWORLD_DATA")
    manifest_path = _required_path("ALFWORLD_TASK_MANIFEST")
    default_config = Path(__file__).resolve().parent / "configs" / "base_config.yaml"
    config_path = Path(os.environ.get("ALFWORLD_CONFIG", str(default_config))).resolve()
    if not config_path.is_file():
        raise RuntimeError(f"ALFWorld config is missing: {config_path}")
    version = importlib.metadata.version("alfworld")
    if version != "0.5.0":
        raise RuntimeError(f"expected alfworld==0.5.0, got {version}")
    service = ALFWorldService(
        data_root,
        manifest_path,
        official_environment_factory(data_root, config_path),
        alfworld_version=version,
        verify_hashes=os.environ.get("ALFWORLD_VERIFY_HASHES", "0") == "1",
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        service.close_all()

    application = FastAPI(title="AgentGym ALFWorld TextWorld", version="1", lifespan=lifespan)
    application.state.service = service

    @application.exception_handler(ServiceError)
    async def service_error(_: Request, error: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code, "message": error.message}},
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "request validation failed",
                    "details": [
                        {"location": list(item["loc"]), "message": item["msg"], "type": item["type"]}
                        for item in error.errors()
                    ],
                }
            },
        )

    @application.exception_handler(Exception)
    async def internal_error(_: Request, error: Exception) -> JSONResponse:
        logger.error("Unhandled ALFWorld service error", exc_info=(type(error), error, error.__traceback__))
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "internal service error"}},
        )

    @application.get("/health")
    def health() -> dict[str, Any]:
        return service.health()

    @application.post("/create")
    def create() -> dict[str, str]:
        return service.create()

    @application.post("/reset")
    def reset(body: ResetRequestBody) -> dict[str, Any]:
        return service.reset(body.session_id, body.task_id, body.seed)

    @application.get("/observation")
    def observation(session_id: str) -> dict[str, Any]:
        return service.observation(session_id)

    @application.get("/available_actions")
    def available_actions(session_id: str) -> dict[str, list[str]]:
        return service.available_actions(session_id)

    @application.post("/step")
    def step(body: StepRequestBody) -> dict[str, Any]:
        return service.step(body.session_id, body.action)

    @application.post("/close")
    def close(body: SessionRequestBody) -> dict[str, bool]:
        return service.close(body.session_id)

    return application


app = create_app()
