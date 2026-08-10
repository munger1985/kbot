"""Model Serving OCR 独立进程。"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi_offline import FastAPIOffline
from loguru import logger

from model_serving.common.bootstrap import create_model_registry
from model_serving.common.management_router import create_model_management_router
from model_serving.config import get_model_serving_settings
from model_serving.ocr import OCRRequest, OCRResponse, OCRService
from model_serving.persistence import create_model_serving_uow_factory
from platform_core.contracts import INTERNAL_API_V1
from platform_core.database.oracle import create_database_runtime
from platform_core.dictionary import ModelCategory
from platform_core.platform.port_check import check_port_available
from platform_core.security import create_internal_auth_middleware

settings = get_model_serving_settings()
config = settings.ocr


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = create_database_runtime()
    factory = create_model_serving_uow_factory(runtime.session_factory)
    service = OCRService(uow_factory=factory)
    app.state.service_name = config.service_name
    app.state.ocr_service = service
    app.state.model_registry = create_model_registry(
        session_factory=runtime.session_factory,
        runtime_service=service,
        service_name=config.service_name,
        settings=settings,
    )
    yield
    await runtime.close()


app: FastAPI = FastAPIOffline(
    title="KBot OCR API",
    version=config.service_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.platform.debug else None,
    redoc_url="/redoc" if settings.platform.debug else None,
)
app.middleware("http")(
    create_internal_auth_middleware(audience=config.service_name)
)
app.include_router(
    create_model_management_router(category=ModelCategory.OCR.value)
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(f"{INTERNAL_API_V1}/inference", response_model=OCRResponse)
async def inference(body: OCRRequest) -> dict:
    try:
        return await app.state.ocr_service.infer(
            model_id=body.model_id,
            image_base64=body.image_base64,
        )
    except (ValueError, LookupError) as exc:
        raise HTTPException(
            422,
            detail={"code": "OCR_MODEL_INVALID", "message": str(exc)},
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            503,
            detail={
                "code": "OCR_PROVIDER_UNAVAILABLE",
                "message": str(exc),
            },
        ) from exc


def main() -> None:
    """启动 OCR 独立进程。"""
    if not check_port_available(
        config.service_host, config.service_port, config.service_name
    ):
        raise SystemExit(1)
    logger.info(
        "正在启动 OCR 服务 | 监听地址：{}:{}",
        config.service_host,
        config.service_port,
    )
    uvicorn.run(
        app,
        host=config.service_host,
        port=config.service_port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
