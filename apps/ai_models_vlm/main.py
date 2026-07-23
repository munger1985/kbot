"""VLM Microservice Application.

This module provides a FastAPI-based Vision-Language Model (VLM) service.
It supports multimodal reasoning combining images and text, and is compatible with OpenAI-style streaming (SSE) and non-streaming responses.
"""

import os
import sys
import signal
import uuid
import time
import json
import atexit
import subprocess
from datetime import datetime
from typing import Any, Type, AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi_offline import FastAPIOffline
from loguru import logger
from pydantic import ValidationError
from pydantic_core import core_schema

from platform_core.config.settings import get_vlm_config, get_app_config
from platform_core.dictionary import ModelCategory
from platform_core.logger import LogConfig, LogManager
from platform_core.middleware.log_middleware import log_requests
from platform_core.database.oracle import create_database_runtime
from model_serving.vlm.vlm_service import VLMService
from model_serving.vlm.schema import VLMRequest, VLMResponse, ToggleModelRequest
from platform_core.platform.port_check import check_port_available
from model_serving.common.management_router import create_model_management_router
from model_serving.common.model_registry import ModelRegistryService

# --- Enhanced Pydantic Support for PIL.Image ---

def get_pydantic_core_schema(
    cls: Type[Image.Image],
    handler: Any,
) -> core_schema.CoreSchema:
    """Implement Pydantic Core Schema for PIL.Image.Image.

    Allows Pydantic validation logic to recognize and handle PIL image objects, enhancing type safety for multimodal data.

    Args:
        cls: Target class type.
        handler: Pydantic internal handler.

    Returns:
        Configured CoreSchema object.
    """
    return core_schema.no_info_after_validator_function(
        lambda x: x,
        core_schema.any_schema(),
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda img: img.filename if hasattr(img, 'filename') else "PIL.Image"
        ),
    )

# Register Schema handler for PIL Image objects
Image.Image.__get_pydantic_core_schema__ = get_pydantic_core_schema  # type: ignore


# Load environment variables
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Get service configuration from config center
config = get_vlm_config()
SERVICE_NAME: str = config.service_name
SERVICE_VERSION: str = config.service_version
SERVICE_HOST: str = config.service_host
SERVICE_PORT: int = config.service_port

# Get general application configuration
app_config = get_app_config()
DEBUG: bool = app_config.debug
LOG_DIR: str = app_config.log.dir
LOG_LEVEL: str = app_config.log.level
LOG_ROTATION: str = app_config.log.rotation
LOG_RETENTION: str = app_config.log.retention

# Initialize VLM business service
vlm_service = VLMService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage VLM service lifecycle.

    Responsible for logging initialization, model loading, warmup, and resource release during service shutdown.
    """
    # Set service name to app.state (used by middleware)
    app.state.service_name = SERVICE_NAME
    db_runtime = create_database_runtime()
    app.state.db_runtime = db_runtime
    vlm_service.bind_session_factory(db_runtime.session_factory)
    app.state.model_registry = ModelRegistryService(
        app_id=app_config.app_id, session_factory=db_runtime.session_factory,
    )

    # 1. Initialize logging configuration
    log_conf = LogConfig(
        service_name=SERVICE_NAME, 
        log_dir=LOG_DIR, 
        level=LOG_LEVEL, 
        rotation=LOG_ROTATION, 
        retention=LOG_RETENTION
    )
    LogManager(log_conf).setup()
    
    # 2. Start initialization
    start_time = time.time()
    logger.info(f"正在启动 VLM 服务 | 进程号：{os.getpid()} | 时间：{datetime.now()}")
    
    try:
        await vlm_service.initialize()
        await vlm_service.warmup()
        logger.info(f"VLM 服务已就绪 | 耗时：{time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"VLM 服务初始化失败：{e}")
        if not app_config.debug:
            sys.exit(1)
    
    yield  # --- Running state ---
    
    # 3. Resource cleanup
    logger.info("正在停止 VLM 服务并释放资源...")
    try:
        await vlm_service.shutdown()
        logger.success("VLM service exited safely")
    except Exception as e:
        logger.error(f"清理资源时发生异常：{e}")
    finally:
        await db_runtime.close()


# --- Application Instance Configuration ---

app = FastAPIOffline(
    title="VLM Microservice",
    description="Provides multimodal vision-language model inference service",
    version=SERVICE_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if app_config.debug else None,
    redoc_url="/redoc" if app_config.debug else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
app.middleware("http")(log_requests)

# Internal service authentication middleware
from platform_core.platform.security import create_internal_auth_middleware
app.middleware("http")(create_internal_auth_middleware())
app.include_router(create_model_management_router(category=ModelCategory.VLM.value))

def get_vlm_service() -> VLMService:
    """Get VLM service instance dependency."""
    return vlm_service


# --- API Endpoint Implementations ---

@app.get("/health", response_model=dict, tags=["System"])
async def health_check() -> dict[str, Any]:
    """System health check."""
    loaded_models_count = 0
    if vlm_service._initialized and hasattr(vlm_service._model_pool, '_models'):
        loaded_models_count = len(vlm_service._model_pool._models)
    
    return {
        "status": "ok",
        "loaded_models_count": loaded_models_count,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/load", response_model=dict, tags=["Management"])
async def toggle_vlm_model(request: ToggleModelRequest) -> dict[str, str]:
    """Dynamically load or unload VLM models."""
    model_name = request.model_name
    try:
        if request.operation == "load":
            success = await vlm_service.load_model(model_name)
        else:
            success = await vlm_service.unload_model(model_name)
            
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to operate on model {model_name}")
        return {"status": "success", "model_name": model_name}
    except Exception as e:
        logger.exception(f"Model management exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/inference", response_model=VLMResponse, tags=["Inference"])
async def run_vlm_inference(
    request: VLMRequest,
    service: VLMService = Depends(get_vlm_service)
) -> VLMResponse | StreamingResponse:
    """Execute VLM inference tasks.

    Supports single return and SSE streaming return. Follows OpenAI chat completion API specifications.

    Args:
        request: Request body containing model, message stream (including images), and sampling parameters.
        service: VLM business logic service.

    Returns:
        JSON response or SSE text stream.
    """
    start_time = time.time()
    resp_id = f"vlm-chat-{uuid.uuid4()}"
    created_ts = int(time.time())
    model = await service.get_vlm_model(request.model_name)
    logger.info(f"收到推理请求 | 模型：{request.model_name} | 流式模式：{request.stream}")

    try:
        if request.stream:
            async def sse_generator() -> AsyncGenerator[str, None]:
                usage_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                
                try:
                    stream_raw = await service.inference(
                        **request.model_dump(exclude={"stream", "model_name"}), stream=True, model_name=request.model_name
                    )
                    
                    async for content in stream_raw:  # type: ignore
                        # Process metadata and Token statistics
                        if isinstance(content, str) and content.startswith("\n\n=== USAGE ==="):
                            try:
                                stats = json.loads(content.replace("\n\n=== USAGE ===\n", ""))
                                usage_stats.update(stats)
                            except (json.JSONDecodeError, ValueError):
                                pass
                            continue

                        # Construct standard OpenAI-style Chunk
                        chunk = {
                            "id": resp_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": model.model_name,
                            "choices": [{"delta": {"content": content}, "index": 0, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                    # Stream end: send final packet with Usage
                    final_chunk = {
                        "id": resp_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model.model_name,
                        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                        "usage": usage_stats
                    }
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

                except Exception as stream_err:
                    logger.error(f"流式生成中断：{stream_err}")
                    yield f"data: {json.dumps({'error': str(stream_err)})}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(sse_generator(), media_type="text/event-stream")

        # --- Non-streaming logic ---
        raw_resp = await service.inference(**request.model_dump(exclude={"stream"}), stream=False)
        duration = time.time() - start_time
        
        usage = raw_resp.get("usage", {}) # type: ignore
        content = raw_resp["choices"][0]["message"]["content"] # type: ignore

        return VLMResponse(
            id=resp_id,
            object="chat.completion",
            created=created_ts,
            model=model.model_name,
            choices=[{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "index": 0
            }],
            usage=usage,
            processing_time=duration
        )

    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
    except TimeoutError:
        raise HTTPException(status_code=408, detail="Model inference timeout")
    except Exception as e:
        logger.exception("Inference execution failed")
        raise HTTPException(status_code=500, detail=str(e))


# --- Process Management and Signal Monitoring ---

vlm_process: subprocess.Popen | None = None

def stop_standalone_vlm():
    """Clean up and shut down standalone VLM process."""
    global vlm_process
    if vlm_process:
        logger.info(f"正在停止独立 VLM 进程 [进程号：{vlm_process.pid}]...")
        vlm_process.terminate()
        try:
            vlm_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            vlm_process.kill()
        vlm_process = None

def handle_system_signal(sig: int, frame: Any):
    """Handle termination signals sent by the operating system."""
    logger.warning(f"收到系统信号：{sig}，正在触发安全退出...")
    stop_standalone_vlm()
    sys.exit(0)

atexit.register(stop_standalone_vlm)

if __name__ == "__main__":
    # Register signals for standalone running mode
    if os.environ.get("VLM_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, handle_system_signal)
        signal.signal(signal.SIGTERM, handle_system_signal)
    
    # 先检查端口可用性，避免 EADDRINUSE 错误被 stderr 吞掉
    if not check_port_available(SERVICE_HOST, SERVICE_PORT, SERVICE_NAME):
        sys.exit(1)

    logger.info(f"正在启动 VLM 服务 | 端口：{SERVICE_PORT}")
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT, access_log=False)
