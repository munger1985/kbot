"""LLM Microservice Application.

This module provides a FastAPI-based LLM access layer that supports text generation, chat completion, streaming responses (SSE),
and Tool Calling functionality based on the MCP protocol.
"""

import os
import sys
import signal
import json
import time
import oci
import uuid
from datetime import datetime
from typing import Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi_offline import FastAPIOffline
from loguru import logger

from model_serving.config import get_model_serving_settings
from platform_core.contracts import INTERNAL_API_V1, PUBLIC_API_V1
from platform_core.dictionary import ModelCategory
from platform_core.logger import LogConfig, LogManager
from platform_core.middleware.log_middleware import log_requests
from platform_core.database.oracle import create_database_runtime
from platform_core.dictionary import LLMProvider
from model_serving.llm.llm_service import LLMService
from model_serving.llm.schema import *
from platform_core.platform.port_check import check_port_available
from model_serving.common.management_router import create_model_management_router
from model_serving.common.openai_router import create_openai_models_router
from model_serving.common.openai_contracts import openai_error_response
from model_serving.common.model_registry import ModelRegistryService

# Service basic information
settings = get_model_serving_settings()
config = settings.llm
SERVICE_NAME = config.service_name
SERVICE_VERSION = config.service_version
SERVICE_HOST = config.service_host
SERVICE_PORT = config.service_port

# Log and debug configuration
DEBUG_MODE = settings.platform.debug
LOG_DIR = settings.log.dir
LOG_LEVEL = settings.log.level
LOG_ROTATION = settings.log.rotation
LOG_RETENTION = settings.log.retention

# Initialize LLM logic service singleton
llm_service = LLMService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle.

    Args:
        app: FastAPI instance.
    """
    # Set service name to app.state (used by middleware)
    app.state.service_name = SERVICE_NAME
    db_runtime = create_database_runtime()
    app.state.db_runtime = db_runtime
    llm_service.bind_session_factory(db_runtime.session_factory)
    app.state.model_registry = ModelRegistryService(
        app_id=settings.platform.app_id,
        session_factory=db_runtime.session_factory,
        on_model_changed=llm_service.invalidate_model,
    )

    # Initialize logging system
    log_conf = LogConfig(
        service="model_serving",
        process="llm",
        log_dir=LOG_DIR,
        level=LOG_LEVEL,
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION
    )
    LogManager(log_conf).setup()

    start_time = time.time()
    logger.info(f"正在启动 LLM 服务 | 进程号：{os.getpid()} | 时间：{datetime.now()}")

    try:
        await llm_service.initialize()
        await llm_service.warmup()
        logger.info(f"LLM 服务初始化完成 | 耗时：{time.time() - start_time:.2f}s")
    except Exception as e:
        logger.exception(f"Failed to start LLM service: {e}")
        if not DEBUG_MODE:
            sys.exit(1)

    yield  # --- Runtime phase ---

    logger.info("正在执行停止前清理...")
    try:
        await llm_service.shutdown()
        logger.info("LLM 服务已安全停止")
    except Exception as e:
        logger.error(f"停止服务时发生异常：{e}")
    finally:
        await db_runtime.close()


# Initialize FastAPI application
app = FastAPIOffline(
    title="LLM Microservice",
    description="Provides multi-provider LLM adaptation, streaming chat, and tool calling support.",
    version=SERVICE_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if DEBUG_MODE else None,
    redoc_url="/redoc" if DEBUG_MODE else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Request logging middleware
app.middleware("http")(log_requests)

# 5. Internal service authentication middleware
from platform_core.security import (
    create_api_client_auth_middleware,
    create_internal_auth_middleware,
)
app.middleware("http")(
    create_internal_auth_middleware(
        audience=SERVICE_NAME, skip_prefixes=(PUBLIC_API_V1,),
    )
)
app.middleware("http")(create_api_client_auth_middleware())
app.include_router(create_model_management_router(category=ModelCategory.LLM.value))
app.include_router(create_openai_models_router(category=ModelCategory.LLM.value))


def get_llm_service() -> LLMService:
    """Get LLM service instance via dependency injection."""
    return llm_service


@app.get("/health", response_model=dict, tags=["System"], summary="Health Check")
async def health_check() -> dict[str, Any]:
    """Check service health status.

    Returns:
        Dictionary containing status, number of loaded models, and timestamp.
    """
    loaded_models_count = 0
    if llm_service._initialized and hasattr(llm_service._model_pool, '_models'):
        loaded_models_count = len(llm_service._model_pool._models)

    return {
        "status": "ok",
        "loaded_models_count": loaded_models_count,
        "timestamp": datetime.now().isoformat()
    }


@app.post(f"{INTERNAL_API_V1}/chat/completions", response_model=None, tags=["LLM"], summary="对话补全")
async def handle_chat_completions(
    request: ChatRequest,
    service: LLMService = Depends(get_llm_service)
) -> ChatResponse | StreamingResponse:
    """Handle chat completion requests, supporting streaming and non-streaming modes.

    Args:
        request: Chat request parameters.
        service: Injected LLM service.

    Returns:
        ChatResponse object or StreamingResponse for SSE.

    Raises:
        HTTPException: 404 Model not found, 400 Validation error, 408 Timeout, 500 Internal server error.
    """
    start_time = time.time()
    resp_id = f"chatcmpl-{uuid.uuid4()}"
    created_ts = int(time.time())

    # Load model first (triggers async loading if model is not loaded)
    model = await service.get_llm_model(request.served_model_name)
    provider = model.config.provider

    # Get max token limit (use model config default if not provided by user)
    max_tokens_limit = getattr(model.config, "max_tokens", 4096)
    current_max_tokens = request.max_tokens or max_tokens_limit

    try:
        # --- Streaming response logic ---
        if request.stream:
            async def sse_generator():
                try:
                    stream_iter = await service.chat(
                        served_model_name=request.served_model_name,
                        messages=request.messages,
                        stream=True,
                        max_tokens=current_max_tokens,
                        temperature=request.temperature,
                        timeout=request.timeout,
                        tools=request.tools,
                        tool_choice=request.tool_choice,
                        response_format=request.response_format
                    )

                    async for chunk in stream_iter: # type: ignore
                        if chunk == "[DONE]":
                            break

                        # Unified serialization for different Provider Chunks
                        if hasattr(chunk, "model_dump"):
                            payload = chunk.model_dump()
                            payload["model"] = request.served_model_name
                            data = json.dumps(payload, ensure_ascii=False)
                        elif isinstance(chunk, dict):
                            # Check if it's OCI native format and convert to OpenAI format
                            text = None

                            # 1. OCI Cohere format: {"apiFormat": "COHERE", "text": "Hello", "pad": "..."}
                            if 'apiFormat' in chunk and chunk.get('apiFormat') == 'COHERE':
                                text = chunk.get('text', '')

                            # 2. OCI Generic/Grok format: {"index": 0, "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "Hello"}]}, "pad": "..."}
                            elif 'message' in chunk and isinstance(chunk.get('message'), dict):
                                message = chunk['message']
                                content = message.get('content', [])
                                if content and isinstance(content, list) and len(content) > 0:
                                    # Extract content[0].text
                                    first_content = content[0]
                                    if isinstance(first_content, dict) and first_content.get('type') == 'TEXT':
                                        text = first_content.get('text', '')

                            if text is not None:
                                # Convert to OpenAI standard format
                                openai_chunk = {
                                    "id": resp_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_ts,
                                    "model": request.served_model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": text},
                                            "finish_reason": None
                                        }
                                    ]
                                }
                                data = json.dumps(openai_chunk, ensure_ascii=False)
                            else:
                                payload = dict(chunk)
                                if str(payload.get("object", "")).startswith(
                                    "chat.completion"
                                ):
                                    payload["model"] = request.served_model_name
                                data = json.dumps(payload, ensure_ascii=False)
                        else:
                            data = str(chunk)

                        yield f"data: {data}\n\n"

                    yield "data: [DONE]\n\n"
                except Exception as stream_err:
                    logger.exception(f"Streaming response interrupted: {stream_err}")
                    yield f"data: {json.dumps({'error': str(stream_err)})}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                sse_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
            )

        # --- Non-streaming response logic ---
        raw_resp = await service.chat(
            served_model_name=request.served_model_name,
            messages=request.messages,
            stream=False,
            max_tokens=current_max_tokens,
            temperature=request.temperature,
            timeout=request.timeout,
            tools=request.tools,
            tool_choice=request.tool_choice,
            response_format=request.response_format
        )

        proc_time = time.time() - start_time
        logger.info(f"请求处理完成 | 模型：{request.served_model_name} | 耗时：{proc_time:.2f}s")

        # Parse results from different Providers
        content: str | None = None
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tool_calls: list[ToolCall] = []

        # Normalize OpenAI family Provider judgment
        openai_family = [
            LLMProvider.CHATGPT.value,
            LLMProvider.API_DEEPSEEK.value,
            LLMProvider.API_QWEN.value
        ]

        logger.debug(f"开始解析响应 | Provider：{provider} | 原始响应类型：{type(raw_resp)}")

        if provider in openai_family:
            try:
                msg = raw_resp.choices[0].message # type: ignore
                content = msg.content
                logger.debug(f"OpenAI 响应内容长度：{len(content) if content else 0}")
                usage = raw_resp.usage if isinstance(raw_resp.usage, dict) else raw_resp.usage.model_dump() # type: ignore

                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(ToolCall(
                            id=tc.id, type="function",
                            function={"name": tc.function.name, "arguments": tc.function.arguments} # type: ignore
                        ))
            except Exception as e:
                logger.error(f"解析 OpenAI 响应失败：{e}")
                logger.error(f"原始响应：{raw_resp}")
                content = ""

        elif provider == LLMProvider.OCI.value:
            # 1. Get internal response object
            oci_resp = raw_resp.data.chat_response # type: ignore
            logger.debug(f"OCI 响应对象：{type(oci_resp)}")

            # 2. Extract Content (distinguish between Generic format and Cohere format)
            if hasattr(oci_resp, 'choices'): # Generic format (Llama, Grok, etc.)
                content = oci_resp.choices[0].message.content[0].text
            elif hasattr(oci_resp, 'text'): # Cohere format
                content = getattr(oci_resp, 'text', "")
            else:
                logger.warning(f"无法识别的 OCI 响应格式：{dir(oci_resp)}")
                content = ""

            logger.debug(f"OCI 响应内容长度：{len(content) if content else 0}")

            # 3. Extract Usage (core fix point)
            # Convert SDK object to dictionary using oci.util.to_dict to safely get usage field
            resp_dict = oci.util.to_dict(oci_resp)
            raw_usage = resp_dict.get("usage")

            if raw_usage:
                # Convert OCI field names to OpenAI standard field names
                usage = {
                    "prompt_tokens": raw_usage.get("input_tokens", 0),
                    "completion_tokens": raw_usage.get("output_tokens", 0),
                    "total_tokens": raw_usage.get("total_tokens", 0)
                }
            else:
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        else:
            logger.warning(f"无法识别的 Provider：{provider}")
            content = ""

        response_message: dict[str, Any] = {
            "role": "assistant",
            "content": content or "",
        }
        if tool_calls:
            response_message["tool_calls"] = [
                tool_call.model_dump() for tool_call in tool_calls
            ]
        return ChatResponse(
            id=resp_id,
            object="chat.completion",
            created=created_ts,
            model=request.served_model_name,
            choices=[{
                "message": response_message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "index": 0,
            }],
            usage=UsageInfo(**usage),
        )

    except Exception as e:
        logger.exception("Error occurred while generating chat response")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    f"{PUBLIC_API_V1}/chat/completions",
    response_model=None,
    tags=["OpenAI Compatible"],
)
async def openai_chat_completions(
    request: OpenAIChatRequest,
    service: LLMService = Depends(get_llm_service),
):
    """把 OpenAI 的 model 字段解析为 served_model_name。"""
    try:
        result = await handle_chat_completions(request.to_internal(), service)
    except HTTPException as exc:
        return openai_error_response(
            status_code=exc.status_code,
            message=str(exc.detail),
            code="model_request_failed",
        )
    except Exception as exc:
        logger.error(f"OpenAI Chat 调用失败：{exc}")
        return openai_error_response(
            status_code=500,
            message="模型推理失败",
            code="model_inference_failed",
            error_type="server_error",
        )
    if isinstance(result, StreamingResponse):
        return result
    return result.model_dump(exclude_none=True)


def signal_handler(sig: int, frame: Any):
    """Catch system signals to implement graceful shutdown."""
    logger.warning(f"收到系统信号 {sig}，正在退出...")
    sys.exit(0)


if __name__ == "__main__":
    # Register signal handlers only in standalone mode
    if os.environ.get("LLM_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    # 先检查端口可用性，避免 EADDRINUSE 错误被 stderr 吞掉
    if not check_port_available(SERVICE_HOST, SERVICE_PORT, "LLM"):
        sys.exit(1)

    logger.info(f"LLM 适配层已就绪 -> {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(
        app,
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        log_config=None  # Fully managed by Loguru
    )
