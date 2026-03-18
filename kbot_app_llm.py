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
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi_offline import FastAPIOffline
from loguru import logger

from core.config.settings import get_llm_config, get_app_config
from core.logger import LogConfig, LogManager
from core.middleware.log_middleware import log_requests
from core.dictionary import LLMProvider
from microservices.llm.llm_service import LLMService
from microservices.llm.schema import *

# Load environment variables
load_dotenv()

# Service basic information
config = get_llm_config()
SERVICE_NAME = config.service_name
SERVICE_VERSION = config.service_version
SERVICE_HOST = config.service_host
SERVICE_PORT = config.service_port

# Log and debug configuration
app_config = get_app_config()
DEBUG_MODE = app_config.debug
LOG_DIR = app_config.log.dir
LOG_LEVEL = app_config.log.level
LOG_ROTATION = app_config.log.rotation
LOG_RETENTION = app_config.log.retention

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

    # Initialize logging system
    log_conf = LogConfig(
        service_name=SERVICE_NAME,
        log_dir=LOG_DIR,
        level=LOG_LEVEL,
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION
    )
    LogManager(log_conf).setup()

    start_time = time.time()
    logger.info(f"Starting LLM service | PID: {os.getpid()} | Time: {datetime.now()}")

    try:
        await llm_service.initialize()
        await llm_service.warmup()
        logger.info(f"LLM service initialization completed | Elapsed time: {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.exception(f"Failed to start LLM service: {e}")
        if not DEBUG_MODE:
            sys.exit(1)

    yield  # --- Runtime phase ---

    logger.info("Performing shutdown cleanup...")
    try:
        await llm_service.shutdown()
        logger.info("LLM service has been shut down safely")
    except Exception as e:
        logger.error(f"Exception occurred while shutting down service: {e}")


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


@app.post("/load", response_model=dict, tags=["Management"], summary="Load/Unload Model")
async def handle_toggle_model(request: ToggleModelRequest) -> dict[str, Any]:
    """Dynamically manage models in memory.

    Args:
        request: Model operation request.

    Returns:
        Operation result.
    """
    try:
        method = llm_service.load_model if request.operation == "load" else llm_service.unload_model
        logger.info(f"Executing model operation: {request.operation} -> {request.model_name}")
        
        success = await method(request.model_name)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to {request.operation} model {request.model_name}")
            
        return {"status": "success", "model_name": request.model_name}
    except Exception as e:
        logger.exception(f"Model management exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions", response_model=None, tags=["LLM"], summary="Chat Completion Endpoint")
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
    model = await service.get_model_instance(request.model_name)
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
                        model_name=request.model_name,
                        messages=request.messages,
                        stream=True,
                        max_tokens=current_max_tokens,
                        temperature=request.temperature,
                        timeout=request.timeout,
                        tools=request.tools,
                        tool_choice=request.tool_choice
                    )

                    async for chunk in stream_iter: # type: ignore
                        if chunk == "[DONE]":
                            break

                        # Unified serialization for different Provider Chunks
                        if hasattr(chunk, 'model_dump_json'):
                            # OpenAI-compatible format Chunk
                            data = chunk.model_dump_json()
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
                                    "model": request.model_name,
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
                                data = json.dumps(chunk)
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
            model_name=request.model_name,
            messages=request.messages,
            stream=False,
            max_tokens=current_max_tokens,
            temperature=request.temperature,
            timeout=request.timeout,
            tools=request.tools,
            tool_choice=request.tool_choice
        )

        proc_time = time.time() - start_time
        logger.info(f"Request processing completed | Model: {request.model_name} | Elapsed time: {proc_time:.2f}s")

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

        logger.debug(f"Starting response parsing | Provider: {provider} | Raw Response Type: {type(raw_resp)}")

        if provider in openai_family:
            try:
                msg = raw_resp.choices[0].message # type: ignore
                content = msg.content
                logger.debug(f"OpenAI response content length: {len(content) if content else 0}")
                usage = raw_resp.usage if isinstance(raw_resp.usage, dict) else raw_resp.usage.model_dump() # type: ignore

                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(ToolCall(
                            id=tc.id, type="function",
                            function={"name": tc.function.name, "arguments": tc.function.arguments} # type: ignore
                        ))
            except Exception as e:
                logger.error(f"Failed to parse OpenAI response: {e}")
                logger.error(f"Raw Response: {raw_resp}")
                content = ""

        elif provider == LLMProvider.OCI.value:
            # 1. Get internal response object
            oci_resp = raw_resp.data.chat_response # type: ignore
            logger.debug(f"OCI response object: {type(oci_resp)}")

            # 2. Extract Content (distinguish between Generic format and Cohere format)
            if hasattr(oci_resp, 'choices'): # Generic format (Llama, Grok, etc.)
                content = oci_resp.choices[0].message.content[0].text
            elif hasattr(oci_resp, 'text'): # Cohere format
                content = getattr(oci_resp, 'text', "")
            else:
                logger.warning(f"Unknown OCI response format: {dir(oci_resp)}")
                content = ""

            logger.debug(f"OCI response content length: {len(content) if content else 0}")

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
            logger.warning(f"Unknown Provider: {provider}")
            content = ""

        return ChatResponse(
            id=resp_id,
            object="chat.completion",
            created=created_ts,
            model=request.model_name,
            choices=[{"message": {"role": "assistant", "content": content or ""}, "finish_reason": "stop", "index": 0}],
            usage=UsageInfo(**usage),
            processing_time=proc_time,
            tool_calls=tool_calls if tool_calls else None
        )

    except Exception as e:
        logger.exception("Error occurred while generating chat response")
        raise HTTPException(status_code=500, detail=str(e))


def signal_handler(sig: int, frame: Any):
    """Catch system signals to implement graceful shutdown."""
    logger.warning(f"Received signal {sig}, exiting...")
    sys.exit(0)


if __name__ == "__main__":
    # Register signal handlers only in standalone mode
    if os.environ.get("LLM_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"LLM adaptation layer is ready -> {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(
        app,
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        log_config=None  # Fully managed by Loguru
    )