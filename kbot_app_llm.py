"""LLM 微服务应用程序。

本模块提供基于 FastAPI 的 LLM 接入层，支持文本生成、对话补全、流式响应（SSE）
以及基于 MCP 协议的工具调用（Tool Calling）功能。
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

# 加载环境变量
load_dotenv()

# 服务基础信息
config = get_llm_config()
SERVICE_NAME = config.service_name
SERVICE_VERSION = config.service_version
SERVICE_HOST = config.service_host
SERVICE_PORT = config.service_port

# 日志与调试配置
app_config = get_app_config()
DEBUG_MODE = app_config.debug
LOG_DIR = app_config.log.dir
LOG_LEVEL = app_config.log.level
LOG_ROTATION = app_config.log.rotation
LOG_RETENTION = app_config.log.retention

# 初始化 LLM 逻辑服务单例
llm_service = LLMService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用程序生命周期。

    Args:
        app: FastAPI 实例。
    """
    # 设置服务名称到 app.state（供中间件使用）
    app.state.service_name = SERVICE_NAME

    # 初始化日志系统
    log_conf = LogConfig(
        service_name=SERVICE_NAME,
        log_dir=LOG_DIR,
        level=LOG_LEVEL,
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION
    )
    LogManager(log_conf).setup()

    start_time = time.time()
    logger.info(f"正在启动 LLM 服务 | PID: {os.getpid()} | 时间: {datetime.now()}")

    try:
        await llm_service.initialize()
        await llm_service.warmup()
        logger.info(f"LLM 服务初始化完成 | 耗时: {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.exception(f"LLM 服务启动失败: {e}")
        if not DEBUG_MODE:
            sys.exit(1)

    yield  # --- 运行阶段 ---

    logger.info("正在执行关机清理...")
    try:
        await llm_service.shutdown()
        logger.info("LLM 服务已安全关闭")
    except Exception as e:
        logger.error(f"关闭服务时发生异常: {e}")


# 初始化 FastAPI 应用
app = FastAPIOffline(
    title="LLM 微服务",
    description="提供多供应商 LLM 适配、流式聊天及工具调用支持。",
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

# 4. 请求日志中间件
app.middleware("http")(log_requests)


def get_llm_service() -> LLMService:
    """依赖注入获取 LLM 服务实例。"""
    return llm_service


@app.get("/health", response_model=dict, tags=["System"], summary="健康检查")
async def health_check() -> dict[str, Any]:
    """检查服务健康状态。

    Returns:
        包含状态、加载模型数和时间戳的字典。
    """
    loaded_models_count = 0
    if llm_service._initialized and hasattr(llm_service._model_pool, '_models'):
        loaded_models_count = len(llm_service._model_pool._models)

    return {
        "status": "ok",
        "loaded_models_count": loaded_models_count,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/load", response_model=dict, tags=["Management"], summary="加载/卸载模型")
async def handle_toggle_model(request: ToggleModelRequest) -> dict[str, Any]:
    """动态管理内存中的模型。

    Args:
        request: 模型操作请求。

    Returns:
        操作结果。
    """
    try:
        method = llm_service.load_model if request.operation == "load" else llm_service.unload_model
        logger.info(f"执行模型操作: {request.operation} -> {request.model_name}")
        
        success = await method(request.model_name)
        if not success:
            raise HTTPException(status_code=500, detail=f"模型 {request.model_name} {request.operation} 失败")
            
        return {"status": "success", "model_name": request.model_name}
    except Exception as e:
        logger.exception(f"模型管理异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions", response_model=None, tags=["LLM"], summary="对话补全接口")
async def handle_chat_completions(
    request: ChatRequest,
    service: LLMService = Depends(get_llm_service)
) -> ChatResponse | StreamingResponse:
    """处理聊天补全请求，支持流式与非流式。

    Args:
        request: 对话请求参数。
        service: 注入的 LLM 服务。

    Returns:
        ChatResponse 对象或用于 SSE 的 StreamingResponse。

    Raises:
        HTTPException: 404 模型不存在, 400 校验错误, 408 超时, 500 服务器内部错误。
    """
    start_time = time.time()
    resp_id = f"chatcmpl-{uuid.uuid4()}"
    created_ts = int(time.time())

    # 先加载模型（如果模型未加载，此步会触发异步加载）
    model = await service.get_model_instance(request.model_name)
    provider = model.config.provider

    # 获取最大 Token 限制（如果用户没传，则使用模型配置的默认值）
    max_tokens_limit = getattr(model.config, "max_tokens", 4096)
    current_max_tokens = request.max_tokens or max_tokens_limit

    try:
        # --- 流式响应逻辑 ---
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

                        # 统一序列化不同 Provider 的 Chunk
                        if hasattr(chunk, 'model_dump_json'):
                            # OpenAI 兼容格式的 Chunk
                            data = chunk.model_dump_json()
                        elif isinstance(chunk, dict):
                            # 检查是否为 OCI 原生格式，需要转换为 OpenAI 格式
                            text = None

                            # 1. OCI Cohere 格式: {"apiFormat": "COHERE", "text": "你好", "pad": "..."}
                            if 'apiFormat' in chunk and chunk.get('apiFormat') == 'COHERE':
                                text = chunk.get('text', '')

                            # 2. OCI Generic/Grok 格式: {"index": 0, "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "你好"}]}, "pad": "..."}
                            elif 'message' in chunk and isinstance(chunk.get('message'), dict):
                                message = chunk['message']
                                content = message.get('content', [])
                                if content and isinstance(content, list) and len(content) > 0:
                                    # 提取 content[0].text
                                    first_content = content[0]
                                    if isinstance(first_content, dict) and first_content.get('type') == 'TEXT':
                                        text = first_content.get('text', '')

                            if text is not None:
                                # 转换为 OpenAI 标准格式
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
                    logger.exception(f"流式响应中断: {stream_err}")
                    yield f"data: {json.dumps({'error': str(stream_err)})}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                sse_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
            )

        # --- 非流式响应逻辑 ---
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
        logger.info(f"请求处理完成 | 模型: {request.model_name} | 耗时: {proc_time:.2f}s")

        # 解析不同 Provider 的结果
        content: str | None = None
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tool_calls: list[ToolCall] = []

        # 归一化 OpenAI 家族的 Provider 判断
        openai_family = [
            LLMProvider.CHATGPT.value,
            LLMProvider.API_DEEPSEEK.value,
            LLMProvider.API_QWEN.value
        ]

        logger.debug(f"开始解析响应 | Provider: {provider} | Raw Response Type: {type(raw_resp)}")

        if provider in openai_family:
            try:
                msg = raw_resp.choices[0].message # type: ignore
                content = msg.content
                logger.debug(f"OpenAI 响应 content 长度: {len(content) if content else 0}")
                usage = raw_resp.usage if isinstance(raw_resp.usage, dict) else raw_resp.usage.model_dump() # type: ignore

                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(ToolCall(
                            id=tc.id, type="function",
                            function={"name": tc.function.name, "arguments": tc.function.arguments} # type: ignore
                        ))
            except Exception as e:
                logger.error(f"解析 OpenAI 响应失败: {e}")
                logger.error(f"Raw Response: {raw_resp}")
                content = ""

        elif provider == LLMProvider.OCI.value:
            # 1. 获取内部响应对象
            oci_resp = raw_resp.data.chat_response # type: ignore
            logger.debug(f"OCI 响应对象: {type(oci_resp)}")

            # 2. 提取 Content (区分 Generic 格式和 Cohere 格式)
            if hasattr(oci_resp, 'choices'): # Generic 格式 (Llama, Grok 等)
                content = oci_resp.choices[0].message.content[0].text
            elif hasattr(oci_resp, 'text'): # Cohere 格式
                content = getattr(oci_resp, 'text', "")
            else:
                logger.warning(f"未知的 OCI 响应格式: {dir(oci_resp)}")
                content = ""

            logger.debug(f"OCI 响应 content 长度: {len(content) if content else 0}")

            # 3. 提取 Usage (核心修复点)
            # 使用 oci.util.to_dict 将 SDK 对象转为字典，安全获取 usage 字段
            resp_dict = oci.util.to_dict(oci_resp)
            raw_usage = resp_dict.get("usage")

            if raw_usage:
                # 转换 OCI 字段名到 OpenAI 标准字段名
                usage = {
                    "prompt_tokens": raw_usage.get("input_tokens", 0),
                    "completion_tokens": raw_usage.get("output_tokens", 0),
                    "total_tokens": raw_usage.get("total_tokens", 0)
                }
            else:
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        else:
            logger.warning(f"未知的 Provider: {provider}")
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
        logger.exception("生成对话响应时发生错误")
        raise HTTPException(status_code=500, detail=str(e))


def signal_handler(sig: int, frame: Any):
    """捕捉系统信号实现优雅停机。"""
    logger.warning(f"接收到信号 {sig}，正在退出...")
    sys.exit(0)


if __name__ == "__main__":
    # 仅在独立模式下注册信号处理
    if os.environ.get("LLM_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"LLM 适配层已就绪 -> {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(
        app,
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        log_config=None  # 完全由 Loguru 接管
    )