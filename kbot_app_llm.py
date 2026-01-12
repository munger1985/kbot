"""LLM 微服务应用程序。

本模块提供基于 FastAPI 的 LLM 接入层，支持文本生成、对话补全、流式响应（SSE）
以及基于 MCP 协议的工具调用（Tool Calling）功能。
"""

import os
import sys
import signal
import json
import time
import atexit
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
from pydantic import ValidationError
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
        logger.info(f"执行模型操作: {request.operation} -> {request.model_id}")
        
        success = await method(request.model_id)
        if not success:
            raise HTTPException(status_code=500, detail=f"模型 {request.model_id} {request.operation} 失败")
            
        return {"status": "success", "model_id": request.model_id}
    except Exception as e:
        logger.exception(f"模型管理异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 主要的聊天接口（重点改造部分） ====================

@app.post("/v1/chat/completions", response_model=None, tags=["LLM"], summary="生成聊天响应")
async def chat(
    request: ChatRequest,
    llm_service: LLMService = Depends(get_llm_service)
) -> ChatResponse | StreamingResponse:
    """生成聊天响应，支持MCP工具调用
    
    Args:
    - **model_id**: int = Field(..., description="要使用的特定模型ID")
    - **messages**: list[dict[str, str]] | str = Field(..., description="聊天消息列表")
    - **max_tokens**: int | None = Field(None, description="要生成的最大令牌数")
    - **temperature**: float | None = Field(None, description="采样温度（0.0-1.0，越低越确定）")
    - **stream**: bool = Field(False, description="是否流式传输响应")
    - **timeout**: int | None = Field(None, description="超时时间（秒）")
    - **top_p**: float | None = Field(None, description="Top-p采样参数")
    - **frequency_penalty**: float | None = Field(None, description="频率惩罚")
    - **presence_penalty**: float | None = Field(None, description="存在惩罚")
    - **tools**: list[dict[str, Any]] | None = Field(None, description="MCP工具列表")
    - **tool_choice**: Any | None = Field(None, description="工具选择策略")
    - **enable_tool_calls**: bool = Field(False, description="是否启用工具调用")
    
    Returns:
    **非流式模式**: 包含消息和处理时间的JSON对象
    - **id**: str = Field(default_factory=lambda: f"sse-{uuid.uuid4()}", description="响应流的唯一标识符")
    - **object**: str = Field("chat.completion", description="对象类型，始终为'chat.completion'")
    - **created**: int = Field(default_factory=lambda: int(time.time()), description="响应创建时的Unix时间戳")
    - **model**: str = Field(..., description="响应模型名称")
    - **choices**: list[dict[str, Any]] = Field(..., description="包含消息的聊天完成选项列表")
    - **usage**: dict[str, int] = Field(..., description="令牌使用统计，包括prompt_tokens、completion_tokens和total_tokens")
    - **processing_time**: float = Field(..., description="处理时间（秒）（自定义字段）")

    **流式模式**: 标准OpenAI SSE格式
    ```
    data: {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model_id,
        "choices": [{
            "delta": {"content": content},
            "index": 0,
            "finish_reason": None
        }]
    }
    data: [DONE]
    ```

    Raises:
    - **HTTPException**: 当聊天生成失败时抛出500错误，当响应格式不支持时抛出400错误，当请求超时时抛出408错误
    """
    
    start_time = time.time()
    response_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    model_name = llm_service._model_pool._model_names.get(request.model_id, str(request.model_id))
    model_token = llm_service._model_pool._max_tokens.get(request.model_id, 4000)
    provider = llm_service.get_provider(request.model_id)
    
    if provider is None:
        raise HTTPException(status_code=404, detail=f"模型 {model_name} 在模型池中未找到")
    
    try:
        # 准备工具调用参数
        tools = request.tools
        tool_choice = request.tool_choice
        
        # 如果启用了工具调用但没有提供工具，记录警告
        if request.enable_tool_calls and not tools:
            logger.warning("启用了工具调用但没有提供工具列表")
        
        logger.debug(f"工具调用配置 - tools: {len(tools) if tools else 0}, tool_choice: {tool_choice}")
        
        # 处理流式响应
        if request.stream and provider == LLMProvider.OPENAI.value:
            async def generate_openai_sse():
                try:
                    # 获取流式响应，包含工具调用支持
                    max_tokens = request.max_tokens if request.max_tokens else model_token
                    chunk_stream = await llm_service.chat(
                        model_id=request.model_id,
                        messages=request.messages,
                        stream=True,
                        max_tokens=max_tokens,
                        temperature=request.temperature,
                        timeout=request.timeout,
                        top_p=request.top_p,
                        frequency_penalty=request.frequency_penalty,
                        presence_penalty=request.presence_penalty,
                        tools=tools,
                        tool_choice=tool_choice
                    )
                    
                    # 直接返回OpenAI客户端的流式响应
                    async for chunk in chunk_stream: # type: ignore
                        try:
                            if hasattr(chunk, 'model_dump_json') and callable(getattr(chunk, 'model_dump_json')):
                                yield f"data: {chunk.model_dump_json()}\n\n"
                            elif isinstance(chunk, str):
                                yield f"data: {chunk}\n\n"
                            else:
                                yield f"data: {json.dumps(chunk)}\n\n"
                        except Exception as e:
                            logger.error(f"序列化失败: {e}, chunk类型: {type(chunk)}, 内容: {chunk}")
                            yield f"data: {json.dumps({'error': '序列化失败'})}\n\n"
                    
                    yield "data: [DONE]\n\n"
                    
                except Exception as e:
                    logger.exception(f"流式传输错误: {str(e)}")
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": e.__class__.__name__,
                            "code": 500
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_openai_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        # 处理OCI流式响应（保持原有逻辑）
        elif request.stream and provider == LLMProvider.OCI.value:
            if "cohere" in model_name.lower():
                async def generate_oci_cohere_sse():
                    try:
                        # 获取流式响应
                        chunk_stream = await llm_service.chat(
                            model_id=request.model_id,
                            messages=request.messages,
                            stream=True,
                            max_tokens=request.max_tokens,
                            temperature=request.temperature,
                            timeout=request.timeout,
                            top_p=request.top_p,
                            frequency_penalty=request.frequency_penalty,
                            presence_penalty=request.presence_penalty
                        )
                        
                        async for chunk in chunk_stream: # type: ignore
                            if chunk == "[DONE]":
                                break
                            try:
                                content = chunk["text"]
                                chunk_dict = {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": model_name,
                                    "choices": [{
                                        "delta": {"content": content},
                                        "index": 0,
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(chunk_dict)}\n\n"
                            except KeyError:
                                continue
                        
                        # 发送结束标记
                        end_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model_name,
                            "choices": [{
                                "delta": {},
                                "index": 0,
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(end_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        
                    except Exception as e:
                        logger.exception(f"流式传输错误: {str(e)}")
                        error_chunk = {
                            "error": {
                                "message": str(e),
                                "type": e.__class__.__name__,
                                "code": 500
                            }
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        yield "data: [DONE]\n\n"

                return StreamingResponse(
                    generate_oci_cohere_sse(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive"
                    }
                )
        
            elif request.stream and ("grok" in model_name.lower() or "llama" in model_name.lower()):
                async def generate_oci_grok_sse():
                    try:
                        # 获取流式响应
                        chunk_stream = await llm_service.chat(
                            model_id=request.model_id,
                            messages=request.messages,
                            stream=True,
                            max_tokens=request.max_tokens,
                            temperature=request.temperature,
                            timeout=request.timeout,
                            top_p=request.top_p,
                            frequency_penalty=request.frequency_penalty,
                            presence_penalty=request.presence_penalty
                        )
                        
                        async for chunk in chunk_stream: # type: ignore
                            if chunk == "[DONE]":
                                break
                            try:
                                content = chunk["message"]["content"][0]["text"]
                                chunk_dict = {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": model_name,
                                    "choices": [{
                                        "delta": {"content": content},
                                        "index": 0,
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(chunk_dict)}\n\n"
                            except KeyError:
                                continue
                        
                        # 发送结束标记
                        end_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model_name,
                            "choices": [{
                                "delta": {},
                                "index": 0,
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(end_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        
                    except Exception as e:
                        logger.exception(f"流式传输错误: {str(e)}")
                        error_chunk = {
                            "error": {
                                "message": str(e),
                                "type": e.__class__.__name__,
                                "code": 500
                            }
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        yield "data: [DONE]\n\n"

                return StreamingResponse(
                    generate_oci_grok_sse(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive"
                    }
                )
            else:
                raise HTTPException(status_code=400, detail="流式传输不支持该模型")
            
        # 处理非流式响应
        elif not request.stream:
            # 调用LLM服务，支持工具调用
            response = await llm_service.chat(
                model_id=request.model_id,
                messages=request.messages,
                stream=False,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                timeout=request.timeout,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                tools=tools,
                tool_choice=tool_choice
            )
            
            processing_time = time.time() - start_time
            logger.info(f"聊天完成耗时: {processing_time:.2f}秒")
            
            content = None
            usage_data = None
            tool_calls = None
            
            # 处理不同提供者的响应
            if provider == LLMProvider.OPENAI.value:
                content = response.choices[0].message.content # type: ignore
                usage_data = response.usage # type: ignore
                
                # 检查是否有工具调用
                if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls: # type: ignore
                    tool_calls = response.choices[0].message.tool_calls # type: ignore
                    logger.info(f"检测到工具调用: {len(tool_calls)} 个")

            elif provider == LLMProvider.OCI.value:
                # Grok 非流式响应
                if "grok" in model_name.lower() or "llama" in model_name.lower():
                    content = response.data.chat_response.choices[0].message.content[0].text # type: ignore
                    usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                # Cohere 非流式响应
                else:
                    content = response.data.chat_response.text # type: ignore
                    usage_data = response.data.chat_response.usage # type: ignore

            else:
                logger.warning(f"不支持的提供者: {provider}")

            # 构建响应
            chat_response = ChatResponse(
                id=response_id,
                object="chat.completion",
                created=created_time,
                model=model_name,
                choices=[{
                     "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop",
                    "index": 0
                }],
                usage={
                    "prompt_tokens": usage_data["prompt_tokens"] if isinstance(usage_data, dict) else usage_data.prompt_tokens, # type: ignore
                    "completion_tokens": usage_data["completion_tokens"] if isinstance(usage_data, dict) else usage_data.completion_tokens, # type: ignore
                    "total_tokens": usage_data["total_tokens"] if isinstance(usage_data, dict) else usage_data.total_tokens # type: ignore
                },
                processing_time=processing_time
            )
            
            # 如果有工具调用，添加到响应中
            if tool_calls:
                # 转换工具调用格式
                converted_tool_calls: list[ToolCall] = []
    
                for tool_call_item in tool_calls:
                    # 确保 function 字段是字典格式
                    function_dict = {}
                    
                    if hasattr(tool_call_item, 'function'):
                        func_data = tool_call_item.function
                        
                        # 处理 Function 对象
                        if hasattr(func_data, 'name') and hasattr(func_data, 'arguments'):
                            function_dict = {
                                "name": func_data.name,
                                "arguments": (
                                    func_data.arguments 
                                    if isinstance(func_data.arguments, str)
                                    else json.dumps(func_data.arguments)
                                )
                            }
                        # 处理字典格式
                        elif isinstance(func_data, dict):
                            function_dict = {
                                "name": func_data.get('name', 'unknown'),
                                "arguments": (
                                    func_data.get('arguments', '{}')
                                    if isinstance(func_data.get('arguments'), str)
                                    else json.dumps(func_data.get('arguments', {}))
                                )
                            }
                    else:
                        # 处理其他格式
                        function_dict = {
                            "name": getattr(tool_call_item, 'tool_name', getattr(tool_call_item, 'name', 'unknown')),
                            "arguments": (
                                getattr(tool_call_item, 'arguments', '{}')
                                if isinstance(getattr(tool_call_item, 'arguments', '{}'), str)
                                else json.dumps(getattr(tool_call_item, 'parameters', getattr(tool_call_item, 'arguments', {})))
                            )
                        }
                    
                    # 创建 ToolCall
                    converted_tool_calls.append(
                        ToolCall(
                            id=getattr(tool_call_item, 'id', str(uuid.uuid4())),
                            type=getattr(tool_call_item, 'type', 'function'),
                            function=function_dict  # type: ignore
                        )
                    )
                
                chat_response.tool_calls = converted_tool_calls
            
            return chat_response
        
        else:
            # 响应格式不支持
            raise HTTPException(400, detail="不支持的响应格式")

    except ValidationError as e:
        raise HTTPException(400, detail=str(e))
    except TimeoutError:
        raise HTTPException(408, detail="请求超时")
    except Exception as e:
        logger.exception("聊天生成失败")
        raise HTTPException(500, detail={
            "error": str(e),
            "type": e.__class__.__name__
        })


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