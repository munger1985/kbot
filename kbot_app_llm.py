"""LLM微服务应用程序。

该模块提供了一个FastAPI应用程序，用于与各种LLM提供者交互。它支持文本生成、聊天完成和MCP工具调用功能。
"""

import os
import sys
import signal
import json
import subprocess
import time
import atexit
import uuid
import uvicorn
from dotenv import load_dotenv
from datetime import datetime
from typing import Any
from contextlib import asynccontextmanager
from fastapi_offline import FastAPIOffline
from pydantic import ValidationError
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

from core.config.settings import get_settings, get_app_config
from core.logger_manager import LogConfig, LogManager
from core.dictionary import LLMProvider
from microservices.llm.llm_service import LLMService
from microservices.llm.schema import *



# 加载环境变量配置
load_dotenv()

# 获取模型配置
config = get_settings()
service_name = config.llm.service_name
service_version = config.llm.service_version
service_host = config.llm.service_host
service_port = config.llm.service_port

# 获取应用配置
app_config = get_app_config()
debug = app_config.debug
log_dir = app_config.log.dir
log_level = app_config.log.level
rotation = app_config.log.rotation
retention = app_config.log.retention

# 创建LLM服务实例
llm_service = LLMService()

# ==================== 应用生命周期 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期上下文管理器"""
    
    # 初始化日志
    conf = LogConfig(service_name=service_name, log_dir=log_dir, level=log_level, rotation=rotation, retention=retention)
    LogManager(conf).setup()
    
    # 启动事件
    start_time = time.time()
    logger.info(f"正在初始化LLM服务，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")    
    logger.info(f"进程ID: {os.getpid()}")

    # 初始化LLM服务
    try:
        await llm_service.initialize()
        await llm_service.warmup()
        logger.info(f"LLM服务启动成功，耗时: {time.time() - start_time:.2f} 秒")

    except Exception as e:
        logger.exception(f"初始化LLM服务失败: {e}")
        # 在生产环境中，可能需要在这里退出应用程序
        if not debug:
            sys.exit(1)
    
    yield  # 服务运行期间
    
    # 关闭事件
    logger.info(f"正在关闭LLM服务，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    shutdown_start = time.time()
    
    try:
        await llm_service.shutdown()
        logger.info("LLM服务关闭成功")
    except Exception as e:
        logger.exception(f"关闭LLM服务时出错: {e}")
    
    logger.info(f"LLM服务关闭耗时: {time.time() - shutdown_start:.2f} 秒")
    logger.info(f"服务总运行时间: {time.time() - start_time:.2f} 秒")

# ==================== FastAPI应用创建 ====================

# 创建FastAPI应用
app = FastAPIOffline(
    title="LLM 微服务",
    description="提供使用各种LLM提供者的文本生成、聊天完成和MCP工具调用服务",
    version=service_version,
    lifespan=lifespan,
    docs_url="/docs" if debug else None,
    redoc_url="/redoc" if debug else None
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== API路由 ====================

# 依赖项：获取LLM服务实例
def get_llm_service():
    return llm_service

@app.get("/health", response_model=dict, tags=["LLM"], summary="LLM服务健康检查接口")
async def health() -> dict[str, Any]:
    """微服务健康检查接口。
    
    Returns:
    - **dict**: 包含服务状态、已加载模型数量和时间戳的响应数据
    ```
        {
        "status": "ok",
        "loaded_models_count": len(loaded_models),
        "timestamp": datetime.now().isoformat()
        }
    ```
    """

    # 获取已加载的模型信息
    loaded_models = {}
    if llm_service._initialized and hasattr(llm_service._model_pool, '_models'):
        loaded_models = llm_service._model_pool._models
    
    return {
        "status": "ok",
        "loaded_models_count": len(loaded_models),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/load", response_model=dict, tags=["LLM"], summary="加载或卸载模型")
async def load_model(request: ToggleModelRequest) -> dict:
    """通过模型ID加载模型到内存中。"""
    model_name = llm_service._model_pool._model_names.get(request.model_id, str(request.model_id))
    try:
        if request.operation == "load":
            logger.info(f"接收到指令：加载模型 {model_name}")
            success = await llm_service.load_model(request.model_id)
        else:
            logger.info(f"接收到指令：卸载模型 {model_name}")
            success = await llm_service.unload_model(request.model_id)
        if not success:
            raise HTTPException(status_code=500, detail=f"模型 {model_name} 操作失败")
        return {"status": "success", "model_name:": model_name}
    except Exception as e:
        logger.exception(f"操作模型 {model_name} 时发生错误: {e}")
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
        "model": model_name,
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


# 全局变量，用于存储微服务进程
llm_service_process = None

def start_llm_service():
    """启动LLM微服务作为独立进程"""
    try:
        logger.info("正在启动LLM微服务作为独立进程...")
        llm_service_path = os.path.abspath(__file__)
        
        process = subprocess.Popen(
            [sys.executable, llm_service_path],
            env={**os.environ, "LLM_SERVICE_STANDALONE": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 检查进程是否成功启动
        if process.poll() is not None:
            stderr = process.stderr.read().decode('utf-8') if process.stderr else ""
            raise RuntimeError(f"启动LLM服务失败: {stderr}")
            
        logger.success(f"LLM服务启动成功，进程ID: {process.pid}")
        return process
        
    except Exception as e:
        logger.exception(f"启动LLM服务时出错: {str(e)}")
        raise

def shutdown_llm_service():
    """终止LLM微服务进程"""
    global llm_service_process
    if llm_service_process:
        logger.info("正在终止LLM微服务进程...")
        try:
            llm_service_process.terminate()
            llm_service_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("LLM微服务进程未能正常终止; 强制关闭...")
            llm_service_process.kill()
        llm_service_process = None

def signal_handler(sig, frame):
    """处理终止信号"""
    logger.info(f"收到信号: {sig}, 正在关闭...")
    shutdown_llm_service()
    sys.exit(0)

# 注册退出处理程序，确保在应用程序退出时关闭微服务
atexit.register(shutdown_llm_service)

if __name__ == "__main__":
    # 如果是作为独立进程启动，则注册信号处理器
    if os.environ.get("LLM_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"已启动LLM微服务，监听地址: {service_host}:{service_port}")
    logger.info("MCP工具调用支持已启用")
    uvicorn.run(app, host=service_host, port=service_port)