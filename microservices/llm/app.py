"""LLM microservice application.

This module provides a FastAPI application that exposes HTTP endpoints for interacting
with various LLM providers. It supports text generation and chat completion.

该模块提供 FastAPI 微服务应用程序，用于与各种 LLM 提供者交互。它支持文本生成和聊天完成。

"""

import sys
import os
import time
import uuid
import json
import signal
import subprocess
import platform
import atexit
from datetime import datetime
from typing import Any
from loguru import logger
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from contextlib import asynccontextmanager

# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from microservices.llm.llm_service import LLMService
from core.config import settings
from core.log.logger import setup_logging

# 初始化日志
setup_logging(service_name="llm")

# 创建LLM服务实例
llm_service = LLMService()

# 定义 lifespan 上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager. //应用程序生命周期上下文管理器"""
    # 启动事件
    start_time = time.time()
    logger.info(f"Initializing LLM service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")    
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Process ID: {os.getpid()}")
    
    # 初始化LLM服务
    try:
        await llm_service.initialize()
        logger.info(f"LLM service started successfully, elapsed time: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to initialize LLM service: {e}")
        # 在生产环境中，可能需要在这里退出应用程序
        current_env = os.getenv("KBOT_ENV")
        if current_env == "production":
            sys.exit(1)
    
    yield  # 服务运行期间
    
    # 关闭事件
    logger.info(f"Closing LLM service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    shutdown_start = time.time()
    
    try:
        await llm_service.shutdown()
        logger.info("Successfully closed LLM service")
    except Exception as e:
        logger.error(f"Error closing LLM service: {e}")
    
    logger.info(f"LLM service closed in {time.time() - shutdown_start:.2f} seconds")
    logger.info(f"Total service runtime: {time.time() - start_time:.2f} seconds")

# 创建FastAPI应用
app = FastAPI(
    title="LLM service",
    description="Provides text generation and chat completion services using various LLM providers.",
    version=settings["app"]["version"],
    lifespan=lifespan,
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatResponse(BaseModel):
    """Response model for chat (OpenAI compatible). //聊天响应模型(兼容OpenAI)"""

    id: str = Field(default_factory=lambda: f"sse-{uuid.uuid4()}", 
                   description="Unique identifier for the chat completion")
    object: str = Field("chat.completion", 
                       description="The object type, always 'chat.completion'")
    created: int = Field(default_factory=lambda: int(time.time()), 
                        description="Unix timestamp of when the response was created")
    model: str = Field(..., description="The model used for the completion")
    choices: list[dict[str, Any]] = Field(...,
        description="list of chat completion choices containing messages")
    usage: dict[str, int] = Field(...,
        description="Token usage statistics including prompt_tokens, completion_tokens and total_tokens")
    processing_time: float = Field(..., 
                                 description="Processing time in seconds (custom field)")


class ChatRequest(BaseModel):
    """Request model for chat. //聊天请求模型"""

    model_unique_name: str = Field(..., description="Specific model id to use")
    messages: list[dict[str, str]] | str = Field(..., description="list of chat messages")
    max_tokens: int | None = Field(None, description="Maximum number of tokens to generate")
    temperature: float | None = Field(
        None, description="Sampling temperature (0.0-1.0, lower is more deterministic)"
    )
    stream: bool = Field(False, description="Whether to stream the response")
    timeout: int | None = Field(None, description="Timeout in seconds")
    top_p: float | None = Field(None, description="Top-p sampling parameter")
    frequency_penalty: float | None = Field(None, description="Frequency penalty")
    presence_penalty: float | None = Field(None, description="Presence penalty")


# 依赖项：获取嵌入服务实例
def get_llm_service():
    return llm_service

@app.get("/health", response_model=dict, tags=["LLM"])
async def health() -> dict[str, Any]:
    """Health check endpoint. //微服务接口健康检查
    Returns:
        Loaded models count. //已加载的模型数量
    """
    # 获取已加载的模型信息
    loaded_models = {}
    if llm_service._initialized and hasattr(llm_service._model_pool, '_models'):
        loaded_models = llm_service._model_pool._models
    
    # 返回已加载模型数量
    return {
        "status": "ok",
        "loaded_models_count": len(loaded_models),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/v1/chat/completions", response_model=None, tags=["LLM"])
async def chat(
    request: ChatRequest,
    llm_service: LLMService = Depends(get_llm_service)
) -> ChatResponse | StreamingResponse:
    """Generate chat response //生成聊天响应
    
    - **model_unique_name**: 要使用的模型ID
    - **messages**: 聊天消息列表
    - **max_tokens**: 要生成的最大令牌数（可选）
    - **temperature**: 采样温度（0.0-1.0，越低越确定）
    - **stream**: 是否流式返回响应
    - **timeout**: 超时时间（秒）
    - **top_p**: Top-p采样参数
    - **frequency_penalty**: 频率惩罚
    - **presence_penalty**: 存在惩罚
    
    返回:
    - 流式模式: 标准OpenAI SSE格式
    - 非流式模式: 包含消息和处理时间的JSON对象
    """
    start_time = time.time()
    response_id = f"chatcmpl-{uuid.uuid4()}"  # OpenAI格式的ID
    created_time = int(time.time())
    model_name = request.model_unique_name
    
    logger.info(f"Generating chat response using model {request.model_unique_name}")
    
    try:
        if request.stream:
            async def generate():
                try:
                    # 获取流式响应
                    chunk_stream = await llm_service.chat(
                        model_unique_name=request.model_unique_name,
                        messages=request.messages,
                        stream=True,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        timeout=request.timeout,
                        top_p=request.top_p,
                        frequency_penalty=request.frequency_penalty,
                        presence_penalty=request.presence_penalty
                    )
                    
                    # 初始化usage数据
                    usage_data = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                    
                    async for content in chunk_stream: # type: ignore
                        # 如果是usage数据，更新usage_data
                        if content.startswith("\n\n=== USAGE ==="):
                            try:
                                usage_data.update(json.loads(content.replace("\n\n=== USAGE ===\n", "")))
                            except json.JSONDecodeError:
                                logger.warning("Failed to parse usage data")
                            continue
                            
                        # 标准OpenAI SSE格式
                        chunk_data = {
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
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    
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
                        }],
                        "usage": usage_data
                    }
                    yield f"data: {json.dumps(end_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    
                except Exception as e:
                    logger.error(f"Stream error: {str(e)}")
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
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
            
        else:
            # 非流式响应
            response = await llm_service.chat(
                model_unique_name=request.model_unique_name,
                messages=request.messages,
                stream=False,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                timeout=request.timeout,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Chat completion took {processing_time:.2f}s")
            
            # 获取usage数据
            usage_data = response.get("usage", { # type: ignore
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })
            
            return ChatResponse(
                id=response_id,
                object="chat.completion",
                created=created_time,
                model=model_name,
                choices=[{
                    "message": {
                        "role": "assistant",
                        "content": response["content"] # type: ignore
                    },
                    "finish_reason": "stop",
                    "index": 0
                }],
                usage={
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0)
                },
                processing_time=processing_time
            )

    except ValidationError as e:
        raise HTTPException(400, detail=str(e))
    except TimeoutError:
        raise HTTPException(408, detail="Request timeout")
    except Exception as e:
        logger.exception("Chat completion failed")
        raise HTTPException(500, detail={
            "error": str(e),
            "type": e.__class__.__name__
        })


# 全局变量，用于存储微服务进程
llm_service_process = None

def start_llm_service():
    """Start the llm microservice as an independent process."""
    logger.info("Start the llm microservice as an independent process.")
    llm_service_path = os.path.abspath(__file__)
    
    # 启动llm微服务，使用环境变量中的端口并设置为独立模式
    process = subprocess.Popen(
        [sys.executable, llm_service_path],
        env={**os.environ, "LLM_SERVICE_STANDALONE": "1"}
    )
    return process

def shutdown_llm_service():
    """Terminate the llm microservice process."""
    global llm_service_process
    if llm_service_process:
        logger.info("Terminating the llm microservice process...")
        try:
            llm_service_process.terminate()
            llm_service_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("The llm microservice process failed to terminate properly; forcing shutdown...")
            llm_service_process.kill()
        llm_service_process = None

def signal_handler(sig, frame):
    """Handling termination signal."""
    logger.info(f"Signal received: {sig}, shutting down....")
    shutdown_llm_service()
    sys.exit(0)

# 注册退出处理程序，确保在应用程序退出时关闭微服务
atexit.register(shutdown_llm_service)

if __name__ == "__main__":
    import uvicorn
    # 从环境变量获取主机和端口，如果没有设置，则使用默认值
    host = os.environ.get("KBOT_LLM_HOST", "0.0.0.0")
    port = int(os.environ.get("KBOT_LLM_PORT", 8002))
    
    # 如果是作为独立进程启动，则注册信号处理器
    if os.environ.get("LLM_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Started the llm microservice, listening on {host}:{port}")
    uvicorn.run(app, host=host, port=port)