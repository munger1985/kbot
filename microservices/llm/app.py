"""LLM microservice application.

This module provides a FastAPI application that exposes HTTP endpoints for interacting
with various LLM providers. It supports text generation and chat completion.

该模块提供 FastAPI 微服务应用程序，用于与各种 LLM 提供者交互。它支持文本生成和聊天完成。

"""

import os
import sys
import signal
import json
import subprocess
import time
import atexit
import configparser
import uuid
import uvicorn
import socket
from dotenv import load_dotenv
from datetime import datetime
from typing import Any
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, ValidationError
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from nacos import NacosClient
from nacos_manager import nacos_manager # type: ignore
from logger_manager import LogManager, LogConfig # type: ignore
from llm_service import LLMService
from model import LLMProvider

# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
from dao.repositories.kbot_md_models_repo import KbotMdModelsRepository


# 加载环境变量配置
load_dotenv()
nacos_addr = os.getenv("NACOS_SERVER_ADDR") # Nacos服务器地址
nacos_namespace = os.getenv("NACOS_NAMESPACE") or "public" # Nacos命名空间
nacos_group = os.getenv("NACOS_GROUP") or "DEV_GROUP" # Nacos分组
nacos_username = os.getenv("NACOS_USERNAME") # Nacos账号名称
nacos_password = os.getenv("NACOS_PASSWORD") # Nacos账号密码

try:
    # 从 nacos 获取 llm 服务配置
    config_parser = configparser.ConfigParser()
    nacos_config = nacos_manager.get_config("llm", nacos_group)
    config_parser.read_string(f"[{nacos_group}]\n{nacos_config}")
    service_name = config_parser.get(nacos_group, "service_name") or "llm-service" # 全局微服务名称
    service_version = config_parser.get(nacos_group, "service_version") or "1.0.0" # 微服务版本
    service_host = config_parser.get(nacos_group, "service_host") or "0.0.0.0" # 微服务地址
    service_port = int(config_parser.get(nacos_group, "service_port")) or 9202 # 微服务通信端口
except Exception as e:
    # 如果从 nacos 获取 llm 服务配置失败，则使用默认配置
    logger.warning("Failed to get llm service config from nacos: {}".format(e))
    service_name = "llm-service"
    service_version = "1.0.0"
    service_host = "0.0.0.0"
    service_port = 9202

# Nacos 服务注册
def register_service():
    client = NacosClient(
        server_addresses=nacos_addr,
        namespace=nacos_namespace
        # username='nacos',
        # password='nacos'
        )
    client.add_naming_instance(
        service_name=service_name,
        group_name=nacos_group,
        ip=service_host,
        port=service_port,
        ephemeral=True,
        healthy=True
    )
    # nacos 心跳发送器
    while True:
        if signal.SIGINT or signal.SIGTERM:
            break
        try:
            # 健康检查：检测服务端口是否存活
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((service_host, service_port))
            is_healthy = (result == 0)
            sock.close()

            # 更新实例健康状态
            client.send_heartbeat(
                service_name=service_name,
                group_name=nacos_group,
                ip=service_host,
                port=service_port
            )
            
            logger.info(f"Heartbeat sent, healthy: {is_healthy}")
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            break
        
        time.sleep(10)  # 间隔需小于Nacos心跳超时时间（默认15秒）

# 创建LLM服务实例
llm_service = LLMService()

# 定义 lifespan 上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager. //应用程序生命周期上下文管理器"""
    # 通过 nacos_manager 获取logger配置
    try:
        log_config = nacos_manager.get_config("logger", nacos_group)
        config_parser.read_string(f"[{nacos_group}]\n{log_config}")
        log_dir = config_parser.get(nacos_group, "dir") or "logs/"
        log_level = config_parser.get(nacos_group, "level") or "DEBUG"
        rotation = config_parser.get(nacos_group, "rotation") or "10 MB"
        retention = config_parser.get(nacos_group, "retention") or "20 days"
        
    except Exception as e:
        # 如果获取 logger 配置失败，则使用默认配置
        logger.warning(f"Failed to get logger config from nacos: {str(e)}")
        log_dir = "logs/"
        log_level = "DEBUG"
        rotation = "10 MB"
        retention = "10 days"
    
    # 初始化日志
    conf = LogConfig(service_name=service_name, log_dir=log_dir, level=log_level, rotation=rotation, retention=retention)
    LogManager(conf).setup()
    
    # 启动事件
    start_time = time.time()
    logger.info(f"Initializing LLM service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")    
    logger.info(f"Process ID: {os.getpid()}")

    
    
    # 初始化LLM服务
    try:
        await llm_service.initialize()
        logger.info(f"LLM service started successfully, elapsed time: {time.time() - start_time:.2f} seconds")

        # 注册服务到 Nacos
        register_service()
        logger.info("LLM service registered to Nacos.")

    except Exception as e:
        logger.exception(f"Failed to initialize LLM service: {e}")
        # 在生产环境中，可能需要在这里退出应用程序
        current_env = nacos_namespace
        if current_env == "prod":
            sys.exit(1)
    
    yield  # 服务运行期间
    
    # 关闭事件
    logger.info(f"Closing LLM service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    shutdown_start = time.time()
    
    try:
        await llm_service.shutdown()
        logger.info("Successfully closed LLM service")
    except Exception as e:
        logger.exception(f"Error closing LLM service: {e}")
    
    logger.info(f"LLM service closed in {time.time() - shutdown_start:.2f} seconds")
    logger.info(f"Total service runtime: {time.time() - start_time:.2f} seconds")

# 创建FastAPI应用
app = FastAPI(
    title="LLM service",
    description="Provides text generation and chat completion services using various LLM providers.",
    version=service_version,
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
    provider = await KbotMdModelsRepository().get_provider_by_unique_name(model_name)
    logger.info(f"Generating chat response using model {request.model_unique_name}")
    
    try:
        # OpenAI streaming 格式响应
        if request.stream and provider == LLMProvider.OPENAI.value:
            async def generate_openai_sse():
                try:
                    # 获取流式响应
                    max_tokens = min(request.max_tokens, 4000) if request.max_tokens else 4000
                    chunk_stream = await llm_service.chat(
                        model_unique_name=request.model_unique_name,
                        messages=request.messages,
                        stream=True,
                        max_tokens=max_tokens,
                        temperature=request.temperature,
                        timeout=request.timeout,
                        top_p=request.top_p,
                        frequency_penalty=request.frequency_penalty,
                        presence_penalty=request.presence_penalty
                    )
                    
                    async for chunk in chunk_stream: # type: ignore
                        # ChatCompletionChunk
                        content = chunk.choices[0].delta.content
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
                    logger.exception(f"Stream error: {str(e)}")
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
                    "Connection": "keep-alive"
                }
            )

        elif request.stream and provider == LLMProvider.OCI.value:
            async def generate_oci_sse():
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
                    
                    async for chunk in chunk_stream: # type: ignore                        
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
                    logger.exception(f"Stream error: {str(e)}")
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
                generate_oci_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
            
        elif not request.stream:
            # OpenAI 非流式响应
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
            content = None
            usage_data = None
            # 获取响应内容和usage数据
            if provider == LLMProvider.OPENAI.value:
                content = response.choices[0].message.content # type: ignore
                usage_data = response.usage # type: ignore

            elif provider == LLMProvider.OCI.value:
                content = response.data.chat_response.text # type: ignore
                usage_data = response.data.chat_response.usage # type: ignore

            else:
                logger.warning(f"Unsupported provider: {provider}")

            return ChatResponse(
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
        
        else:
            # 响应格式不支持
            raise HTTPException(400, detail="Response format not supported")

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
    """Start the LLM microservice as an independent process."""
    try:
        logger.info("Starting LLM microservice as independent process...")
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
            raise RuntimeError(f"Failed to start LLM service: {stderr}")
            
        logger.success(f"LLM service started successfully with PID {process.pid}")
        return process
        
    except Exception as e:
        logger.exception(f"Error starting LLM service: {str(e)}")
        raise

def shutdown_llm_service():
    """Terminate the LLM microservice process."""
    global llm_service_process
    if llm_service_process:
        logger.info("Terminating the LLM microservice process...")
        try:
            llm_service_process.terminate()
            llm_service_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("The LLM microservice process failed to terminate properly; forcing shutdown...")
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
    # 如果是作为独立进程启动，则注册信号处理器
    if os.environ.get("LLM_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Started the LLM microservice, listening on {service_host}:{service_port}")
    uvicorn.run(app, host=service_host, port=service_port)