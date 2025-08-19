"""VLM microservice application.

This module provides a FastAPI application that exposes HTTP endpoints for interacting
with various VLM providers. It supports text VLM.

该模块提供 FastAPI 微服务应用程序，用于公开与各种VLM提供者交互的 HTTP 端点。它支持文本VLM。

"""

import os
import sys
import signal
import uuid
import time
import json
import configparser
import uvicorn
import subprocess
import atexit
import socket
from datetime import datetime
from PIL import Image
from typing import Any, Type
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, ValidationError
from pydantic_core import core_schema
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from nacos import NacosClient
from vlm_service import VLMService
from ms_core import nacos_manager, ModelConfig, AppConfig, load_config, LogManager, LogConfig


# Add Pydantic support for PIL.Image.Image
def get_pydantic_core_schema(
    cls: Type[Image.Image],
    handler: Any,
) -> core_schema.CoreSchema:
    """
    Implement __get_pydantic_core_schema__ for PIL.Image.Image.
    This allows Pydantic to properly handle PIL.Image.Image types.
    """
    return core_schema.no_info_after_validator_function(
        lambda x: x,
        core_schema.any_schema(),
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda img: img.filename if hasattr(img, 'filename') else "PIL.Image"
        ),
    )

# Register the schema for PIL.Image.Image
Image.Image.__get_pydantic_core_schema__ = get_pydantic_core_schema # type: ignore

# 加载环境变量配置
load_dotenv()

try:
    # 从 nacos 获取 vlm 服务配置
    config = load_config("model_config")
    if not isinstance(config, ModelConfig):
        raise ValueError
    service_name = config.vlm.service_name or "vlm-service" # 全局微服务名称
    service_version = config.vlm.service_version or "1.0.0" # 微服务版本
    service_host = config.vlm.service_host or "0.0.0.0" # 微服务地址
    service_port = config.vlm.service_port or 9204 # 微服务通信端口
except Exception as e:
    # 如果从 nacos 获取 vlm 服务配置失败，则使用默认配置
    logger.warning(f"Failed to get vlm service config from nacos: {e}")
    service_name = "vlm-service"
    service_version = "1.0.0"
    service_host = "0.0.0.0"
    service_port = 9204

# 创建VLM服务实例
vlm_service = VLMService()

# 定义 lifespan 上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager. //应用程序生命周期上下文管理器"""
    # 通过 nacos_manager 获取 logger 配置
    try:
        log_config = load_config("app_config")
        if not isinstance(log_config, AppConfig):
            raise ValueError
        
        log_dir = log_config.kbot.log.dir or "logs/"
        log_level = log_config.kbot.log.level or "DEBUG"
        rotation = log_config.kbot.log.rotation or "10 MB"
        retention = log_config.kbot.log.retention or "20 days"
        
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
    logger.info(f"Initializing VLM service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...") 
    logger.info(f"Process ID: {os.getpid()}")
    
    # 初始化VLM服务
    try:
        await vlm_service.initialize()
        logger.info(f"VLM service started successfully, elapsed time: {time.time() - start_time:.2f} seconds")

        # 注册服务到 Nacos
        nacos_manager.register_service(service_name=service_name, service_host=service_host, service_port=service_port)
        logger.info("VLM service registered to Nacos.")

    except Exception as e:
        logger.error(f"VLM service initialization failed: {e}")
        # 在生产环境中，可能需要在这里退出应用程序
        current_env = os.getenv("NACOS_GROUP", "dev")
        if current_env == "prod":
            sys.exit(1)
    
    yield  # 服务运行期间
    
    # 关闭事件
    logger.info(f"Closing VLM service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    shutdown_start = time.time()
    
    try:
        await vlm_service.shutdown()
        logger.info("VLM service is closed.")
    except Exception as e:
        logger.error(f"VLM service shutdown failed: {e}")
    
    logger.info(f"VLM service closed successfully, elapsed time: {time.time() - shutdown_start:.2f} seconds")
    logger.info(f"Total running time: {time.time() - start_time:.2f} seconds")

# 创建 FastAPI 应用
app = FastAPI(
    title="VLM service",
    description="Provides text VLM services to convert text into vector representations.",
    version=service_version,
    lifespan=lifespan,
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求模型
class VLMRequest(BaseModel):
    """Request model for VLM inference. //VLM推理请求模型"""

    model_unique_name: str = Field(..., description="Specific model id to use")
    messages: list[dict[str, Any]] = Field(..., description="list of messages")
    max_tokens: int | None = Field(None, description="Maximum number of tokens to generate")
    temperature: float | None = Field(
        None, description="Sampling temperature (0.0-1.0, lower is more deterministic)"
    )
    stream: bool = Field(False, description="Whether to stream the response")
    timeout: int | None = Field(None, description="Timeout in seconds")
    top_p: float | None = Field(None, description="Top-p sampling parameter")
    frequency_penalty: float | None = Field(None, description="Frequency penalty")
    presence_penalty: float | None = Field(None, description="Presence penalty")


class VLMResponse(BaseModel):
    """Response model for VLM inference (OpenAI compatible). //VLM推理响应模型(兼容OpenAI)"""

    id: str = Field(default_factory=lambda: f"sse-{uuid.uuid4()}", 
                   description="Unique identifier for the completion")
    object: str = Field("chat.completion", 
                       description="The object type, always 'chat.completion'")
    created: int = Field(default_factory=lambda: int(time.time()), 
                        description="Unix timestamp of when the response was created")
    model: str = Field(..., description="The model used for the completion")
    choices: list[dict[str, Any]] = Field(...,
        description="list of completion choices containing messages")
    usage: dict[str, int] = Field(...,
        description="Token usage statistics including prompt_tokens, completion_tokens and total_tokens")
    processing_time: float = Field(..., 
                                 description="Processing time in seconds (custom field)")


# 依赖项：获取VLM服务实例
def get_vlm_service():
    return vlm_service

@app.get("/health", response_model=dict, tags=["VLM"])
async def health() -> dict[str, Any]:
    """Health check endpoint. //微服务接口健康检查
    Returns:
        Loaded models count. //已加载的模型数量
    """
    
    # 获取已加载的模型信息
    loaded_models = {}
    if vlm_service._initialized and hasattr(vlm_service._model_pool, '_models'):
        loaded_models = vlm_service._model_pool._models
    
    # 返回已加载模型数量
    return {
        "status": "ok",
        "loaded_models_count": len(loaded_models),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/v1/inference", response_model=VLMResponse, tags=["VLM"])
async def inference(
    request: VLMRequest,
    vlm_service: VLMService = Depends(get_vlm_service)
) -> VLMResponse | StreamingResponse:
    """Generate VLM response //生成VLM响应
    
    - **model_unique_name**: 要使用的模型ID
    - **messages**: 消息列表
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
    response_id = f"vlmcmpl-{uuid.uuid4()}"  # OpenAI格式的ID
    created_time = int(time.time())
    model_name = request.model_unique_name
    
    logger.info(f"Generating VLM response using model {request.model_unique_name}")
    
    try:
        if request.stream:
            async def generate():
                try:
                    # 获取流式响应
                    chunk_stream = await vlm_service.inference(
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
            response = await vlm_service.inference(
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
            logger.info(f"VLM completion took {processing_time:.2f}s")
            
            # 获取usage数据
            usage_data = response.get("usage", { # type: ignore
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })
            
            return VLMResponse(
                id=response_id,
                object="chat.completion",
                created=created_time,
                model=model_name,
                choices=[{
                    "message": {
                        "role": "assistant",
                        "content": response["choices"][0]["message"]["content"] # type: ignore
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
        logger.exception("VLM completion failed")
        raise HTTPException(500, detail={
            "error": str(e),
            "type": e.__class__.__name__
        })



# 全局变量，用于存储微服务进程
vlm_service_process = None

def start_vlm_service():
    """Start the VLM microservice as an independent process."""
    try:
        logger.info("Start the VLM microservice as an independent process.")
        vlm_service_path = os.path.abspath(__file__)
        
        process = subprocess.Popen(
            [sys.executable, vlm_service_path],
            env={**os.environ, "VLM_SERVICE_STANDALONE": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 检查进程是否成功启动
        if process.poll() is not None:
            stderr = process.stderr.read().decode('utf-8') if process.stderr else ""
            raise RuntimeError(f"Failed to start VLM service: {stderr}")
            
        logger.success(f"VLM service started successfully with PID {process.pid}")
        return process
        
    except Exception as e:
        logger.exception(f"Error starting VLM service: {str(e)}")
        raise

def shutdown_vlm_service():
    """Terminate the VLM microservice process."""
    global vlm_service_process
    if vlm_service_process:
        logger.info("Terminating the VLM microservice process...")
        try:
            vlm_service_process.terminate()
            vlm_service_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("The VLM microservice process failed to terminate properly; forcing shutdown...")
            vlm_service_process.kill()
        vlm_service_process = None


def signal_handler(sig, frame):
    """Handling termination signal."""
    logger.info(f"Signal received: {sig}, shutting down....")
    shutdown_vlm_service()
    sys.exit(0)

# 注册退出处理程序，确保在应用程序退出时关闭微服务
atexit.register(shutdown_vlm_service)

if __name__ == "__main__":
    # 如果是作为独立进程启动，则注册信号处理器
    if os.environ.get("vlm_service_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Started the VLM microservice, listening on {service_host}:{service_port}")
    uvicorn.run(app, host=service_host, port=service_port)