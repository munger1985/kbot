"""VLM 微服务应用程序。

该模块提供了一个 FastAPI 应用程序，用于暴露与各种 VLM 提供商交互的 HTTP 端点。它支持文本 VLM。
"""

import os
import sys
import signal
import uuid
import time
import json
import uvicorn
import subprocess
import atexit
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
from vlm_service import VLMService
from ms_core import nacos_manager, ConfigManager, LogManager, LogConfig


# 添加对 PIL.Image.Image 的 Pydantic 支持
def get_pydantic_core_schema(
    cls: Type[Image.Image],
    handler: Any,
) -> core_schema.CoreSchema:
    """
    为 PIL.Image.Image 实现 __get_pydantic_core_schema__。
    这允许 Pydantic 正确处理 PIL.Image.Image 类型。
    """
    return core_schema.no_info_after_validator_function(
        lambda x: x,
        core_schema.any_schema(),
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda img: img.filename if hasattr(img, 'filename') else "PIL.Image"
        ),
    )

# 为 PIL.Image.Image 注册 schema
Image.Image.__get_pydantic_core_schema__ = get_pydantic_core_schema # type: ignore

# 加载环境变量配置
load_dotenv()

# 从 nacos 获取 vlm 服务配置
config = ConfigManager.get_model_config()
service_name = config.vlm.service_name
service_version = config.vlm.service_version
service_host = config.vlm.service_host
service_port = config.vlm.service_port

# 创建VLM服务实例
vlm_service = VLMService()

# 定义 lifespan 上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期上下文管理器"""
    # 通过 nacos_manager 获取 logger 配置
    log_config = ConfigManager.get_app_config()  
    log_dir = log_config.kbot.log.dir
    log_level = log_config.kbot.log.level
    rotation = log_config.kbot.log.rotation
    retention = log_config.kbot.log.retention
    
    # 初始化日志
    conf = LogConfig(service_name=service_name, log_dir=log_dir, level=log_level, rotation=rotation, retention=retention)
    LogManager(conf).setup()
    
    # 启动事件
    start_time = time.time()
    logger.info(f"正在初始化 VLM 服务，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...") 
    logger.info(f"进程ID: {os.getpid()}")
    
    # 初始化VLM服务
    try:
        await vlm_service.initialize()
        await vlm_service.warmup()
        logger.info(f"VLM 服务启动成功，耗时: {time.time() - start_time:.2f} 秒")

        # 注册服务到 Nacos
        nacos_manager.register_service(service_name=service_name, service_host=service_host, service_port=service_port)
        logger.info("VLM 服务已注册到 Nacos")

    except Exception as e:
        logger.error(f"VLM 服务初始化失败: {e}")
        # 在生产环境中，可能需要在这里退出应用程序
        current_env = os.getenv("NACOS_GROUP", "dev")
        if current_env == "prod":
            sys.exit(1)
    
    yield  # 服务运行期间
    
    # 关闭事件
    logger.info(f"正在关闭 VLM 服务，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    shutdown_start = time.time()
    
    try:
        await vlm_service.shutdown()
        logger.info("VLM 服务已关闭")
    except Exception as e:
        logger.error(f"VLM 服务关闭失败: {e}")
    
    logger.info(f"VLM 服务关闭成功，耗时: {time.time() - shutdown_start:.2f} 秒")
    logger.info(f"总运行时间: {time.time() - start_time:.2f} 秒")

# 创建 FastAPI 应用
app = FastAPI(
    title="VLM 服务",
    description="提供文本 VLM 服务，将文本转换为向量表示",
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
    """VLM推理请求模型"""

    model_unique_name: str = Field(..., description="要使用的特定模型ID")
    messages: list[dict[str, Any]] = Field(..., description="消息列表")
    max_tokens: int | None = Field(None, description="要生成的最大令牌数")
    temperature: float | None = Field(
        None, description="采样温度 (0.0-1.0，越低越确定)"
    )
    stream: bool = Field(False, description="是否流式返回响应")
    timeout: int | None = Field(None, description="超时时间（秒）")
    top_p: float | None = Field(None, description="Top-p采样参数")
    frequency_penalty: float | None = Field(None, description="频率惩罚")
    presence_penalty: float | None = Field(None, description="存在惩罚")

class ToggleModelRequest(BaseModel):
    """启用或禁用模型请求表单。"""
    model_unique_name: str = Field(..., description="模型唯一标识符")
    operation: str = Field(..., description="操作类型，'load' 或 'unload'")
    
# 定义响应模型
class VLMResponse(BaseModel):
    """VLM推理响应模型(兼容OpenAI)"""

    id: str = Field(default_factory=lambda: f"sse-{uuid.uuid4()}", 
                   description="完成的唯一标识符")
    object: str = Field("chat.completion", 
                       description="对象类型，始终为 'chat.completion'")
    created: int = Field(default_factory=lambda: int(time.time()), 
                        description="响应创建时的Unix时间戳")
    model: str = Field(..., description="用于完成的模型")
    choices: list[dict[str, Any]] = Field(...,
        description="包含消息的完成选择列表")
    usage: dict[str, int] = Field(...,
        description="令牌使用统计，包括 prompt_tokens、completion_tokens 和 total_tokens")
    processing_time: float = Field(..., 
                                 description="处理时间（秒）（自定义字段）")


# 依赖项：获取VLM服务实例
def get_vlm_service():
    return vlm_service

@app.get("/health", response_model=dict, tags=["VLM"])
async def health() -> dict[str, Any]:
    """微服务接口健康检查
    
    返回:
        已加载的模型数量
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

@app.post("/load", response_model=dict, tags=["VLM"])
async def load_model(request: ToggleModelRequest) -> dict:
    """通过模型ID加载模型到内存中。
    
    Args:
        request: 启用或禁用模型请求表单，包含模型唯一名称和操作类型
        
    Returns:
        dict: 包含操作状态和模型ID的响应数据
        
    Raises:
        HTTPException: 当模型加载失败时抛出500错误
    """
    try:
        if request.operation == "load":
            logger.info(f"接收到指令：加载模型 {request.model_unique_name}")
            success = await vlm_service.load_model(request.model_unique_name)
        else:
            logger.info(f"接收到指令：卸载模型 {request.model_unique_name}")
            success = await vlm_service.unload_model(request.model_unique_name)
        if not success:
            raise HTTPException(status_code=500, detail=f"模型 {request.model_unique_name} 操作失败")
        return {"status": "success", "model_unique_name": request.model_unique_name}
    except Exception as e:
        logger.exception(f"操作模型 {request.model_unique_name} 时发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
      
@app.post("/v1/inference", response_model=VLMResponse, tags=["VLM"])
async def inference(
    request: VLMRequest,
    vlm_service: VLMService = Depends(get_vlm_service)
) -> VLMResponse | StreamingResponse:
    """生成VLM响应
    
    参数:
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
    
    logger.info(f"正在使用模型 {request.model_unique_name} 生成 VLM 响应")
    
    try:
        if request.stream:
            async def generate():
                try:
                    # 获取流式响应
                    chunk_stream = await vlm_service.inference(
                        model_id=request.model_unique_name,
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
                                logger.warning("解析使用数据失败")
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
                    logger.error(f"流处理错误: {str(e)}")
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
                model_id=request.model_unique_name,
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
            logger.info(f"VLM 完成耗时 {processing_time:.2f} 秒")
            
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
        raise HTTPException(408, detail="请求超时")
    except Exception as e:
        logger.exception("VLM 完成失败")
        raise HTTPException(500, detail={
            "error": str(e),
            "type": e.__class__.__name__
        })



# 全局变量，用于存储微服务进程
vlm_service_process = None

def start_vlm_service():
    """以独立进程方式启动 VLM 微服务"""
    try:
        logger.info("正在以独立进程方式启动 VLM 微服务")
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
            raise RuntimeError(f"启动 VLM 服务失败: {stderr}")
            
        logger.success(f"VLM 服务启动成功，进程ID: {process.pid}")
        return process
        
    except Exception as e:
        logger.exception(f"启动 VLM 服务时出错: {str(e)}")
        raise

def shutdown_vlm_service():
    """终止 VLM 微服务进程"""
    global vlm_service_process
    if vlm_service_process:
        logger.info("正在终止 VLM 微服务进程...")
        try:
            vlm_service_process.terminate()
            vlm_service_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("VLM 微服务进程未能正常终止; 强制关闭...")
            vlm_service_process.kill()
        vlm_service_process = None


def signal_handler(sig, frame):
    """处理终止信号"""
    logger.info(f"收到信号: {sig}, 正在关闭...")
    shutdown_vlm_service()
    sys.exit(0)

# 注册退出处理程序，确保在应用程序退出时关闭微服务
atexit.register(shutdown_vlm_service)

if __name__ == "__main__":
    # 如果是作为独立进程启动，则注册信号处理器
    if os.environ.get("vlm_service_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"已启动 VLM 微服务，监听地址: {service_host}:{service_port}")
    uvicorn.run(app, host=service_host, port=service_port)