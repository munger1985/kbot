"""VLM 微服务应用程序。

本模块提供基于 FastAPI 的视觉语言模型 (VLM) 服务。
支持将图像与文本结合进行多模态推理，并兼容 OpenAI 风格的流式 (SSE) 与非流式响应。
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
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi_offline import FastAPIOffline
from loguru import logger
from pydantic import ValidationError
from pydantic_core import core_schema

from core.config.settings import get_vlm_config, get_app_config
from core.logger import LogConfig, LogManager
from core.middleware.log_middleware import log_requests
from microservices.vlm.vlm_service import VLMService
from microservices.vlm.schema import VLMRequest, VLMResponse, ToggleModelRequest

# --- Pydantic 对 PIL.Image 的增强支持 ---

def get_pydantic_core_schema(
    cls: Type[Image.Image],
    handler: Any,
) -> core_schema.CoreSchema:
    """为 PIL.Image.Image 实现 Pydantic 核心 Schema。

    允许 Pydantic 验证逻辑识别并处理 PIL 图像对象，增强多模态数据的类型安全。

    Args:
        cls: 目标类类型。
        handler: Pydantic 内部处理器。

    Returns:
        配置好的 CoreSchema 对象。
    """
    return core_schema.no_info_after_validator_function(
        lambda x: x,
        core_schema.any_schema(),
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda img: img.filename if hasattr(img, 'filename') else "PIL.Image"
        ),
    )

# 注册 PIL 图像对象的 Schema 处理器
Image.Image.__get_pydantic_core_schema__ = get_pydantic_core_schema  # type: ignore


# 加载环境变量
load_dotenv()

# 从配置中心获取服务配置
config = get_vlm_config()
SERVICE_NAME: str = config.service_name
SERVICE_VERSION: str = config.service_version
SERVICE_HOST: str = config.service_host
SERVICE_PORT: int = config.service_port

# 获取通用应用配置
app_config = get_app_config()
DEBUG: bool = app_config.debug
LOG_DIR: str = app_config.log.dir
LOG_LEVEL: str = app_config.log.level
LOG_ROTATION: str = app_config.log.rotation
LOG_RETENTION: str = app_config.log.retention

# 初始化 VLM 业务服务
vlm_service = VLMService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理 VLM 服务的生命周期。

    负责日志初始化、模型加载、预热以及服务关闭时的资源释放。
    """
    # 设置服务名称到 app.state（供中间件使用）
    app.state.service_name = SERVICE_NAME

    # 1. 初始化日志配置
    log_conf = LogConfig(
        service_name=SERVICE_NAME, 
        log_dir=LOG_DIR, 
        level=LOG_LEVEL, 
        rotation=LOG_ROTATION, 
        retention=LOG_RETENTION
    )
    LogManager(log_conf).setup()
    
    # 2. 启动初始化
    start_time = time.time()
    logger.info(f"正在启动 VLM 服务 | PID: {os.getpid()} | 时间: {datetime.now()}")
    
    try:
        await vlm_service.initialize()
        await vlm_service.warmup()
        logger.info(f"VLM 服务就绪 | 耗时: {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"VLM 服务初始化失败: {e}")
        if not app_config.debug:
            sys.exit(1)
    
    yield  # --- 运行状态 ---
    
    # 3. 资源清理
    logger.info("正在关闭 VLM 服务并释放资源...")
    try:
        await vlm_service.shutdown()
        logger.success("VLM 服务已安全退出")
    except Exception as e:
        logger.error(f"清理资源时发生异常: {e}")


# --- 应用实例配置 ---

app = FastAPIOffline(
    title="VLM 微服务",
    description="提供多模态视觉语言模型推理服务",
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

# 请求日志中间件
app.middleware("http")(log_requests)


def get_vlm_service() -> VLMService:
    """获取 VLM 服务实例依赖。"""
    return vlm_service


# --- API 端点实现 ---

@app.get("/health", response_model=dict, tags=["System"])
async def health_check() -> dict[str, Any]:
    """系统健康检查。"""
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
    """动态加载或卸载 VLM 模型。"""
    model_name = request.model_name
    try:
        if request.operation == "load":
            success = await vlm_service.load_model(model_name)
        else:
            success = await vlm_service.unload_model(model_name)
            
        if not success:
            raise HTTPException(status_code=500, detail=f"模型 {model_name} 操作未成功")
        return {"status": "success", "model_name": model_name}
    except Exception as e:
        logger.exception(f"模型管理异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/inference", response_model=VLMResponse, tags=["Inference"])
async def run_vlm_inference(
    request: VLMRequest,
    service: VLMService = Depends(get_vlm_service)
) -> VLMResponse | StreamingResponse:
    """执行 VLM 推理任务。

    支持单次返回和 SSE 流式返回。遵循 OpenAI 聊天补全接口规范。

    Args:
        request: 包含模型、消息流（含图片）和采样参数的请求体。
        service: VLM 业务逻辑服务。

    Returns:
        JSON 响应或 SSE 文本流。
    """
    start_time = time.time()
    resp_id = f"vlm-chat-{uuid.uuid4()}"
    created_ts = int(time.time())

    logger.info(f"收到推理请求 | 模型: {request.model_name} | 流模式: {request.stream}")

    try:
        if request.stream:
            async def sse_generator() -> AsyncGenerator[str, None]:
                usage_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                
                try:
                    stream_raw = await service.inference(
                        **request.model_dump(exclude={"stream"}), stream=True
                    )
                    
                    async for content in stream_raw:  # type: ignore
                        # 处理元数据与 Token 统计
                        if isinstance(content, str) and content.startswith("\n\n=== USAGE ==="):
                            try:
                                stats = json.loads(content.replace("\n\n=== USAGE ===\n", ""))
                                usage_stats.update(stats)
                            except (json.JSONDecodeError, ValueError):
                                pass
                            continue

                        # 构造标准的 OpenAI 风格 Chunk
                        chunk = {
                            "id": resp_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": request.model_name,
                            "choices": [{"delta": {"content": content}, "index": 0, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                    # 流结束：发送带 Usage 的最后一个包
                    final_chunk = {
                        "id": resp_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": request.model_name,
                        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                        "usage": usage_stats
                    }
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

                except Exception as stream_err:
                    logger.error(f"流生成中断: {stream_err}")
                    yield f"data: {json.dumps({'error': str(stream_err)})}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(sse_generator(), media_type="text/event-stream")

        # --- 非流式逻辑 ---
        raw_resp = await service.inference(**request.model_dump(exclude={"stream"}), stream=False)
        duration = time.time() - start_time
        
        usage = raw_resp.get("usage", {}) # type: ignore
        content = raw_resp["choices"][0]["message"]["content"] # type: ignore

        return VLMResponse(
            id=resp_id,
            object="chat.completion",
            created=created_ts,
            model=request.model_name,
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
        logger.exception("推理执行失败")
        raise HTTPException(status_code=500, detail=str(e))


# --- 进程管理与信号监控 ---

vlm_process: subprocess.Popen | None = None

def stop_standalone_vlm():
    """清理并关闭独立的 VLM 进程。"""
    global vlm_process
    if vlm_process:
        logger.info(f"正在终止 VLM 独立进程 [PID: {vlm_process.pid}]...")
        vlm_process.terminate()
        try:
            vlm_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            vlm_process.kill()
        vlm_process = None

def handle_system_signal(sig: int, frame: Any):
    """处理操作系统发送的终止信号。"""
    logger.warning(f"接收到系统信号: {sig}，正在触发安全退出...")
    stop_standalone_vlm()
    sys.exit(0)

atexit.register(stop_standalone_vlm)

if __name__ == "__main__":
    # 针对独立运行模式注册信号
    if os.environ.get("VLM_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, handle_system_signal)
        signal.signal(signal.SIGTERM, handle_system_signal)
    
    logger.info(f"VLM 服务启动 | 端口: {SERVICE_PORT}")
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT, access_log=False)