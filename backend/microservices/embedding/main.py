import os
import asyncio
import logging
import uvicorn
from fastapi import FastAPI
from typing import Optional

from .config import load_config, get_model_config_from_env
from ...services.embedding.service import EmbeddingService
from ...api.embedding import router as embedding_router

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("embedding_service")

# 全局服务实例
_service_instance: Optional[EmbeddingService] = None

def get_service_instance() -> EmbeddingService:
    """获取服务实例"""
    global _service_instance
    if _service_instance is None:
        raise RuntimeError("Service not initialized")
    return _service_instance

async def initialize_service():
    """初始化服务"""
    global _service_instance
    
    # 加载配置
    config = load_config()
    
    # 创建服务实例
    service = EmbeddingService(
        max_idle_time=config.max_idle_time,
        max_batch_size=config.max_batch_size,
        max_wait_time=config.max_wait_time,
        health_check_interval=config.health_check_interval
    )
    
    # 启动服务
    await service.start()
    
    # 加载默认模型配置
    for model_id, model_config in config.default_models.items():
        await service.update_model_config(model_id, model_config)
    
    # 从环境变量加载额外的模型配置
    for model_id in ["text2vec", "bge", "m3e", "custom"]:  # 预定义的模型ID列表
        env_config = get_model_config_from_env(model_id)
        if env_config:
            await service.update_model_config(model_id, env_config)
    
    _service_instance = service
    logger.info("Embedding service initialized")
    return service

async def shutdown_service():
    """关闭服务"""
    global _service_instance
    if _service_instance:
        await _service_instance.stop()
        _service_instance = None
        logger.info("Embedding service shutdown")

def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="Embedding Service",
        description="文本嵌入服务",
        version="1.0.0"
    )
    
    # 添加路由
    app.include_router(embedding_router)
    
    # 添加启动和关闭事件
    @app.on_event("startup")
    async def startup():
        await initialize_service()
    
    @app.on_event("shutdown")
    async def shutdown():
        await shutdown_service()
    
    return app

def start_service():
    """启动服务"""
    config = load_config()
    app = create_app()
    
    # 设置日志级别
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level)
    
    # 启动服务
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=config.workers,
        log_level=config.log_level.lower()
    )

if __name__ == "__main__":
    start_service()