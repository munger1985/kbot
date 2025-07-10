import os
import time
import psutil
import logging
import numpy as np
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from uuid import uuid4

from .schemas import (
    EmbeddingRequest, 
    EmbeddingResponse, 
    ModelConfigUpdate, 
    ModelConfigUpdateResponse,
    HealthResponse,
    StatsResponse
)
from ...services.embedding.service import EmbeddingService

# 创建路由器
router = APIRouter(prefix="/api/embedding", tags=["embedding"])

# 服务实例ID，用于区分多个实例
INSTANCE_ID = str(uuid4())

# 获取服务实例的依赖函数
async def get_embedding_service() -> EmbeddingService:
    # 在实际应用中，这里可能会从某个全局变量或依赖注入容器中获取服务实例
    # 这里简化处理，假设服务已经在应用启动时初始化
    from ...microservices.embedding.main import get_service_instance
    return get_service_instance()

@router.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """生成文本嵌入向量"""
    try:
        # 调用服务生成嵌入
        embeddings = await embedding_service.embed(request.model_id, request.texts)
        
        # 转换为可序列化的格式
        embeddings_list = embeddings.tolist()
        
        return {
            "embeddings": embeddings_list,
            "model_id": request.model_id,
            "dimensions": embeddings.shape[1] if embeddings.ndim > 1 else len(embeddings)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")

@router.put("/models/{model_id}/config", response_model=ModelConfigUpdateResponse)
async def update_model_config(
    model_id: str,
    config_update: ModelConfigUpdate,
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """更新模型配置"""
    try:
        updated = await embedding_service.update_model_config(
            model_id, 
            config_update.config, 
            config_update.version
        )
        return {
            "updated": updated,
            "model_id": model_id
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error updating model config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update model config: {str(e)}")

@router.get("/health", response_model=HealthResponse)
async def check_health(
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """健康检查端点"""
    try:
        health_status = await embedding_service.get_health_status()
        return health_status
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models": {},
            "error": str(e)
        }

@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """获取服务统计信息"""
    try:
        # 获取模型统计信息
        model_stats = embedding_service.get_model_stats()
        
        # 获取系统资源使用情况
        process = psutil.Process(os.getpid())
        cpu_usage = process.cpu_percent() / psutil.cpu_count()
        memory_usage = process.memory_percent()
        
        return {
            "models": model_stats,
            "instance_id": INSTANCE_ID,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage
        }
    except Exception as e:
        logging.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")