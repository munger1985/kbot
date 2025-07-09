import logging
import numpy as np
from typing import List, Dict, Any, Optional

from .model_pool import EmbeddingModelPool
from .batch_processor import BatchProcessor
from .health_check import HealthChecker

logger = logging.getLogger(__name__)

class EmbeddingService:
    """嵌入服务，整合模型池、批处理和健康检查功能"""
    
    def __init__(self, max_idle_time=3600, max_batch_size=64, max_wait_time=0.1, health_check_interval=300):
        self.model_pool = EmbeddingModelPool(max_idle_time=max_idle_time)
        self.batch_processor = BatchProcessor(
            model_pool=self.model_pool,
            max_batch_size=max_batch_size,
            max_wait_time=max_wait_time
        )
        self.health_checker = HealthChecker(model_pool=self.model_pool)
        self.health_check_interval = health_check_interval
    
    async def start(self):
        """启动服务"""
        await self.model_pool.start()
        await self.health_checker.start(check_interval=self.health_check_interval)
        logger.info("Embedding service started")
    
    async def stop(self):
        """停止服务"""
        await self.health_checker.stop()
        await self.model_pool.stop()
        logger.info("Embedding service stopped")
    
    async def update_model_config(self, model_id: str, config: Dict[str, Any], version: Optional[str] = None) -> bool:
        """更新模型配置"""
        return await self.model_pool.update_model_config(model_id, config, version)
    
    async def embed(self, model_id: str, texts: List[str]) -> np.ndarray:
        """生成文本嵌入"""
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # 使用批处理器处理请求
        return await self.batch_processor.add_to_batch(model_id, texts)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """获取服务健康状态"""
        return await self.health_checker.check_all_models()
    
    def get_model_stats(self) -> Dict[str, Any]:
        """获取模型使用统计"""
        stats = {}
        for model_id in self.model_pool.models:
            stats[model_id] = {
                "request_count": self.model_pool.request_counts.get(model_id, 0),
                "last_used": self.model_pool.last_used.get(model_id, 0)
            }
        return stats