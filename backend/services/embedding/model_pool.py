import time
import asyncio
import logging
from typing import Dict, Any, Optional
import numpy as np

from ...models.embedding.txt.base import CloudEmbeddingConfig
from ...models.embedding.txt.cloud import CloudEmbedding

logger = logging.getLogger(__name__)

class EmbeddingModelPool:
    """管理多个embedding模型实例的池"""
    
    def __init__(self, max_idle_time=3600):  # 默认1小时不用自动卸载
        self.models = {}  # 存储已初始化的模型
        self.model_configs = {}  # 存储模型配置
        self.config_versions = {}  # 存储配置版本
        self.last_used = {}  # 跟踪模型最后使用时间
        self.request_counts = {}  # 跟踪模型请求次数
        self.max_idle_time = max_idle_time
        self.cleanup_task = None
    
    async def start(self):
        """启动定期清理任务"""
        self.cleanup_task = asyncio.create_task(self._cleanup_idle_models())
        logger.info("Model pool started with idle cleanup task")
    
    async def stop(self):
        """停止所有任务并释放资源"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        await self.shutdown_all()
        logger.info("Model pool stopped")
    
    async def _cleanup_idle_models(self):
        """定期检查并卸载闲置模型"""
        try:
            while True:
                await asyncio.sleep(300)  # 每5分钟检查一次
                current_time = time.time()
                models_to_remove = []
                
                for model_id, last_used in self.last_used.items():
                    if current_time - last_used > self.max_idle_time:
                        models_to_remove.append(model_id)
                
                for model_id in models_to_remove:
                    if model_id in self.models:
                        await self.models[model_id].shutdown()
                        del self.models[model_id]
                        logger.info(f"Model {model_id} unloaded due to inactivity")
        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}")
    
    async def get_model(self, model_id: str) -> CloudEmbedding:
        """获取指定ID的模型，如果不存在则初始化"""
        if model_id not in self.models:
            if model_id not in self.model_configs:
                raise ValueError(f"Model config for {model_id} not found")
            
            config = self.model_configs[model_id]
            model = CloudEmbedding(config)
            await model.startup()
            self.models[model_id] = model
            self.request_counts[model_id] = 0
            logger.info(f"Model {model_id} initialized")
        
        # 更新使用统计
        self.last_used[model_id] = time.time()
        self.request_counts[model_id] = self.request_counts.get(model_id, 0) + 1
        return self.models[model_id]
    
    async def update_model_config(self, model_id: str, config: Dict[str, Any], version: Optional[str] = None) -> bool:
        """更新模型配置，带版本控制"""
        # 如果提供了版本且与当前版本相同，则跳过更新
        if version and model_id in self.config_versions and self.config_versions[model_id] == version:
            return False  # 配置未变更
        
        # 更新配置和版本
        embedding_config = CloudEmbeddingConfig(**config)
        self.model_configs[model_id] = embedding_config
        if version:
            self.config_versions[model_id] = version
        
        # 如果模型已存在，则重新初始化
        if model_id in self.models:
            old_model = self.models[model_id]
            await old_model.shutdown()
            
            new_model = CloudEmbedding(embedding_config)
            await new_model.startup()
            self.models[model_id] = new_model
            logger.info(f"Model {model_id} reinitialized with new config")
        
        return True  # 配置已更新
    
    async def reinitialize_model(self, model_id: str) -> bool:
        """重新初始化模型（用于健康检查失败后恢复）"""
        if model_id not in self.model_configs:
            return False
        
        if model_id in self.models:
            try:
                await self.models[model_id].shutdown()
            except Exception:
                pass  # 忽略关闭错误
        
        try:
            config = self.model_configs[model_id]
            model = CloudEmbedding(config)
            await model.startup()
            self.models[model_id] = model
            logger.info(f"Model {model_id} successfully reinitialized")
            return True
        except Exception as e:
            logger.error(f"Failed to reinitialize model {model_id}: {str(e)}")
            return False
    
    async def shutdown_all(self):
        """关闭所有模型"""
        for model_id, model in list(self.models.items()):
            try:
                await model.shutdown()
                logger.info(f"Model {model_id} shutdown")
            except Exception as e:
                logger.error(f"Error shutting down model {model_id}: {str(e)}")
        self.models = {}