import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class HealthChecker:
    """模型健康检查器"""
    
    def __init__(self, model_pool):
        self.model_pool = model_pool
        self.check_task = None
    
    async def start(self, check_interval=300):
        """启动定期健康检查任务"""
        self.check_task = asyncio.create_task(self._periodic_health_check(check_interval))
        logger.info(f"Health checker started with interval {check_interval}s")
    
    async def stop(self):
        """停止健康检查任务"""
        if self.check_task:
            self.check_task.cancel()
            try:
                await self.check_task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker stopped")
    
    async def _periodic_health_check(self, check_interval):
        """定期检查所有模型的健康状态"""
        try:
            while True:
                await asyncio.sleep(check_interval)
                await self.check_all_models()
        except asyncio.CancelledError:
            logger.info("Health check task cancelled")
        except Exception as e:
            logger.error(f"Error in health check task: {str(e)}")
    
    async def check_all_models(self):
        """检查所有已加载模型的健康状态"""
        logger.info("Running health check on all models")
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }
        
        has_errors = False
        
        # 检查每个已加载模型的连接状态
        for model_id, model in list(self.model_pool.models.items()):
            try:
                # 简单的连接测试
                test_result = await self._test_model_connection(model)
                health_status["models"][model_id] = {
                    "status": "connected" if test_result else "error",
                    "last_used": datetime.fromtimestamp(self.model_pool.last_used.get(model_id, 0)).isoformat()
                }
                
                if not test_result:
                    has_errors = True
                    logger.warning(f"Model {model_id} connection test failed, attempting to reinitialize")
                    # 尝试重新初始化模型
                    await self.model_pool.reinitialize_model(model_id)
            except Exception as e:
                has_errors = True
                health_status["models"][model_id] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"Health check for model {model_id} failed: {str(e)}")
                # 尝试重新初始化模型
                await self.model_pool.reinitialize_model(model_id)
        
        if has_errors:
            health_status["status"] = "degraded"
        
        return health_status
    
    async def _test_model_connection(self, model):
        """测试模型连接是否正常"""
        try:
            # 使用简单文本测试连接
            test_text = ["Connection test"]
            await model.embed(test_text)
            return True
        except Exception as e:
            logger.error(f"Model connection test failed: {str(e)}")
            return False