import asyncio
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class BatchProcessor:
    """处理embedding请求的批处理器"""
    
    def __init__(self, model_pool, max_batch_size=64, max_wait_time=0.1):
        self.model_pool = model_pool
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.batches = {}  # model_id -> batch
        self.locks = {}  # model_id -> lock
    
    async def add_to_batch(self, model_id: str, texts: List[str]) -> np.ndarray:
        """添加文本到批处理队列并等待结果"""
        if model_id not in self.locks:
            self.locks[model_id] = asyncio.Lock()
            self.batches[model_id] = {"texts": [], "futures": [], "timer": None}
        
        # 创建future用于获取结果
        future = asyncio.Future()
        
        async with self.locks[model_id]:
            batch = self.batches[model_id]
            text_indices = list(range(len(batch["texts"]), len(batch["texts"]) + len(texts)))
            batch["texts"].extend(texts)
            batch["futures"].append((future, text_indices))
            
            # 检查是否应该处理批次
            should_process = len(batch["texts"]) >= self.max_batch_size
            
            if should_process:
                # 如果批次已满，立即处理
                if batch["timer"]:
                    batch["timer"].cancel()
                    batch["timer"] = None
                await self._process_batch(model_id)
            elif len(batch["texts"]) > 0 and not batch["timer"]:
                # 如果不立即处理，设置定时器
                batch["timer"] = asyncio.create_task(self._process_after_delay(model_id))
        
        # 等待结果
        return await future
    
    async def _process_after_delay(self, model_id: str):
        """等待一段时间后处理批次"""
        try:
            await asyncio.sleep(self.max_wait_time)
            async with self.locks[model_id]:
                await self._process_batch(model_id)
        except asyncio.CancelledError:
            pass  # 定时器被取消
        except Exception as e:
            logger.error(f"Error in batch timer for {model_id}: {str(e)}")
    
    async def _process_batch(self, model_id: str):
        """处理一个批次"""
        batch = self.batches[model_id]
        if batch["timer"]:
            batch["timer"].cancel()
            batch["timer"] = None
        
        if not batch["texts"]:
            return
        
        texts = batch["texts"]
        futures = batch["futures"]
        
        # 重置批次
        self.batches[model_id] = {"texts": [], "futures": [], "timer": None}
        
        try:
            # 获取模型并处理批次
            model = await self.model_pool.get_model(model_id)
            logger.info(f"Processing batch of {len(texts)} texts for model {model_id}")
            embeddings = await model.embed(texts)
            
            # 分发结果给所有等待的future
            for future, indices in futures:
                if not future.done():
                    future.set_result(embeddings[indices])
        except Exception as e:
            logger.error(f"Error processing batch for model {model_id}: {str(e)}")
            # 处理错误
            for future, _ in futures:
                if not future.done():
                    future.set_exception(e)