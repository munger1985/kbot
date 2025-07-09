import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any, List

from backend.services.embedding.model_base import EmbeddingModel
from backend.services.embedding.model_pool import EmbeddingModelPool
from backend.services.embedding.batch_processor import BatchProcessor
from backend.services.embedding.health_check import HealthChecker
from backend.services.embedding.service import EmbeddingService

# 模拟嵌入模型类
class MockEmbeddingModel(EmbeddingModel):
    def __init__(self, model_id: str, dimensions: int = 768):
        self.model_id = model_id
        self.dimensions = dimensions
        self.is_loaded = False
        self.embed_calls = []
        self.load_calls = 0
        self.unload_calls = 0
        self.health_check_calls = 0
    
    async def load(self):
        self.load_calls += 1
        self.is_loaded = True
    
    async def unload(self):
        self.unload_calls += 1
        self.is_loaded = False
    
    async def embed(self, texts: List[str]) -> np.ndarray:
        self.embed_calls.append(texts)
        # 返回随机嵌入向量
        return np.random.random((len(texts), self.dimensions))
    
    async def health_check(self) -> Dict[str, Any]:
        self.health_check_calls += 1
        return {"status": "connected", "latency": 0.01}

# 模型工厂函数模拟
async def mock_model_factory(model_id: str, config: Dict[str, Any]) -> EmbeddingModel:
    dimensions = config.get("dimensions", 768)
    return MockEmbeddingModel(model_id, dimensions)

# 测试模型池
@pytest.mark.asyncio
async def test_model_pool():
    # 创建模型池
    pool = EmbeddingModelPool(max_idle_time=60)
    pool._create_model = mock_model_factory
    
    # 启动模型池
    await pool.start()
    
    # 测试获取模型
    model_id = "test_model"
    config = {"dimensions": 512}
    await pool.update_model_config(model_id, config)
    
    model = await pool.get_model(model_id)
    assert model.model_id == model_id
    assert model.dimensions == 512
    assert model.is_loaded
    assert model.load_calls == 1
    
    # 测试重复获取同一个模型
    model2 = await pool.get_model(model_id)
    assert model is model2  # 应该是同一个实例
    assert model.load_calls == 1  # 不应该重复加载
    
    # 测试更新模型配置
    new_config = {"dimensions": 768}
    updated = await pool.update_model_config(model_id, new_config)
    assert updated
    
    # 获取更新后的模型
    model3 = await pool.get_model(model_id)
    assert model3 is not model  # 应该是新实例
    assert model3.dimensions == 768
    
    # 测试停止模型池
    await pool.stop()

# 测试批处理器
@pytest.mark.asyncio
async def test_batch_processor():
    # 创建模型池模拟
    pool = MagicMock(spec=EmbeddingModelPool)
    model = MockEmbeddingModel("test_model")
    pool.get_model = AsyncMock(return_value=model)
    
    # 创建批处理器
    processor = BatchProcessor(pool, max_batch_size=2, max_wait_time=0.1)
    
    # 测试批处理
    texts1 = ["text1"]
    texts2 = ["text2", "text3"]
    
    # 异步提交两个批次
    task1 = asyncio.create_task(processor.add_to_batch("test_model", texts1))
    task2 = asyncio.create_task(processor.add_to_batch("test_model", texts2))
    
    # 等待任务完成
    result1 = await task1
    result2 = await task2
    
    # 验证结果
    assert result1.shape == (1, model.dimensions)
    assert result2.shape == (2, model.dimensions)
    
    # 验证模型调用
    assert len(model.embed_calls) == 2  # 应该有两次调用
    # 由于批处理的异步性质，无法保证确切的批次组合，但总文本数应该正确
    total_texts = sum(len(texts) for texts in model.embed_calls)
    assert total_texts == 3

# 测试健康检查器
@pytest.mark.asyncio
async def test_health_checker():
    # 创建模型池模拟
    pool = MagicMock(spec=EmbeddingModelPool)
    model1 = MockEmbeddingModel("model1")
    model2 = MockEmbeddingModel("model2")
    
    # 模拟模型池的get_all_models方法
    async def mock_get_all_models():
        return {"model1": model1, "model2": model2}
    
    pool.get_all_models = mock_get_all_models
    
    # 创建健康检查器
    checker = HealthChecker(pool)
    
    # 执行健康检查
    result = await checker.check_all_models()
    
    # 验证结果
    assert result["status"] == "healthy"
    assert "timestamp" in result
    assert "models" in result
    assert "model1" in result["models"]
    assert "model2" in result["models"]
    assert model1.health_check_calls == 1
    assert model2.health_check_calls == 1

# 测试嵌入服务
@pytest.mark.asyncio
async def test_embedding_service():
    # 创建服务
    service = EmbeddingService(
        max_idle_time=60,
        max_batch_size=10,
        max_wait_time=0.1,
        health_check_interval=60
    )
    
    # 替换模型工厂
    service.model_pool._create_model = mock_model_factory
    
    # 启动服务
    await service.start()
    
    # 配置模型
    model_id = "test_model"
    config = {"dimensions": 512}
    updated = await service.update_model_config(model_id, config)
    assert updated
    
    # 测试嵌入
    texts = ["这是测试文本1", "这是测试文本2"]
    embeddings = await service.embed(model_id, texts)
    
    # 验证结果
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 512)
    
    # 测试健康状态
    health = await service.get_health_status()
    assert health["status"] in ["healthy", "degraded", "unhealthy"]
    assert "models" in health
    
    # 测试统计信息
    stats = service.get_model_stats()
    assert model_id in stats
    assert "request_count" in stats[model_id]
    assert stats[model_id]["request_count"] > 0
    
    # 停止服务
    await service.stop()

# 测试空文本列表
@pytest.mark.asyncio
async def test_empty_texts():
    service = EmbeddingService()
    service.model_pool._create_model = mock_model_factory
    await service.start()
    
    # 测试空文本列表
    with pytest.raises(ValueError):
        await service.embed("test_model", [])
    
    await service.stop()

# 测试无效模型ID
@pytest.mark.asyncio
async def test_invalid_model_id():
    service = EmbeddingService()
    service.model_pool._create_model = mock_model_factory
    await service.start()
    
    # 尝试使用未配置的模型ID
    with pytest.raises(Exception):  # 具体异常类型取决于实现
        await service.embed("non_existent_model", ["测试文本"])
    
    await service.stop()