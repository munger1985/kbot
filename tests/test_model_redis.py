import sys
from pathlib import Path
import asyncio


from dotenv import load_dotenv
load_dotenv()

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dao.repositories.kbot_md_models_repo import *
from core.database.meta_redis import AsyncRedisPool

async def test_sync_all_available_models():
    """测试同步所有可用模型到 Redis"""
    repo = KbotMdModelsRepository()
    
    # 执行同步操作
    result = await repo.sync_all_available_models()
    assert result is True
    
    # 验证 Redis 中是否有数据
    redis = AsyncRedisPool(db=1)
    async with redis as r:
        keys = await r.keys("model:*")
        print(f"Synced models in Redis: {keys}")
        
        # 如果 Redis 中没有数据，检查 Redis 连接和操作是否正常
        if not keys:
            is_connected = await r.health_check()
            print(f"Redis connection status: {'OK' if is_connected else 'Failed'}")
            all_keys = await r.keys("*")
            print(f"All keys in Redis: {all_keys}")

async def test_enable_model(model_id: int):
    """测试启用模型"""
    repo = KbotMdModelsRepository()
    result = await repo.enable_model(model_id)
    assert result is True
    print(f"Model {model_id} enabled successfully")

async def test_disable_model(model_id: int):
    """测试禁用模型"""
    repo = KbotMdModelsRepository()
    result = await repo.disable_model(model_id)
    assert result is True
    print(f"Model {model_id} disabled successfully")

async def test_get_all_available_models():
    """测试获取所有可用模型"""
    repo = KbotMdModelsRepository()
    models = await repo.get_all_available_models()
    
    print(f"Available models: {models}")

async def test_get_available_model_by_id(model_id: int):
    """测试通过 ID 获取模型"""
    repo = KbotMdModelsRepository()
    model = await repo.get_available_model_by_id(model_id)
    
    print(f"Model {model_id}: {model}")

if __name__ == "__main__":
    model_id = 23
    # asyncio.run(test_enable_model(model_id))
    # asyncio.run(test_disable_model(model_id))
    # asyncio.run(test_get_all_available_models())
    # asyncio.run(test_get_available_model_by_id(model_id))
    asyncio.run(test_sync_all_available_models())