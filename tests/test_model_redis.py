import sys
from pathlib import Path
import asyncio

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dao.repositories.kbot_md_models_repo import KbotMdModelsRepository
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

async def test_get_all_cached_models():
    """测试获取所有缓存模型"""
    redis = AsyncRedisPool(db=1)
    async with redis as r:
        # 检查 Redis 连接是否正常
        
        
        # 检查键是否存在
        result = await r.keys("model:*")
        print(f"Cached models: {result}")
        
        # 如果结果为空，检查是否有其他键
        if not result:
            all_keys = await r.keys("*")
            print(f"All keys in Redis: {all_keys}")


if __name__ == "__main__":
    asyncio.run(test_sync_all_available_models())