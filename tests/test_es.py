import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from dao.repositories.kbot_biz_txt_embedding_factory import EmbeddingRepositoryFactory



async def test_get_summary_id_by_chunk_id_real():
    """
    测试实际的 get_summary_id_by_chunk_id 方法
    """

    # 创建 Elasticsearch 存储库实例
    repo = await EmbeddingRepositoryFactory.create_repository(kb_id=104)
    if repo is None:
        return
    
    # 调用方法并验证结果
    file_id = "5a3d8540-9d09-43eb-aad2-0ea87be306a3"  # 替换为实际的文件 ID
    chunk_id = "b433cfd3-e78e-402e-aa26-6f430a556541"  # 替换为实际的块 ID
    
    result = await repo.get_summary_id_by_chunk_id(file_id, chunk_id)
    
    # 根据实际情况调整断言
    print("summary_chunk_id:")
    print(result)

if __name__ == '__main__':
    import asyncio
    asyncio.run(test_get_summary_id_by_chunk_id_real())
    
