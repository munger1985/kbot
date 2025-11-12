
import asyncio
from pathlib import Path
import sys

# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Use absolute imports from project root
from dao.repositories.kbot_biz_txt_embedding_factory import EmbeddingRepositoryFactory

async def print_all_embeddings(max_display: int = 50) -> bool:
    """打印所有嵌入记录到控制台
    
    Args:
        kb_id: 知识库ID，如果为None则打印所有记录
        max_display: 最大显示记录数，避免控制台输出过多
        
    Returns:
        是否执行成功
    """
    kb_id = 104
    try:
        print(f"开始获取嵌入记录，kb_id: {kb_id}")
        embed_repo = await EmbeddingRepositoryFactory().create_repository(kb_id)
        # 调用get_all_embeddings获取所有记录
        all_embeddings = await embed_repo.get_all_embeddings(kb_id=kb_id) # type: ignore
        
        if not all_embeddings:
            print("没有找到任何记录")
            return True
        
        total_count = len(all_embeddings)
        print(f"共找到 {total_count} 条记录")
        
        # 打印摘要信息
        print("\n" + "="*80)
        print(f"嵌入记录统计 (kb_id: {kb_id if kb_id else 'ALL'})")
        print("="*80)
        print(f"总记录数: {total_count}")
        
        # 按状态统计
        status_count = {}
        kb_count = {}
        for embedding in all_embeddings:
            status = embedding.status
            kb = embedding.kb_id
            status_count[status] = status_count.get(status, 0) + 1
            kb_count[kb] = kb_count.get(kb, 0) + 1
        
        print(f"状态分布: {status_count}")
        if len(kb_count) > 1:
            print(f"知识库分布: {kb_count}")
        print("-" * 80)
        
        # 打印详细记录（限制数量）
        display_count = min(total_count, max_display)
        print(f"\n详细记录 (显示前 {display_count} 条):")
        print("-" * 80)
        
        for i, embedding in enumerate(all_embeddings[:display_count]):
            print(f"\n[{i+1}] 记录ID: {embedding.embed_id}")
            print(f"    知识库ID: {embedding.kb_id}")
            print(f"    文件ID: {embedding.file_id}")
            print(f"    安全等级: {embedding.security_level}")
            print(f"    状态: {embedding.status}")
            
            # 处理chunk_doc显示（截断长文本）
            chunk_doc = embedding.chunk_doc
            if chunk_doc and len(chunk_doc) > 100:
                chunk_doc = chunk_doc[:100] + "..."
            print(f"    内容: {chunk_doc}")
            
            # 显示元数据
            if embedding.chunk_metadata:
                print(f"    块元数据: {embedding.chunk_metadata}")
            if embedding.biz_metadata:
                print(f"    业务元数据: {embedding.biz_metadata}")
            
            # 显示向量信息
            embedding_length = len(embedding.embedding) if embedding.embedding else 0
            print(f"    向量维度: {embedding_length}")
            if embedding_length > 0:
                # 显示前5个向量值作为示例
                sample_vector = embedding.embedding[:5]
                print(f"    向量示例: {sample_vector}...")
            
            print("-" * 40)
        
        # 如果记录数超过显示限制，提示用户
        if total_count > display_count:
            print(f"\n... 还有 {total_count - display_count} 条记录未显示")
            print("如需查看全部记录，请增加 max_display 参数")
        
        # 打印文件统计
        file_stats = {}
        for embedding in all_embeddings:
            file_id = embedding.file_id
            file_stats[file_id] = file_stats.get(file_id, 0) + 1
        
        print(f"\n文件统计 (共 {len(file_stats)} 个文件):")
        for file_id, count in list(file_stats.items())[:10]:  # 只显示前10个文件
            print(f"  {file_id}: {count} 个chunk")
        
        if len(file_stats) > 10:
            print(f"  ... 还有 {len(file_stats) - 10} 个文件")
        
        return True
        
    except Exception as e:
        print(f"打印嵌入记录失败: {e}")
        print(f"错误: {e}")
        return False
    finally:
        from core.database.vec_elasticsearch import es_client_manager
        await es_client_manager.close_all()
    
if __name__ == "__main__":
    asyncio.run(print_all_embeddings())