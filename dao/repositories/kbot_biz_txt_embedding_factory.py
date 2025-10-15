from enum import Enum
from loguru import logger
from .kbot_embedding_repo.kbot_biz_txt_embedding_oracle import OracleEmbeddingRepository
from .kbot_embedding_repo.kbot_biz_txt_embedding_es import ElasticsearchEmbeddingRepository


class RepositoryType(Enum):
    ORACLE = "oracle"
    ELASTICSEARCH = "elasticsearch"

class EmbeddingRepositoryFactory:
    """嵌入存储库工厂类"""
    
    @staticmethod
    async def create_repository(kb_id: int) -> OracleEmbeddingRepository | ElasticsearchEmbeddingRepository:
        """
        创建嵌入存储库实例
        
        Args:
            kb_id: 知识库ID
            
        Returns:
            OracleEmbeddingRepository 或 ElasticsearchEmbeddingRepository 实例
        """
        
        repo_type = RepositoryType.ORACLE
        
        if repo_type == RepositoryType.ORACLE:
            repository = OracleEmbeddingRepository(kb_id)
        elif repo_type == RepositoryType.ELASTICSEARCH:
            repository = ElasticsearchEmbeddingRepository(kb_id)
        else:
            raise ValueError(f"不支持的存储库类型: {repo_type}")
        
        # 初始化连接
        if await repository.initialize():
            logger.info(f"成功创建 {repo_type.value} 存储库实例")
            return repository
        else:
            raise Exception(f"初始化{repo_type.value}存储库失败")