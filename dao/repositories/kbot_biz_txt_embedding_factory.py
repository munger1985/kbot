from enum import Enum
from loguru import logger
from .kbot_embedding_repo.kbot_biz_txt_embedding_oracle import OracleEmbeddingRepository
from .kbot_embedding_repo.kbot_biz_txt_embedding_es import ElasticsearchEmbeddingRepository
from dao.repositories.kbot_md_db_conf_repo import KbotMdDbConfRepository
from core.dictionary import DbType


class EmbeddingRepositoryFactory:
    """嵌入存储库工厂类"""
    
    @staticmethod
    async def create_repository(kb_id: int) -> OracleEmbeddingRepository | ElasticsearchEmbeddingRepository | None:
        """
        创建嵌入存储库实例
        
        Args:
            kb_id: 知识库ID
            
        Returns:
            OracleEmbeddingRepository 或 ElasticsearchEmbeddingRepository 实例
        """
        
        #repo_type = RepositoryType.ORACLE
        db_repo = KbotMdDbConfRepository()
        db_conf = await db_repo.get_by_kbid(kb_id)
        if db_conf is None:
            logger.error(f"未找到知识库 {kb_id} 的向量库配置")
            return None
        
        connstr = db_conf.db_conn_str
        db_type = db_conf.db_type
        
        if db_type == DbType.ORACLE:
            repository = OracleEmbeddingRepository(kb_id)
        elif db_type == DbType.ELASTICSEARCH:
            repository = ElasticsearchEmbeddingRepository(kb_id)
        else:
            raise ValueError(f"不支持的存储库类型: {DbType(db_type) or db_type}")
        
        # 初始化连接
        if await repository.initialize(connstr): # type: ignore
            logger.info(f"成功创建 {DbType(db_type)} 存储库实例")
            return repository
        else:
            raise Exception(f"初始化{DbType(db_type)}存储库失败")