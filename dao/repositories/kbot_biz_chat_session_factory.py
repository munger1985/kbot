from loguru import logger
from .kbot_chat_session_repo.kbot_biz_chat_session_es import ElasticsearchChatSessionRepository
from .kbot_chat_session_repo.kbot_biz_chat_session_oracle import OracleChatSessionRepository
from dao.repositories.kbot_md_db_conf_repo import KbotMdDbConfRepository
from core.dictionary import DbType


class ChatSessionRepositoryFactory:
    """会话存储库工厂类"""
    
    @staticmethod
    async def create_repository(kb_id: int) -> ElasticsearchChatSessionRepository | OracleChatSessionRepository:
        """
        创建会话存储库实例
        
        Args:
            kb_id: 知识库ID
            
        Returns:
            ElasticsearchChatSessionRepository 或 OracleChatSessionRepository 实例
        """
        
        #repo_type = RepositoryType.ORACLE
        db_repo = KbotMdDbConfRepository()
        db_conf = await db_repo.get_by_kbid(kb_id)
        if db_conf is None:
            logger.error(f"未找到知识库 {kb_id} 的向量库配置")
            raise ValueError(f"未找到知识库 {kb_id} 的向量库配置")
        
        connstr = db_conf.db_conn_str
        if connstr is None:
            logger.error(f"知识库 {kb_id} 的数据库连接字符串为空")
            raise ValueError(f"知识库 {kb_id} 的数据库连接字符串为空")
        
        db_type = db_conf.db_type
        
        if db_type == DbType.ORACLE:
            repository = OracleChatSessionRepository(kb_id)
        elif db_type == DbType.ELASTICSEARCH:
            repository = ElasticsearchChatSessionRepository(kb_id)
        else:
            raise ValueError(f"不支持的存储库类型: {DbType(db_type) or db_type}")
        
        # 初始化连接
        if await repository.initialize(connstr):
            logger.info(f"成功创建 {DbType(db_type)} 存储库实例")
            return repository
        else:
            raise Exception(f"初始化{DbType(db_type)}存储库失败")