from datetime import datetime, timezone
from loguru import logger
from fastapi import BackgroundTasks

from platform_core.database import db_instance
from platform_core.exceptions import *
from dao.repositories import FileRepository
from services.search import TxtBaseSearchResult
from platform_clients import AIModelClient
from agent.memory import MemoryService
from ..orchestrator import DocOrchestrator


class DocAgent:
    """文档智能体核心类，提供流式/非流式对话、检索、记忆持久化等功能"""

    def __init__(self):
        """初始化文档智能体，加载依赖的客户端与服务实例"""
        self.model_client = AIModelClient()
        self.memory_service = MemoryService()
        self.orchestrator = DocOrchestrator()

    @property
    def db_session(self):
        """获取数据库会话对象"""
        return db_instance().get_session()

    # ========================== 核心业务接口 ==========================
    async def rag_retrieval(
        self,
        session_id: str,
        agent_id: int,
        question: str,
        standalone_query: str,
        search_keywords: str,
        security_level: int,
        user_id: str,
        tags: list[str] = [],
        background_tasks: BackgroundTasks | None = None,
    ) -> list[dict]:
        """
        知识库检索入口：已修复 ContextMemory 传递与持久化逻辑
        """
        request_time = datetime.now(tz=timezone.utc)
        
        # 1. 确保会话存在并获取初始上下文 (包含 user_profile 等)
        # 假设 ensure_session_exists 现在返回或初始化了 ContextMemory
        await self.memory_service.ensure_session_exists(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            question=question
        )

        # 3. 运行核心流水线获取素材
        pipe_out = await self.orchestrator.run_pipeline(
            agent_id=agent_id,
            standalone_query=standalone_query,
            search_keywords=search_keywords,
            security_level=security_level,
            tags=tags
        )

        # 4. 将检索结果存入 context 并富化元数据
        enriched_refs = await self._enrich_results_with_metadata(pipe_out['kb_results'])
      
        return enriched_refs

    async def _enrich_results_with_metadata(self, kb_results: list[TxtBaseSearchResult]) -> list[dict]:
        """
        将原始检索结果转换为带文件元数据的字典，用于前端展示

        Args:
            kb_results: 知识库原始检索结果列表

        Returns:
             enriched_refs:  enriched 后的引用文档列表
        """
        if not kb_results:
            return []

        # 安全收集文件ID，确保为字符串类型
        file_ids = []
        for res in kb_results:
            file_id = res.file_id
            if not isinstance(file_id, str):
                file_id = str(file_id)
            file_ids.append(file_id)

        unique_file_ids = list(set(file_ids))
        logger.debug(f"从{len(kb_results)}条结果中收集到{len(unique_file_ids)}个唯一文件ID")

        file_name_map = {}
        try:
            async with self.db_session as session:
                file_repo = FileRepository(session)
                file_name_map = await file_repo.get_names_by_ids(unique_file_ids)
                logger.debug(f"已映射{len(file_name_map)}个文件ID与文件名")
        except Exception as e:
            logger.error(f"获取引用文档文件名失败：{e}")

        references = []
        for idx, res in enumerate(kb_results):
            try:
                ref = res.to_dict()

                # 确保文件ID为字符串，用于构建URL
                file_id = res.file_id
                if not isinstance(file_id, str):
                    file_id = str(file_id)

                ref["file_name"] = file_name_map.get(file_id, "未知文件")
                references.append(ref)
            except Exception as e:
                logger.error(f"处理检索结果异常，索引{idx}：{e}，异常类型：{type(e).__name__}，结果：{res}")
                raise
        return references