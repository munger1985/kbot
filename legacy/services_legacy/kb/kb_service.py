from loguru import logger
from dao.repositories import KBRepository
from platform_core.database.oracle import get_session
from platform_core.exceptions import *
from services.graph.graph_service import GraphService
from .file_service import FileService
from .schema import KBModelParams


class KBService:
    def __init__(self):
        pass

    @property
    def db_session(self):
        return get_session()
        
    async def remove_tb(self, kb_id: int, cascade: bool = True):
        """根据ID删除文件类知识库, 并按需删除其中所有文件以及文件向量数据

        Args:
            kb_id: 知识库ID
            cascade: 是否删除知识库中的所有文件，默认True
        """
        try:
            if cascade:
                # 1. 删除知识库中的所有文件
                await FileService().delete_file_service(kb_id)
                logger.info(f"知识库 {kb_id} 中的所有文件删除成功")
                # 2. 删除知识库中的所有图谱数据
                await GraphService().delete_graph_by_kb(kb_id)
                logger.info(f"知识库 {kb_id} 中的所有图谱数据删除成功")

            # 删除知识库
            async with self.db_session as session:
                kb_repo = KBRepository(session)
                await kb_repo.delete(kb_id)
            logger.info(f"知识库 {kb_id} 删除成功")
        
        except Exception as e:
            handle_exception(e, "删除知识库失败")

    async def remove_db(self, kb_id: int, cascade: bool = True):
        """根据ID删除DB类知识库, 并按需删除其中所有已录入的数据库元数据

        Args:
            kb_id: 知识库ID
        Args:
            kb_id: 知识库ID
            cascade: 是否删除知识库中的所有元数据，默认True
        """
        try:
            # if cascade:
            #     # 延迟导入, 避免循环依赖
            #     # from services.search import SQLDDLService, SQLExampleService
            #     # 1. 删除知识库中的所有示例SQL
            #     deleted_sql_count = await SQLExampleService().delete_by_kb(kb_id)
            #     logger.info(f"知识库 {kb_id} 中的所有示例SQL删除成功, 共 {deleted_sql_count} 条数据")
            #     # 2. 删除知识库中的所有元数据
            #     deleted_ddl_count = await SQLDDLService().delete_by_kb(kb_id)
            #     logger.info(f"知识库 {kb_id} 中的所有元数据删除成功, 共 {deleted_ddl_count} 条数据")

            # 删除知识库
            async with self.db_session as session:
                kb_repo = KBRepository(session)
                await kb_repo.delete(kb_id)
            logger.info(f"知识库 {kb_id} 删除成功")
        
        except Exception as e:
            handle_exception(e, "删除知识库失败")

    async def get(self, kb_id: int) -> dict:
        """根据ID获取知识库
        
        Args:
            kb_id: 知识库ID，若为空则获取所有知识库

        Returns:
            dict: 知识库对象
        """
        try:
            async with self.db_session as session:
                kb_repo = KBRepository(session)
                kb = await kb_repo.get_by_id(kb_id)
            
            return {
                "kb_id": kb.kb_id,
                "domain_id": kb.domain_id,
                "kb_name": kb.kb_name,
                "category": kb.kb_category,
                "engine": kb.engine,
                "descs": kb.descs,
                "models": kb.models,
                "is_active": kb.kb_status == 1,
                "security_level": kb.security_level,
                "process_priority": kb.process_priority,
                "created_by": kb.created_by,
                "created_at": kb.created_time,
                "updated_by": kb.updated_by,
                "updated_at": kb.updated_time
            }
        
        except Exception as e:
            handle_exception(e, "获取知识库失败")
        
    async def get_all(self, domain_id: int | None = None, 
                      is_active: bool | None = None, 
                      category: int | None = None) -> list[dict]:
        
        """获取所有知识库列表

        Args:
            domain_id: 业务域ID,
            is_active: 是否活跃

        Returns:
            list[dict]: 知识库对象列表
        """
        try:
            async with self.db_session as session:
                kb_repo = KBRepository(session)
                kbs = await kb_repo.get_all(domain_id=domain_id, 
                                             category=category,
                                             is_active=is_active)

            if is_active is not None:

                return [{
                    "kb_id": kb.kb_id,
                    "domain_id": kb.domain_id,
                    "kb_name": kb.kb_name,
                    "category": kb.kb_category,
                    "descs": kb.descs,
                    "security_level": kb.security_level
                } for kb in kbs]
            
            else:
                return [{
                    "kb_id": kb.kb_id,
                    "domain_id": kb.domain_id,
                    "kb_name": kb.kb_name,
                    "category": kb.kb_category,
                    "engine": kb.engine,
                    "descs": kb.descs,
                    "models": kb.models,
                    "dbconf": kb.dbconf,
                    "is_active": kb.kb_status == 1,
                    "security_level": kb.security_level,
                    "process_priority": kb.process_priority,
                    "created_by": kb.created_by,
                    "created_at": kb.created_time,
                    "updated_by": kb.updated_by,
                    "updated_at": kb.updated_time
            } for kb in kbs]
        
        except Exception as e:
            handle_exception(e, "获取知识库列表失败")
    
    
    async def toggle_active(
            self,
            kb_id: int,
            is_active: bool,
            user_name: str
        ):
        """切换知识库状态

        Args:
            kb_id: 知识库ID
            is_active: 是否活跃
            user_name: 操作人
        """
        try:
            async with self.db_session as session:
                kb_repo = KBRepository(session)
                await kb_repo.toggle_active(kb_id=kb_id, is_active=is_active, user_name=user_name)
            logger.info(f"知识库 {kb_id} 状态切换为 {'启用' if is_active else '禁用'}")

        except Exception as e:
            handle_exception(e, "切换知识库状态失败")

    async def get_dbconf_of_kb(self, kb_id: int) -> dict:
        """根据知识库ID获取数据库配置

        Args:
            kb_id: 知识库ID

        Returns:
            dict: 数据库配置对象
        """
        try:
            async with self.db_session as session:
                kb_repo = KBRepository(session)
                kb = await kb_repo.get_by_id(kb_id)
                dbconf = kb.dbconf
                if not dbconf:
                    raise NotFoundError(f"知识库 {kb_id} 没有数据库配置")
                return dbconf
        except Exception as e:
            handle_exception(e, "获取知识库数据库配置失败")

    async def get_models_and_dbconf(self, kb_id: int) -> KBModelParams:
        """根据知识库ID获取模型和数据库配置

        Args:
            kb_id: 知识库ID

        Returns:
            dict: 模型和数据库配置对象
        """
        try:
            async with self.db_session as session:
                kb_repo = KBRepository(session)
                kb = await kb_repo.get_by_id(kb_id)
                models = kb.models
                dbconf = kb.dbconf
                if not models:
                    raise NotFoundError(f"知识库 {kb_id} 没有模型配置")
                return KBModelParams(
                    llm_model=models.get("llm_model", ""),
                    vlm_model=models.get("vlm_model", ""),
                    visual_embedding_model=models.get("visual_embedding_model", ""),
                    txt_embedding_model=models.get("txt_embedding_model", ""),
                    do_rerank=False,
                    llm_params=None,
                    rerank_top_k=None,
                    dbconf=dbconf
                )
        except Exception as e:
            handle_exception(e, "获取知识库模型和数据库配置失败")

    async def get_by_agent_and_category(self, agent_id: str, category: int) -> list[dict]:
        """根据智能体ID和知识库类型获取知识库列表

        Args:
            agent_id: 智能体ID
            category: 知识库类型

        Returns:
            list[dict]: 知识库对象列表
        """
        try:
            async with self.db_session as session:
                kb_repo = KBRepository(session)
                return await kb_repo.get_by_agent_and_category(agent_id=agent_id, category=category)
        except Exception as e:
            handle_exception(e, "获取知识库列表失败")

    async def get_name_and_desc(self, kb_id: int) -> tuple:
        """根据知识库ID获取知识库名称和描述

        Args:
            kb_id: 知识库ID

        Returns:
            tuple[str, str]: 知识库名称和描述
        """
        try:
            async with self.db_session as session:
                kb_repo = KBRepository(session)
                kb = await kb_repo.get_by_id(kb_id)
                return kb.kb_name, kb.descs
        except Exception as e:
            handle_exception(e, "获取知识库名称和描述失败")
