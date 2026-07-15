import uuid
from loguru import logger
from typing import Any

from dao.entities import WorkflowEntity
from dao.repositories import WorkflowRepository
from core.database import db_instance
from core.exceptions import *
from utils.clients import AIModelClient
from services.basic import AgentService


class WorkflowService:
    """SOP 工作流服务 — 纯 PG 存储（已移除 ES 双写）"""

    def __init__(self):
        self.model_client = AIModelClient()
        self.agent_service = AgentService()

    @property
    def db_session(self):
        return db_instance().get_session()

    async def add_workflow(
        self,
        user_name: str,
        agent_id: int,
        name: str,
        description: str,
        nodes: dict[str, Any],
        edges: list[dict[str, Any]],
        mode: str = "guided"
    ) -> str:
        """添加 SOP 工作流：PG 单写"""
        async with self.db_session as session:
            pg_repo = WorkflowRepository(session)
            workflow_id = str(uuid.uuid4())

            # 生成向量并写入 PG
            vector = await self._get_embedding(agent_id, f"{name} {description}")

            new_workflow = WorkflowEntity(
                id=workflow_id,
                agent_id=agent_id,
                name=name,
                description=description,
                embedding=vector,
                nodes=nodes,
                edges=edges,
                mode=mode,
                is_active=False,
                created_by=user_name
            )
            try:
                await pg_repo.create_workflow(new_workflow)
            except Exception as e:
                if "duplicate key" in str(e) or "UniqueViolation" in str(e):
                    raise ConflictError(f"流程名称「{name}」已存在，请更换名称")
                raise

            logger.info(f"成功创建 SOP 工作流: {workflow_id}")
            return workflow_id

    async def update_workflow(
        self, 
        workflow_id: str, 
        nodes: dict[str, Any], 
        edges: list[dict[str, Any]], 
        user_name: str
    ) -> None:
        """更新 SOP 工作流图结构：仅 PG"""
        async with self.db_session as session:
            pg_repo = WorkflowRepository(session)
            update_data = {"nodes": nodes, "edges": edges, "updated_by": user_name}
            await pg_repo.update_workflow(workflow_id, update_data)
            logger.info(f"成功更新 SOP 工作流图结构: {workflow_id}")

    async def update_metadata(
        self,
        user_name: str,
        workflow_id: str,
        name: str | None = None,
        description: str | None = None,
        mode: str | None = None
    ) -> None:
        """更新工作流元数据：名称/描述变更时重新计算向量"""
        raw_data = {
            "name": name,
            "description": description,
            "mode": mode,
            "updated_by": user_name
        }
        update_data = {k: v for k, v in raw_data.items() if v is not None}
        if len(update_data) <= 1:
            return

        async with self.db_session as session:
            pg_repo = WorkflowRepository(session)
            updated_entity = await pg_repo.update_workflow(workflow_id, update_data)

            # 元数据变更时重新计算向量
            search_fields = ["name", "description"]
            if any(field in update_data for field in search_fields):
                new_name = updated_entity.name
                new_desc = updated_entity.description or ""
                vector = await self._get_embedding(updated_entity.agent_id, f"{new_name} {new_desc}")
                await pg_repo.update_workflow(workflow_id, {"embedding": vector})

            logger.info(f"修改工作流元数据 {workflow_id} 成功")

    async def get_workflow(self, workflow_id: str) -> dict[str, Any]:
        """根据 ID 获取工作流详情"""
        async with self.db_session as session:
            pg_repo = WorkflowRepository(session)
            workflow = await pg_repo.get_by_id(workflow_id)
            return workflow.to_dict()
        
    async def search_workflow(self, agent_id: int, query: str, top_k: int = 1) -> list[dict[str, Any]]:
        """语义搜索：ParadeDB 混合检索"""
        try:
            vector = await self._get_embedding(agent_id, query)
            async with self.db_session as session:
                pg_repo = WorkflowRepository(session)
                hits = await pg_repo.search_workflow(
                    agent_id=agent_id,
                    query_text=query,
                    query_vector=vector,
                    top_k=top_k
                )
            return [{
                "workflow_id": hit["_source"]["workflow_id"],
                "name": hit["_source"]["name"],
                "score": hit.get("_score", 0)
            } for hit in hits]
        except Exception as e:
            logger.error(f"检索 SOP 工作流失败: {e}")
            return []
        
    async def _get_embedding(self, agent_id: int, text: str) -> list[float]:
        """获取文本的向量表示"""
        model_params = await self.agent_service.get_agent_model_params(agent_id)
        embedding = await self.model_client.get_embedding(model_params.txt_embedding_model, text)
        return embedding
    
    async def toggle_workflow(self, workflow_id: str, is_active: bool, user_id: str) -> None:
        """切换工作流状态"""
        async with self.db_session as session:
            repo = WorkflowRepository(session)
            await repo.toggle_workflow(workflow_id, is_active, user_id)
            logger.info(f"成功切换工作流状态 {workflow_id} 为 {is_active}")

    async def delete_workflow(self, workflow_id: str) -> None:
        """删除工作流：PG 硬删除"""
        async with self.db_session as session:
            repo = WorkflowRepository(session)
            await repo.delete_workflow(workflow_id)
            logger.info(f"成功删除工作流 {workflow_id}")

    async def get_workflows_by_agent(self, agent_id: int, is_active: bool | None = None) -> list[dict[str, Any]]:
        """获取某个 Agent 下所有流程"""
        async with self.db_session as session:
            repo = WorkflowRepository(session)
            workflows = await repo.get_workflows_by_agent(agent_id, is_active)
            return [workflow.to_dict() for workflow in workflows]
