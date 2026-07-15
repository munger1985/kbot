# dao/repositories/workflow_repo.py — SOP 工作流仓储（Oracle 23ai 适配版）

from loguru import logger
from sqlalchemy import select, update, delete, text
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone

from ..entities import WorkflowEntity
from .base_repo import BaseRepository
from core.exceptions import APIException, DatabaseException, DataNotFoundException


class WorkflowRepository(BaseRepository[WorkflowEntity]):
    """SOP 编排流程仓储 (Oracle 23ai)"""

    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def get_by_id(self, workflow_id: str) -> WorkflowEntity:
        """根据 ID 获取完整的 Workflow"""
        try:
            result = await self.session.execute(
                select(WorkflowEntity).where(WorkflowEntity.id == workflow_id)
            )
            entity = result.scalar_one_or_none()
            if not entity:
                raise DataNotFoundException(f"未找到 ID 为 {workflow_id} 的 Workflow")
            return entity
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException(f"获取 Workflow 失败", original_error=e)

    async def create_workflow(self, entity: WorkflowEntity) -> WorkflowEntity:
        """创建流程图实体"""
        try:
            self.session.add(entity)
            await self.session.flush()
            return entity
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException(f"创建 Workflow 失败", original_error=e)

    async def update_workflow(self, workflow_id: str, update_data: dict) -> WorkflowEntity:
        """更新流程图节点、连线或配置"""
        try:
            # Oracle 不支持 UPDATE ... RETURNING，先用 UPDATE 再重新查询
            stmt = (
                update(WorkflowEntity)
                .where(WorkflowEntity.id == workflow_id)
                .values(**update_data)
            )
            result = await self.session.execute(stmt)
            if result.rowcount == 0:
                raise DataNotFoundException(f"更新失败，未找到 ID 为 {workflow_id} 的 Workflow")
            # 重新查询
            return await self.get_by_id(workflow_id)
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException(f"更新 Workflow 失败", original_error=e)

    async def delete_workflow(self, workflow_id: str) -> None:
        """物理删除 Workflow"""
        try:
            stmt = delete(WorkflowEntity).where(WorkflowEntity.id == workflow_id)
            await self.session.execute(stmt)
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException(f"删除 Workflow 失败", original_error=e)

    async def get_workflows_by_agent(
        self, agent_id: str, is_active: bool | None = None
    ) -> list[WorkflowEntity]:
        """获取某个 Agent 下所有流程"""
        try:
            query = select(WorkflowEntity).where(WorkflowEntity.agent_id == agent_id)
            if is_active is not None:
                active_flag = "1" if is_active else "0"
                query = query.where(WorkflowEntity.is_active == active_flag)
            result = await self.session.execute(query)
            return list(result.scalars().all())
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException(f"获取 Agent 流程列表失败", original_error=e)

    async def toggle_workflow(self, workflow_id: str, is_active: bool, user_id: str) -> None:
        """切换工作流状态：启用/禁用"""
        try:
            active_flag = "1" if is_active else "0"
            await self.session.execute(
                update(WorkflowEntity)
                .where(WorkflowEntity.id == workflow_id)
                .values(is_active=active_flag, updated_by=user_id,
                         updated_at=datetime.now(timezone.utc))
            )
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException(f"切换工作流状态失败: {workflow_id}", original_error=e)

    async def search_workflow(
        self, agent_id: str, query_text: str, query_vector: list[float], top_k: int = 5
    ) -> list[dict]:
        """向量检索 + 可选全文搜索（Oracle 23ai）"""
        try:
            scores: dict[str, float] = {}
            rrf_k = 60

            # Oracle Text 全文搜索（如果 available）
            if query_text.strip():
                try:
                    sql_text = text("""
                        SELECT id, 1.0 AS text_score
                        FROM kbot_agent_workflow
                        WHERE agent_id = :agent_id
                          AND (UPPER(name) LIKE UPPER(:like_q) OR UPPER(description) LIKE UPPER(:like_q))
                    """)
                    rows_text = await self.session.execute(sql_text, {
                        "agent_id": agent_id,
                        "like_q": f"%{query_text}%",
                    })
                    for idx, row in enumerate(rows_text.fetchall()):
                        scores[row[0]] = scores.get(row[0], 0) + 1.0 / (rrf_k + idx)
                except Exception as e:
                    logger.warning(f"[WorkflowRepo] 全文搜索异常: {e}")

            # Oracle 23ai 向量检索
            if query_vector:
                vec_str = ",".join(str(v) for v in query_vector)
                sql_vec = text(f"""
                    SELECT id, 1 - (embedding <=> TO_VECTOR(:vec)) AS vec_score
                    FROM kbot_agent_workflow
                    WHERE agent_id = :agent_id AND embedding IS NOT NULL
                    ORDER BY embedding <=> TO_VECTOR(:vec)
                    FETCH FIRST :top_k ROWS ONLY
                """)
                rows_vec = await self.session.execute(sql_vec, {
                    "agent_id": agent_id,
                    "vec": f"[{vec_str}]",
                    "top_k": top_k * 2,
                })
                for idx, row in enumerate(rows_vec.fetchall()):
                    scores[row[0]] = scores.get(row[0], 0) + 1.0 / (rrf_k + idx)

            if not scores:
                return []

            ranked_ids = sorted(scores, key=lambda rid: -scores[rid])[:top_k]
            results = []
            for rid in ranked_ids:
                entity = await self.session.get(WorkflowEntity, rid)
                if entity:
                    results.append({
                        "_id": entity.id,
                        "_score": scores.get(rid, 0),
                        "_source": {
                            "workflow_id": entity.id,
                            "agent_id": entity.agent_id,
                            "name": entity.name,
                            "description": entity.description,
                        },
                    })
            return results

        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            logger.error(f"Workflow 检索失败: {e}")
            raise DatabaseException(f"Workflow 检索失败", original_error=e)
