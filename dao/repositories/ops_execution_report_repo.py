"""AIOps 执行报告仓库 (Oracle)"""

from typing import Any
from loguru import logger
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from dao.entities.ops_execution_report import OpsExecutionReportEntity


class OpsExecutionReportRepository:
    """AIOps 执行报告 CRUD 操作 (Oracle)"""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, data: dict[str, Any]) -> OpsExecutionReportEntity:
        """创建执行报告记录"""
        entity = OpsExecutionReportEntity(
            entry_id=data["entry_id"],
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            agent_id=data["agent_id"],
            instance_id=data["instance_id"],
            instance_name=data.get("instance_name", ""),
            db_type=data["db_type"],
            environment=data.get("environment", "prod"),
            trigger_type=data.get("trigger_type", "manual"),
            original_question=data.get("original_question", ""),
            diagnosis_summary=data.get("diagnosis_summary", ""),
            actions_executed=data.get("actions_executed") or [],
            pre_snapshot=data.get("pre_snapshot"),
            post_snapshot=data.get("post_snapshot"),
            health_check_result=data.get("health_check_result"),
            rollback_info=data.get("rollback_info"),
            verification_status=data.get("verification_status", "skipped"),
            report_content=data.get("report_content", ""),
            recommendations=data.get("recommendations", ""),
            total_duration_seconds=data.get("total_duration_seconds", 0),
        )
        self._session.add(entity)
        await self._session.flush()
        logger.debug(
            f"[OpsReportRepo] 报告已创建: id={entity.id}, "
            f"instance={entity.instance_name}, status={entity.verification_status}"
        )
        return entity

    async def get_by_entry_id(self, entry_id: str) -> OpsExecutionReportEntity | None:
        """按 entry_id 查询报告"""
        stmt = (
            select(OpsExecutionReportEntity)
            .where(OpsExecutionReportEntity.entry_id == entry_id)
            .order_by(desc(OpsExecutionReportEntity.created_at))
            .fetch(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_instance(
        self, instance_id: str, limit: int = 20,
    ) -> list[OpsExecutionReportEntity]:
        """查询某实例的执行报告列表 (按时间倒序)"""
        stmt = (
            select(OpsExecutionReportEntity)
            .where(OpsExecutionReportEntity.instance_id == instance_id)
            .order_by(desc(OpsExecutionReportEntity.created_at))
            .fetch(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_session(
        self, session_id: str, limit: int = 10,
    ) -> list[OpsExecutionReportEntity]:
        """查询某会话的执行报告列表"""
        stmt = (
            select(OpsExecutionReportEntity)
            .where(OpsExecutionReportEntity.session_id == session_id)
            .order_by(desc(OpsExecutionReportEntity.created_at))
            .fetch(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
