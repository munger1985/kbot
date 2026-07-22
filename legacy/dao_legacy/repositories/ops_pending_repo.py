"""AIOps HITL 挂起请求仓库"""

import json
from datetime import datetime, timezone
from typing import Any
from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .base_repo import BaseRepository
from dao.entities.ops_pending import OpsPendingRequestEntity


class PendingRequestRepository(BaseRepository[OpsPendingRequestEntity]):
    """HITL 挂起请求的 CRUD 操作"""

    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def create(self, data: dict[str, Any]) -> OpsPendingRequestEntity:
        """创建挂起请求记录"""
        entity = OpsPendingRequestEntity(
            request_id=data["request_id"],
            session_id=data["session_id"],
            user_id=data["user_id"],
            agent_id=data["agent_id"],
            instance_id=data["instance_id"],
            entry_id=data["entry_id"],
            suspend_reason=data.get("suspend_reason", ""),
            user_prompt=data.get("user_prompt", ""),
            sql_to_run=data.get("sql_to_run"),
            expected_fields=data.get("expected_fields"),
            suspended_by_skill=data.get("suspended_by_skill", "unknown"),
            current_step_index=data.get("current_step_index", 0),
            completed_steps=data.get("completed_steps"),
            accumulated_results=data.get("accumulated_results"),
            pending_variables=data.get("pending_variables"),
            hitl_history=data.get("hitl_history"),
            runtime_plan=data.get("runtime_plan"),
            status=data.get("status", "pending"),
            timeout_at=data.get("timeout_at"),
        )
        self.session.add(entity)
        await self.session.flush()
        logger.info(f"[PendingRepo] 创建挂起请求: {entity.request_id}")
        return entity

    async def get_by_request_id(self, request_id: str) -> dict[str, Any] | None:
        """根据 request_id 获取挂起请求（返回 dict）"""
        stmt = select(OpsPendingRequestEntity).where(
            OpsPendingRequestEntity.request_id == request_id
        )
        result = await self.session.execute(stmt)
        entity = result.scalar_one_or_none()
        if entity is None:
            return None

        return {
            "request_id": entity.request_id,
            "session_id": entity.session_id,
            "user_id": entity.user_id,
            "agent_id": entity.agent_id,
            "instance_id": entity.instance_id,
            "entry_id": entity.entry_id,
            "suspend_reason": entity.suspend_reason,
            "user_prompt": entity.user_prompt,
            "sql_to_run": entity.sql_to_run,
            "expected_fields": entity.expected_fields,
            "suspended_by_skill": entity.suspended_by_skill,
            "current_step_index": entity.current_step_index,
            "completed_steps": entity.completed_steps,
            "accumulated_results": entity.accumulated_results,
            "pending_variables": entity.pending_variables,
            "hitl_history": entity.hitl_history,
            "runtime_plan": entity.runtime_plan,
            "status": entity.status,
            "requested_at": entity.requested_at,
            "responded_at": entity.responded_at,
            "timeout_at": entity.timeout_at,
            "reminder_count": entity.reminder_count,
        }

    async def mark_answered(self, request_id: str) -> None:
        """标记挂起请求为已处理"""
        stmt = (
            update(OpsPendingRequestEntity)
            .where(OpsPendingRequestEntity.request_id == request_id)
            .values(
                status="answered",
                responded_at=datetime.now(timezone.utc),
            )
        )
        await self.session.execute(stmt)
        await self.session.flush()
        logger.info(f"[PendingRepo] 标记挂起请求为已处理: {request_id}")

    async def mark_timeout(self, request_id: str) -> None:
        """标记挂起请求为超时"""
        stmt = (
            update(OpsPendingRequestEntity)
            .where(OpsPendingRequestEntity.request_id == request_id)
            .values(status="timeout")
        )
        await self.session.execute(stmt)
        await self.session.flush()
        logger.info(f"[PendingRepo] 标记挂起请求为超时: {request_id}")

    async def mark_cancelled(self, request_id: str) -> None:
        """标记挂起请求为已取消"""
        stmt = (
            update(OpsPendingRequestEntity)
            .where(OpsPendingRequestEntity.request_id == request_id)
            .values(status="cancelled")
        )
        await self.session.execute(stmt)
        await self.session.flush()
        logger.info(f"[PendingRepo] 标记挂起请求为已取消: {request_id}")

    async def find_timeout_pending(self) -> list[dict[str, Any]]:
        """查找所有超时且尚未处理的 pending 记录"""
        stmt = select(OpsPendingRequestEntity).where(
            OpsPendingRequestEntity.status == "pending",
            OpsPendingRequestEntity.timeout_at < datetime.now(timezone.utc),
        )
        result = await self.session.execute(stmt)
        entities = result.scalars().all()
        return [await self.get_by_request_id(e.request_id) for e in entities]

    async def get_active_pending_by_session(self, session_id: str) -> dict[str, Any] | None:
        """查找某个会话当前活跃的挂起请求"""
        stmt = (
            select(OpsPendingRequestEntity)
            .where(
                OpsPendingRequestEntity.session_id == session_id,
                OpsPendingRequestEntity.status == "pending",
            )
            .order_by(OpsPendingRequestEntity.requested_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        entity = result.scalar_one_or_none()
        if entity is None:
            return None
        return await self.get_by_request_id(entity.request_id)

    async def increment_reminder(self, request_id: str) -> None:
        """催办次数 +1"""
        stmt = (
            update(OpsPendingRequestEntity)
            .where(OpsPendingRequestEntity.request_id == request_id)
            .values(reminder_count=OpsPendingRequestEntity.reminder_count + 1)
        )
        await self.session.execute(stmt)
        await self.session.flush()
