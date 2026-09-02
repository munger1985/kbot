"""Proposal、HITL、审批授权和 Execution 聚合 Repository。"""

from collections.abc import Callable, Collection
from datetime import datetime
from uuid import UUID

from sqlalchemy import Select, func, literal_column, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.entities import (
    ApprovalTokenEntity,
    ChangeProposalEntity,
    ExecutionEntity,
    HitlEntity,
    OpsRunEntity,
    TargetEntity,
)
from aiops_agent.repositories._base import AIOpsRepository


class ChangeRepository(AIOpsRepository):
    def __init__(
        self,
        session: AsyncSession,
        assert_active: Callable[[], None] | None = None,
    ):
        super().__init__(session, assert_active)

    async def add_proposal(
        self, entity: ChangeProposalEntity
    ) -> ChangeProposalEntity:
        return await self._add(entity)

    async def count_agent_target_executions_since(
        self,
        *,
        agent_id: UUID,
        target_id: UUID,
        since: datetime,
    ) -> int:
        value = await self._session.scalar(
            select(func.count(ExecutionEntity.execution_id))
            .join(
                OpsRunEntity,
                OpsRunEntity.ops_run_id == ExecutionEntity.ops_run_id,
            )
            .where(
                OpsRunEntity.agent_id == agent_id,
                ExecutionEntity.target_id == target_id,
                ExecutionEntity.created_at >= since,
            )
        )
        return int(value or 0)

    async def add_hitl(self, entity: HitlEntity) -> HitlEntity:
        return await self._add(entity)

    async def get_hitl_by_idempotency(
        self,
        *,
        ops_run_id: UUID,
        idempotency_key: str,
    ) -> HitlEntity | None:
        self._check_active()
        statement = select(HitlEntity).where(
            HitlEntity.ops_run_id == ops_run_id,
            HitlEntity.idempotency_key == idempotency_key,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def add_approval_token(
        self, entity: ApprovalTokenEntity
    ) -> ApprovalTokenEntity:
        return await self._add(entity)

    async def add_execution(
        self, entity: ExecutionEntity
    ) -> ExecutionEntity:
        return await self._add(entity)

    async def get_approval_token(
        self,
        *,
        approval_token_id: UUID,
        lock: bool = False,
    ) -> ApprovalTokenEntity | None:
        self._check_active()
        statement: Select = select(ApprovalTokenEntity).where(
            ApprovalTokenEntity.approval_token_id == approval_token_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_approval_token_by_proposal(
        self,
        *,
        proposal_id: UUID,
        lock: bool = False,
    ) -> ApprovalTokenEntity | None:
        self._check_active()
        statement: Select = select(ApprovalTokenEntity).where(
            ApprovalTokenEntity.proposal_id == proposal_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_proposal_scoped(
        self,
        *,
        proposal_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> ChangeProposalEntity | None:
        self._check_active()
        statement: Select = (
            select(ChangeProposalEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == ChangeProposalEntity.target_id,
            )
            .where(
                ChangeProposalEntity.proposal_id == proposal_id,
                TargetEntity.domain_id == domain_id,
            )
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def page_proposals(
        self, *, domain_id: int, target_id: UUID | None = None,
        status: str | None = None, agent_ids: Collection[UUID] = (),
        before_created_at: datetime | None = None,
        before_id: UUID | None = None, limit: int = 51,
    ) -> list[ChangeProposalEntity]:
        """分页读取 Domain 内变更建议，并执行 Agent 授权过滤。"""
        self._check_active()
        statement = select(ChangeProposalEntity).join(
            TargetEntity, TargetEntity.target_id == ChangeProposalEntity.target_id
        ).join(OpsRunEntity, OpsRunEntity.ops_run_id == ChangeProposalEntity.ops_run_id).where(
            TargetEntity.domain_id == domain_id
        )
        if target_id is not None:
            statement = statement.where(ChangeProposalEntity.target_id == target_id)
        if status is not None:
            statement = statement.where(ChangeProposalEntity.status == status)
        if agent_ids:
            statement = statement.where(OpsRunEntity.agent_id.in_(tuple(agent_ids)))
        if before_created_at is not None and before_id is not None:
            statement = statement.where(or_(
                ChangeProposalEntity.created_at < before_created_at,
                (ChangeProposalEntity.created_at == before_created_at)
                & (ChangeProposalEntity.proposal_id < before_id),
            ))
        statement = statement.order_by(
            ChangeProposalEntity.created_at.desc(), ChangeProposalEntity.proposal_id.desc()
        ).limit(limit)
        return list((await self._session.execute(statement)).scalars())

    async def get_proposal(
        self, *, proposal_id: UUID, lock: bool = False
    ) -> ChangeProposalEntity | None:
        self._check_active()
        statement: Select = select(ChangeProposalEntity).where(
            ChangeProposalEntity.proposal_id == proposal_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_proposal_by_ordinal(
        self,
        *,
        ops_run_id: UUID,
        solution_group_key: str,
        command_ordinal: int,
    ) -> ChangeProposalEntity | None:
        """读取动作组中的单个序号，防止验证重放重复释放后续动作。"""
        self._check_active()
        statement = select(ChangeProposalEntity).where(
            ChangeProposalEntity.ops_run_id == ops_run_id,
            ChangeProposalEntity.solution_group_key == solution_group_key,
            ChangeProposalEntity.command_ordinal == command_ordinal,
            ChangeProposalEntity.proposal_version == 1,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_hitl_scoped(
        self,
        *,
        hitl_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> HitlEntity | None:
        self._check_active()
        statement: Select = (
            select(HitlEntity)
            .join(
                OpsRunEntity,
                OpsRunEntity.ops_run_id == HitlEntity.ops_run_id,
            )
            .join(
                TargetEntity,
                TargetEntity.target_id == OpsRunEntity.target_id,
            )
            .where(
                HitlEntity.hitl_id == hitl_id,
                TargetEntity.domain_id == domain_id,
            )
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_pending_hitl(
        self,
        *,
        ops_task_id: UUID,
        request_type: str,
        lock: bool = False,
    ) -> HitlEntity | None:
        self._check_active()
        statement: Select = select(HitlEntity).where(
            HitlEntity.ops_task_id == ops_task_id,
            HitlEntity.request_type == request_type,
            HitlEntity.status == "PENDING",
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_pending_hitl_for_run(
        self,
        *,
        ops_run_id: UUID,
        assignee_user_id: str,
        lock: bool = False,
    ) -> HitlEntity | None:
        self._check_active()
        statement: Select = (
            select(HitlEntity)
            .where(
                HitlEntity.ops_run_id == ops_run_id,
                HitlEntity.assignee_user_id == assignee_user_id,
                HitlEntity.status == "PENDING",
            )
            .order_by(HitlEntity.requested_at.desc())
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalars().first()

    async def find_expired_hitl(self) -> HitlEntity | None:
        self._check_active()
        statement = (
            select(HitlEntity)
            .where(
                HitlEntity.status == "PENDING",
                HitlEntity.expires_at
                <= literal_column("SYSTIMESTAMP"),
            )
            .order_by(HitlEntity.expires_at, HitlEntity.hitl_id)
            .limit(1)
        )
        return (await self._session.execute(statement)).scalars().first()

    async def find_expired_proposal(
        self, *, now: datetime
    ) -> ChangeProposalEntity | None:
        """返回一个待收敛 Proposal；加锁顺序仍由应用层控制。"""
        self._check_active()
        statement = (
            select(ChangeProposalEntity)
            .where(
                ChangeProposalEntity.status.in_(
                    ("ADVISORY_READY", "PENDING_APPROVAL")
                ),
                ChangeProposalEntity.expires_at.is_not(None),
                ChangeProposalEntity.expires_at <= now,
            )
            .order_by(
                ChangeProposalEntity.expires_at,
                ChangeProposalEntity.proposal_id,
            )
            .limit(1)
        )
        return (await self._session.execute(statement)).scalars().first()

    async def get_hitl(
        self, *, hitl_id: UUID, lock: bool = False
    ) -> HitlEntity | None:
        self._check_active()
        statement: Select = select(HitlEntity).where(
            HitlEntity.hitl_id == hitl_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def transition_proposal(
        self,
        *,
        proposal_id: UUID,
        expected_version: int,
        allowed_statuses: Collection[str],
        new_status: str,
        now: datetime,
    ) -> bool:
        self._check_active()
        statement = (
            update(ChangeProposalEntity)
            .where(
                ChangeProposalEntity.proposal_id == proposal_id,
                ChangeProposalEntity.row_version == expected_version,
                ChangeProposalEntity.status.in_(allowed_statuses),
            )
            .values(
                status=new_status,
                updated_at=now,
                row_version=ChangeProposalEntity.row_version + 1,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def answer_hitl(
        self,
        *,
        hitl_id: UUID,
        expected_version: int,
        allowed_statuses: Collection[str],
        new_status: str,
        responded_by: str,
        responded_at: datetime,
        response_json: dict | None,
        response_uri: str | None,
        response_hash: str | None,
    ) -> bool:
        self._check_active()
        statement = (
            update(HitlEntity)
            .where(
                HitlEntity.hitl_id == hitl_id,
                HitlEntity.row_version == expected_version,
                HitlEntity.status.in_(allowed_statuses),
            )
            .values(
                status=new_status,
                responded_by=responded_by,
                responded_at=responded_at,
                response_json=response_json,
                response_uri=response_uri,
                response_hash=response_hash,
                row_version=HitlEntity.row_version + 1,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def consume_approval_token(
        self,
        *,
        approval_token_id: UUID,
        expected_version: int,
        now: datetime,
    ) -> bool:
        self._check_active()
        statement = (
            update(ApprovalTokenEntity)
            .where(
                ApprovalTokenEntity.approval_token_id == approval_token_id,
                ApprovalTokenEntity.row_version == expected_version,
                ApprovalTokenEntity.status == "ISSUED",
                ApprovalTokenEntity.expires_at > now,
            )
            .values(
                status="CONSUMED",
                consumed_at=now,
                row_version=ApprovalTokenEntity.row_version + 1,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def get_execution_by_request(
        self,
        *,
        executor_request_id: str,
        lock: bool = False,
    ) -> ExecutionEntity | None:
        self._check_active()
        statement: Select = select(ExecutionEntity).where(
            ExecutionEntity.executor_request_id == executor_request_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_execution(
        self, *, execution_id: UUID, lock: bool = False
    ) -> ExecutionEntity | None:
        self._check_active()
        statement: Select = select(ExecutionEntity).where(
            ExecutionEntity.execution_id == execution_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_execution_by_idempotency(
        self, *, idempotency_key: str
    ) -> ExecutionEntity | None:
        self._check_active()
        statement = select(ExecutionEntity).where(
            ExecutionEntity.idempotency_key == idempotency_key
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_execution_by_proposal(
        self, *, proposal_id: UUID, lock: bool = False
    ) -> ExecutionEntity | None:
        self._check_active()
        statement: Select = select(ExecutionEntity).where(
            ExecutionEntity.proposal_id == proposal_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_active_execution_for_target(
        self,
        *,
        target_id: UUID,
        exclude_execution_id: UUID,
    ) -> ExecutionEntity | None:
        self._check_active()
        statement = (
            select(ExecutionEntity)
            .where(
                ExecutionEntity.target_id == target_id,
                ExecutionEntity.execution_id != exclude_execution_id,
                ExecutionEntity.status.in_(("SUBMITTED", "RUNNING")),
            )
            .order_by(ExecutionEntity.created_at)
            .limit(1)
        )
        return (await self._session.execute(statement)).scalars().first()

    async def find_due_execution(
        self, *, now: datetime
    ) -> ExecutionEntity | None:
        """返回一个已越过执行截止时间的非终态 Execution。"""
        self._check_active()
        statement = (
            select(ExecutionEntity)
            .where(
                ExecutionEntity.status.in_(
                    ("CREATED", "SUBMITTED", "RUNNING")
                ),
                ExecutionEntity.deadline_at <= now,
            )
            .order_by(
                ExecutionEntity.deadline_at,
                ExecutionEntity.execution_id,
            )
            .limit(1)
        )
        return (await self._session.execute(statement)).scalars().first()

    async def apply_execution_status(
        self,
        *,
        execution_id: UUID,
        incoming_status_version: int,
        new_status: str,
        result_artifact_id: UUID | None,
        result_hash: str | None,
        started_at: datetime | None,
        completed_at: datetime | None,
        error_code: str | None,
        error_message: str | None,
        now: datetime,
    ) -> bool:
        """只接受单调递增的 Executor 状态版本，拒绝乱序回调。"""
        self._check_active()
        statement = (
            update(ExecutionEntity)
            .where(
                ExecutionEntity.execution_id == execution_id,
                ExecutionEntity.status_version < incoming_status_version,
            )
            .values(
                status=new_status,
                status_version=incoming_status_version,
                result_artifact_id=result_artifact_id,
                result_hash=result_hash,
                started_at=started_at,
                completed_at=completed_at,
                error_code=error_code,
                error_message=error_message,
                row_version=ExecutionEntity.row_version + 1,
                updated_at=now,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1
