"""Target、Agent Binding 与策略聚合的 Repository。"""

from collections.abc import Callable, Collection
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Select, and_, case, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.application.errors import StateConflictError
from aiops_agent.entities import (
    PolicyEntity,
    TargetBindingEntity,
    TargetEntity,
    TargetSourceBindingEntity,
)
from aiops_agent.repositories._base import AIOpsRepository


class TargetRepository(AIOpsRepository):
    def __init__(
        self,
        session: AsyncSession,
        assert_active: Callable[[], None] | None = None,
    ):
        super().__init__(session, assert_active)

    async def add_target(self, entity: TargetEntity) -> TargetEntity:
        return await self._add(entity)

    async def delete_target(self, entity: TargetEntity) -> None:
        """仅由用例层在确认停用后删除无关联 Target。"""
        self._check_active()
        await self._session.delete(entity)
        await self._session.flush()

    async def add_binding(
        self, entity: TargetBindingEntity
    ) -> TargetBindingEntity:
        return await self._add(entity)

    async def add_source_binding(
        self, entity: TargetSourceBindingEntity
    ) -> TargetSourceBindingEntity:
        return await self._add(entity)

    async def target_ids_shared_by_sources(
        self,
        *,
        domain_id: int,
        source_ids: Collection[UUID],
    ) -> list[UUID]:
        """返回所有指定监控源共同映射的授权 Target，不以运行健康拦截对话。"""
        normalized = tuple(dict.fromkeys(source_ids))
        if not normalized:
            return []
        rows = await self._session.execute(
            select(TargetSourceBindingEntity.target_id)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetSourceBindingEntity.target_id,
            )
            .where(
                TargetEntity.domain_id == domain_id,
                TargetSourceBindingEntity.status == "ACTIVE",
                TargetSourceBindingEntity.diagnostic_source_id.in_(normalized),
            )
            .group_by(TargetSourceBindingEntity.target_id)
            .having(
                func.count(
                    func.distinct(
                        TargetSourceBindingEntity.diagnostic_source_id
                    )
                )
                == len(normalized)
            )
            .order_by(TargetSourceBindingEntity.target_id)
        )
        return list(rows.scalars())

    async def get_scoped(
        self,
        *,
        target_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> TargetEntity | None:
        self._check_active()
        statement: Select = select(TargetEntity).where(
            TargetEntity.target_id == target_id,
            TargetEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_scoped(
        self,
        *,
        domain_id: int,
        statuses: Collection[str] | None = None,
    ) -> list[TargetEntity]:
        self._check_active()
        statement = select(TargetEntity).where(
            TargetEntity.domain_id == domain_id,
        )
        if statuses:
            statement = statement.where(TargetEntity.status.in_(statuses))
        statement = statement.order_by(
            TargetEntity.display_name,
            TargetEntity.target_id,
        )
        return list((await self._session.execute(statement)).scalars())

    async def page_scoped(
        self,
        *,
        domain_id: int,
        statuses: Collection[str] | None,
        before_updated_at: datetime | None,
        before_id: UUID | None,
        limit: int,
    ) -> list[TargetEntity]:
        self._check_active()
        statement = select(TargetEntity).where(
            TargetEntity.domain_id == domain_id,
        )
        if statuses:
            statement = statement.where(TargetEntity.status.in_(statuses))
        if before_updated_at is not None and before_id is not None:
            statement = statement.where(
                or_(
                    TargetEntity.updated_at < before_updated_at,
                    and_(
                        TargetEntity.updated_at == before_updated_at,
                        TargetEntity.target_id < before_id,
                    ),
                )
            )
        statement = statement.order_by(
            TargetEntity.updated_at.desc(),
            TargetEntity.target_id.desc(),
        ).limit(limit)
        return list((await self._session.execute(statement)).scalars())

    async def claim_due_connectivity(
        self, *, due_before: datetime, pending_before: datetime
    ) -> TargetEntity | None:
        """锁定一个到期 Target，供多副本 Scheduler 安全发起检查。"""
        claimed_id = await self._claim_oracle_uuid(
            plsql="""
                DECLARE
                    CURSOR c_claim IS
                        SELECT TARGET_ID
                        FROM KBOT_OPS_TARGET
                        WHERE READONLY_CONNECTION_ENABLED = 1
                          AND (
                              (
                                  LAST_CONNECTIVITY_CHECK_AT IS NULL
                                  AND (
                                      CONNECTIVITY_CHECK_REQUESTED_AT IS NULL
                                      OR CONNECTIVITY_CHECK_REQUESTED_AT
                                          <= :pending_before
                                  )
                              )
                              OR (
                                  LAST_CONNECTIVITY_CHECK_AT <= :due_before
                                  AND CONNECTIVITY_STATUS <> 'CHECKING'
                              )
                              OR (
                                  CONNECTIVITY_STATUS = 'CHECKING'
                                  AND CONNECTIVITY_CHECK_REQUESTED_AT
                                      <= :pending_before
                              )
                          )
                        ORDER BY LAST_CONNECTIVITY_CHECK_AT NULLS FIRST,
                                 TARGET_ID
                        FOR UPDATE OF TARGET_ID SKIP LOCKED;
                BEGIN
                    :claimed_id := NULL;
                    OPEN c_claim;
                    FETCH c_claim INTO :claimed_id;
                    CLOSE c_claim;
                END;
            """,
            parameters={
                "due_before": due_before,
                "pending_before": pending_before,
            },
        )
        if claimed_id is None:
            return None
        entity = (
            await self._session.execute(
                select(TargetEntity).where(TargetEntity.target_id == claimed_id)
            )
        ).scalar_one_or_none()
        if entity is None:
            raise StateConflictError(f"领取后的 Target 不存在：{claimed_id}")
        return entity

    async def update_target(
        self,
        *,
        target_id: UUID,
        domain_id: int,
        expected_version: int,
        values: dict,
    ) -> bool:
        self._check_active()
        update_values = dict(values)
        update_values.update(
            {
                "row_version": TargetEntity.row_version + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        statement = (
            update(TargetEntity)
            .where(
                TargetEntity.target_id == target_id,
                TargetEntity.domain_id == domain_id,
                TargetEntity.row_version == expected_version,
            )
            .values(**update_values)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def get_agent_binding(
        self,
        *,
        target_id: UUID,
        agent_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> TargetBindingEntity | None:
        self._check_active()
        statement: Select = (
            select(TargetBindingEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetBindingEntity.target_id,
            )
            .where(
                TargetBindingEntity.target_id == target_id,
                TargetBindingEntity.agent_id == agent_id,
                TargetEntity.domain_id == domain_id,
            )
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_binding_scoped(
        self,
        *,
        binding_id: UUID,
        target_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> TargetBindingEntity | None:
        self._check_active()
        statement: Select = (
            select(TargetBindingEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetBindingEntity.target_id,
            )
            .where(
                TargetBindingEntity.binding_id == binding_id,
                TargetBindingEntity.target_id == target_id,
                TargetEntity.domain_id == domain_id,
            )
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_agent_bindings(
        self,
        *,
        target_id: UUID,
        domain_id: int,
    ) -> list[TargetBindingEntity]:
        self._check_active()
        statement = (
            select(TargetBindingEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetBindingEntity.target_id,
            )
            .where(
                TargetBindingEntity.target_id == target_id,
                TargetEntity.domain_id == domain_id,
            )
            .order_by(
                TargetBindingEntity.created_at,
                TargetBindingEntity.binding_id,
            )
        )
        return list((await self._session.execute(statement)).scalars())

    async def update_binding(
        self,
        *,
        binding_id: UUID,
        target_id: UUID,
        expected_version: int,
        values: dict,
    ) -> bool:
        self._check_active()
        update_values = dict(values)
        update_values.update(
            {
                "row_version": TargetBindingEntity.row_version + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        statement = (
            update(TargetBindingEntity)
            .where(
                TargetBindingEntity.binding_id == binding_id,
                TargetBindingEntity.target_id == target_id,
                TargetBindingEntity.row_version == expected_version,
            )
            .values(**update_values)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def list_source_bindings(
        self,
        *,
        target_id: UUID,
        domain_id: int,
        active_only: bool = True,
    ) -> list[TargetSourceBindingEntity]:
        self._check_active()
        statement = (
            select(TargetSourceBindingEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetSourceBindingEntity.target_id,
            )
            .where(
                TargetSourceBindingEntity.target_id == target_id,
                TargetEntity.domain_id == domain_id,
            )
        )
        if active_only:
            statement = statement.where(TargetSourceBindingEntity.status == "ACTIVE")
        statement = statement.order_by(
            case(
                (TargetSourceBindingEntity.role == "PRIMARY", 0),
                else_=1,
            ),
            TargetSourceBindingEntity.priority,
            TargetSourceBindingEntity.target_source_binding_id,
        )
        return list((await self._session.execute(statement)).scalars())

    async def get_source_binding_scoped(
        self,
        *,
        target_source_binding_id: UUID,
        target_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> TargetSourceBindingEntity | None:
        self._check_active()
        statement: Select = (
            select(TargetSourceBindingEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetSourceBindingEntity.target_id,
            )
            .where(
                TargetSourceBindingEntity.target_source_binding_id == target_source_binding_id,
                TargetSourceBindingEntity.target_id == target_id,
                TargetEntity.domain_id == domain_id,
            )
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_source_binding_by_locator(
        self,
        *,
        diagnostic_source_id: UUID,
        source_locator_key: str,
        lock: bool = False,
    ) -> TargetSourceBindingEntity | None:
        """同一 Source 下只允许精确外部目标映射。"""
        self._check_active()
        statement: Select = select(TargetSourceBindingEntity).where(
            TargetSourceBindingEntity.diagnostic_source_id == diagnostic_source_id,
            TargetSourceBindingEntity.source_locator_key == source_locator_key,
            TargetSourceBindingEntity.status == "ACTIVE",
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def update_monitor(
        self,
        *,
        target_source_binding_id: UUID,
        target_id: UUID,
        expected_version: int,
        values: dict,
    ) -> bool:
        self._check_active()
        update_values = dict(values)
        update_values.update(
            {
                "row_version": TargetSourceBindingEntity.row_version + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        statement = (
            update(TargetSourceBindingEntity)
            .where(
                TargetSourceBindingEntity.target_source_binding_id == target_source_binding_id,
                TargetSourceBindingEntity.target_id == target_id,
                TargetSourceBindingEntity.row_version == expected_version,
            )
            .values(**update_values)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def reduce_source_binding_health(
        self,
        *,
        target_source_binding_id: UUID,
        expected_config_version: int,
        expected_health_version: int,
        health_status: str,
        checked_at: datetime,
        last_error_code: str | None,
    ) -> bool:
        self._check_active()
        statement = (
            update(TargetSourceBindingEntity)
            .where(
                TargetSourceBindingEntity.target_source_binding_id
                == target_source_binding_id,
                TargetSourceBindingEntity.row_version
                == expected_config_version,
                TargetSourceBindingEntity.health_version
                == expected_health_version,
            )
            .values(
                health_status=health_status,
                last_health_check_at=checked_at,
                last_error_code=last_error_code,
                health_version=TargetSourceBindingEntity.health_version + 1,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def update_state(
        self,
        *,
        target_id: UUID,
        domain_id: int,
        expected_version: int,
        allowed_statuses: Collection[str],
        new_status: str,
        updated_by: str,
    ) -> bool:
        self._check_active()
        statement = (
            update(TargetEntity)
            .where(
                TargetEntity.target_id == target_id,
                TargetEntity.domain_id == domain_id,
                TargetEntity.row_version == expected_version,
                TargetEntity.status.in_(allowed_statuses),
            )
            .values(
                status=new_status,
                updated_by=updated_by,
                row_version=TargetEntity.row_version + 1,
                updated_at=datetime.now(UTC),
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def update_observed_status(
        self,
        *,
        target_id: UUID,
        observed_status: str,
        checked_at: datetime,
        last_error_code: str | None,
    ) -> bool:
        self._check_active()
        statement = (
            update(TargetEntity)
            .where(
                TargetEntity.target_id == target_id,
            )
            .values(
                observed_status=observed_status,
                last_observed_at=checked_at,
                last_error_code=last_error_code,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def update_connectivity(
        self,
        *,
        target_id: UUID,
        connectivity_check_request_id: UUID,
        expected_config_version: int,
        expected_connectivity_version: int,
        connectivity_status: str,
        checked_at: datetime,
        last_error_code: str | None,
    ) -> bool:
        """仅在配置和检查版本未变化时归并数据库连通性。"""
        self._check_active()
        statement = (
            update(TargetEntity)
            .where(
                TargetEntity.target_id == target_id,
                TargetEntity.connectivity_check_request_id
                == connectivity_check_request_id,
                TargetEntity.row_version == expected_config_version,
                TargetEntity.connectivity_version
                == expected_connectivity_version,
            )
            .values(
                connectivity_status=connectivity_status,
                last_connectivity_check_at=checked_at,
                last_connectivity_success_at=(
                    checked_at
                    if connectivity_status in {"CONNECTED", "DEGRADED"}
                    else TargetEntity.last_connectivity_success_at
                ),
                last_error_code=last_error_code,
                connectivity_version=TargetEntity.connectivity_version + 1,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1


class PolicyRepository(AIOpsRepository):
    async def add(self, entity: PolicyEntity) -> PolicyEntity:
        return await self._add(entity)

    async def get_active(
        self,
        *,
        domain_id: int,
        policy_key: str,
        lock: bool = False,
    ) -> PolicyEntity | None:
        self._check_active()
        statement: Select = select(PolicyEntity).where(
            PolicyEntity.domain_id == domain_id,
            PolicyEntity.policy_key == policy_key,
            PolicyEntity.status == "ACTIVE",
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_scoped(
        self,
        *,
        policy_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> PolicyEntity | None:
        self._check_active()
        statement: Select = select(PolicyEntity).where(
            PolicyEntity.policy_id == policy_id,
            PolicyEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def lock_versions(
        self,
        *,
        domain_id: int,
        policy_key: str,
    ) -> list[PolicyEntity]:
        self._check_active()
        statement = (
            select(PolicyEntity)
            .where(
                PolicyEntity.domain_id == domain_id,
                PolicyEntity.policy_key == policy_key,
            )
            .order_by(PolicyEntity.version_no)
            .with_for_update()
        )
        return list((await self._session.execute(statement)).scalars())

    async def page_scoped(
        self,
        *,
        domain_id: int,
        statuses: Collection[str] | None,
        before_updated_at: datetime | None,
        before_id: UUID | None,
        limit: int,
    ) -> list[PolicyEntity]:
        self._check_active()
        statement = select(PolicyEntity).where(
            PolicyEntity.domain_id == domain_id,
        )
        if statuses:
            statement = statement.where(PolicyEntity.status.in_(statuses))
        if before_updated_at is not None and before_id is not None:
            statement = statement.where(
                or_(
                    PolicyEntity.updated_at < before_updated_at,
                    and_(
                        PolicyEntity.updated_at == before_updated_at,
                        PolicyEntity.policy_id < before_id,
                    ),
                )
            )
        statement = statement.order_by(
            PolicyEntity.updated_at.desc(),
            PolicyEntity.policy_id.desc(),
        ).limit(limit)
        return list((await self._session.execute(statement)).scalars())

    async def transition_status(
        self,
        *,
        policy_id: UUID,
        expected_version: int,
        allowed_statuses: Collection[str],
        new_status: str,
        updated_by: str,
        effective_at: datetime | None = None,
        retired_at: datetime | None = None,
    ) -> bool:
        self._check_active()
        values = {
            "status": new_status,
            "updated_by": updated_by,
            "row_version": PolicyEntity.row_version + 1,
            "updated_at": datetime.now(UTC),
        }
        if effective_at is not None:
            values["effective_at"] = effective_at
        if retired_at is not None:
            values["retired_at"] = retired_at
        statement = (
            update(PolicyEntity)
            .where(
                PolicyEntity.policy_id == policy_id,
                PolicyEntity.row_version == expected_version,
                PolicyEntity.status.in_(allowed_statuses),
            )
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1
