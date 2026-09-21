"""Target 运维事实的校验、确认写入和撤回。"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from aiops_agent.application.configuration.common import (
    ConfigurationScope,
    add_configuration_event,
)
from aiops_agent.application.errors import (
    resource_not_found,
    row_version_changed,
    state_conflict,
    validation_failed,
)
from aiops_agent.entities import TargetFactEntity
from aiops_agent.persistence import AIOpsUnitOfWork
from platform_core.identity import uuid7


ALLOWED_FACT_TYPES = frozenset(
    {"ASM_DISKGROUP", "DATAFILE_PATH", "TABLESPACE_PLACEMENT"}
)
FACT_VALUE_KEY_FIELDS = {
    "ASM_DISKGROUP": "diskgroup_name",
    "DATAFILE_PATH": "directory",
    "TABLESPACE_PLACEMENT": "tablespace_name",
}


def normalize_fact_value(
    *,
    fact_type: str,
    fact_key: str,
    fact_value: dict[str, Any] | None,
) -> tuple[str, dict[str, str]]:
    """只持久化白名单字段，并要求 fact_key 与关键字段一致。"""
    if fact_type not in ALLOWED_FACT_TYPES:
        raise validation_failed("不支持的 Target 事实类型")
    key_field = FACT_VALUE_KEY_FIELDS[fact_type]
    if not isinstance(fact_value, dict):
        raise validation_failed("事实值必须是对象")
    raw = fact_value.get(key_field)
    if raw is None or not str(raw).strip():
        raise validation_failed(f"事实值缺少必填字段 {key_field}")
    normalized_key = str(raw).strip()
    requested_key = str(fact_key or "").strip()
    if not requested_key:
        raise validation_failed("事实键不能为空")
    if normalized_key != requested_key:
        raise validation_failed("事实键必须与事实值中的关键字段一致")
    if len(normalized_key) > 256:
        raise validation_failed("事实键超过长度限制")
    return normalized_key, {key_field: normalized_key}


async def list_active_facts(
    *,
    uow: AIOpsUnitOfWork,
    target_id: UUID,
    domain_id: int,
) -> list[TargetFactEntity]:
    assert uow.targets is not None
    return await uow.targets.list_target_facts(
        target_id=target_id,
        domain_id=domain_id,
        active_only=True,
    )


async def create_confirmed_fact(
    *,
    uow: AIOpsUnitOfWork,
    scope: ConfigurationScope,
    target_id: UUID,
    fact_type: str,
    fact_key: str,
    fact_value: dict[str, Any] | None,
    now: datetime,
    details: dict[str, Any] | None = None,
) -> TargetFactEntity:
    """把人工确认的运维事实写入 Target，不生成可执行 SQL。"""
    assert uow.targets is not None
    normalized_key, normalized_value = normalize_fact_value(
        fact_type=fact_type,
        fact_key=fact_key,
        fact_value=fact_value,
    )
    target = await uow.targets.get_scoped(
        target_id=target_id,
        domain_id=scope.domain_id,
        lock=True,
    )
    if target is None:
        raise resource_not_found("Target")
    existing = await uow.targets.get_active_target_fact(
        target_id=target_id,
        domain_id=scope.domain_id,
        fact_type=fact_type,
        fact_key=normalized_key,
        lock=True,
    )
    if existing is not None:
        raise state_conflict("配置自然键已存在或并发创建冲突")
    entity = TargetFactEntity(
        target_fact_id=uuid7(),
        target_id=target_id,
        domain_id=scope.domain_id,
        fact_type=fact_type,
        fact_key=normalized_key,
        fact_value=normalized_value,
        source="MANUAL_CONFIRMED",
        status="ACTIVE",
        confirmed_by=scope.actor_id,
        confirmed_at=now,
        row_version=1,
        created_by=scope.actor_id,
        updated_by=scope.actor_id,
        created_at=now,
        updated_at=now,
    )
    await uow.targets.add_target_fact(entity)
    event_details = {"target_id": str(target_id)}
    if details:
        event_details.update(details)
    await add_configuration_event(
        uow=uow,
        scope=scope,
        aggregate_type="TARGET_FACT",
        aggregate_id=entity.target_fact_id,
        event_type="TARGET_FACT_CREATED",
        row_version=1,
        details=event_details,
    )
    return entity


async def retire_fact(
    *,
    uow: AIOpsUnitOfWork,
    scope: ConfigurationScope,
    target_id: UUID,
    fact_id: UUID,
    expected_version: int,
    now: datetime,
) -> TargetFactEntity:
    """撤回已确认事实，只改状态不物理删除。"""
    assert uow.targets is not None
    entity = await uow.targets.get_target_fact_scoped(
        fact_id=fact_id,
        target_id=target_id,
        domain_id=scope.domain_id,
        lock=True,
    )
    if entity is None:
        raise resource_not_found("Target Fact")
    if int(entity.row_version) != expected_version:
        raise row_version_changed()
    if entity.status != "ACTIVE":
        raise state_conflict(f"Target Fact 不能从 {entity.status} 撤回")
    entity.status = "RETIRED"
    entity.retired_by = scope.actor_id
    entity.retired_at = now
    entity.updated_by = scope.actor_id
    entity.updated_at = now
    await add_configuration_event(
        uow=uow,
        scope=scope,
        aggregate_type="TARGET_FACT",
        aggregate_id=entity.target_fact_id,
        event_type="TARGET_FACT_RETIRED",
        row_version=int(entity.row_version),
        details={"target_id": str(target_id)},
    )
    return entity
