"""Semantic Model 发布前的纯领域校验。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
import hashlib
import json
from uuid import UUID

from data_query.contracts.management import SemanticModelDefinition
from data_query.domain import (
    SchemaSnapshotStatus,
    SemanticModelVersionStatus,
    can_transition,
)
from data_query.persistence import DataQueryUnitOfWork


class SemanticModelPublicationError(ValueError):
    """管理面应转为稳定错误码的发布前置条件失败。"""


def _definition_hash(definition: dict[str, object]) -> str:
    payload = json.dumps(definition, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def update_semantic_model_draft(
    *, uow_factory: Callable[[], DataQueryUnitOfWork], domain_id: int,
    semantic_model_id: UUID, semantic_model_version_id: UUID,
    definition: SemanticModelDefinition, expected_row_version: int,
) -> int:
    """更新当前Domain仍处于 DRAFT 的版本，并使用乐观锁防止覆盖他人修改。"""
    async with uow_factory() as uow:
        assert uow.semantic_models and uow.semantic_model_versions
        model = await uow.semantic_models.get_by_id(semantic_model_id=semantic_model_id)
        version = await uow.semantic_model_versions.get_by_id(
            semantic_model_version_id=semantic_model_version_id, lock=True,
        )
        if (
            model is None or model.domain_id != domain_id or version is None
            or version.semantic_model_id != semantic_model_id
        ):
            raise SemanticModelPublicationError("MODEL_VERSION_NOT_FOUND")
        if version.status != SemanticModelVersionStatus.DRAFT.value:
            raise SemanticModelPublicationError("MODEL_VERSION_NOT_EDITABLE")
        if int(version.row_version) != expected_row_version:
            raise SemanticModelPublicationError("ROW_VERSION_CONFLICT")
        payload = definition.model_dump(mode="json")
        version.definition_json = payload
        version.definition_hash = _definition_hash(payload)
        await uow.commit()
        return int(version.row_version)


def validate_publishable_model(
    *,
    definition: SemanticModelDefinition,
    snapshot_status: SchemaSnapshotStatus,
    snapshot_objects: Mapping[str, set[str]],
) -> None:
    """确保模型全部映射到同一个 READY Snapshot 的受控对象。

    ``snapshot_objects`` 以 ``schema.object`` 为键，值是该对象可用列名；它由
    Snapshot Worker 从受限元数据采集器生成，不能由管理请求伪造。
    """
    if snapshot_status not in {SchemaSnapshotStatus.READY, SchemaSnapshotStatus.PARTIAL_READY}:
        raise SemanticModelPublicationError("SNAPSHOT_NOT_READY")

    datasets = {item.name: item for item in definition.datasets}
    for dataset in datasets.values():
        physical_name = f"{dataset.physical_schema}.{dataset.physical_object}"
        if physical_name not in snapshot_objects:
            raise SemanticModelPublicationError(f"SNAPSHOT_OBJECT_MISSING:{physical_name}")

    for field in (*definition.dimensions, *definition.measures):
        if field.physical_column is None:
            continue
        dataset = datasets[field.dataset]
        physical_name = f"{dataset.physical_schema}.{dataset.physical_object}"
        if field.physical_column not in snapshot_objects[physical_name]:
            raise SemanticModelPublicationError(
                f"SNAPSHOT_COLUMN_MISSING:{physical_name}.{field.physical_column}"
            )


def snapshot_object_index(snapshot_objects: Mapping[str, object] | None) -> dict[str, set[str]]:
    """将 Snapshot Worker 的受控 JSON 投影转换为发布校验索引。"""
    if not snapshot_objects:
        return {}
    objects = snapshot_objects.get("objects")
    if not isinstance(objects, list):
        return {}
    result: dict[str, set[str]] = {}
    for item in objects:
        if not isinstance(item, Mapping):
            continue
        schema = item.get("schema")
        name = item.get("name")
        columns = item.get("columns")
        if not isinstance(schema, str) or not isinstance(name, str) or not isinstance(columns, list):
            continue
        result[f"{schema}.{name}"] = {column for column in columns if isinstance(column, str)}
    return result


async def publish_semantic_model_version(
    *,
    uow_factory: Callable[[], DataQueryUnitOfWork],
    domain_id: int,
    semantic_model_id: UUID,
    semantic_model_version_id: UUID,
    schema_snapshot_id: UUID,
    expected_row_version: int,
    actor_id: str,
) -> None:
    """以同一事务冻结审核事实，并替换同一模型的旧 ACTIVE 版本。"""
    async with uow_factory() as uow:
        assert uow.semantic_models and uow.semantic_model_versions and uow.schema_snapshots
        model = await uow.semantic_models.get_by_id(semantic_model_id=semantic_model_id, lock=True)
        version = await uow.semantic_model_versions.get_by_id(
            semantic_model_version_id=semantic_model_version_id, lock=True
        )
        snapshot = await uow.schema_snapshots.get_by_id(schema_snapshot_id=schema_snapshot_id)
        if model is None or model.domain_id != domain_id or version is None or snapshot is None:
            raise SemanticModelPublicationError("MODEL_OR_SNAPSHOT_NOT_FOUND")
        if version.semantic_model_id != model.semantic_model_id or version.schema_snapshot_id != snapshot.schema_snapshot_id:
            raise SemanticModelPublicationError("MODEL_VERSION_REFERENCE_MISMATCH")
        if int(version.row_version) != expected_row_version:
            raise SemanticModelPublicationError("ROW_VERSION_CONFLICT")
        current = SemanticModelVersionStatus(version.status)
        if not can_transition(current, SemanticModelVersionStatus.ACTIVE):
            raise SemanticModelPublicationError("MODEL_VERSION_NOT_REVIEWABLE")
        definition = SemanticModelDefinition.model_validate(version.definition_json)
        validate_publishable_model(
            definition=definition,
            snapshot_status=SchemaSnapshotStatus(snapshot.status),
            snapshot_objects=snapshot_object_index(snapshot.objects_json),
        )
        await uow.semantic_model_versions.retire_active_except(
            semantic_model_id=model.semantic_model_id,
            keep_version_id=version.semantic_model_version_id,
        )
        version.status = SemanticModelVersionStatus.ACTIVE.value
        version.reviewed_by = actor_id
        version.published_by = actor_id
        version.published_at = datetime.now(UTC)
        model.active_version = version.version_no
        await uow.commit()


async def submit_semantic_model_version_for_review(
    *, uow_factory: Callable[[], DataQueryUnitOfWork], domain_id: int,
    actor_id: str, semantic_model_id: UUID, semantic_model_version_id: UUID,
    expected_row_version: int,
) -> None:
    """将草稿提交审核；发布仍需单独显式操作。"""
    async with uow_factory() as uow:
        assert uow.semantic_models and uow.semantic_model_versions
        model = await uow.semantic_models.get_by_id(semantic_model_id=semantic_model_id)
        version = await uow.semantic_model_versions.get_by_id(
            semantic_model_version_id=semantic_model_version_id, lock=True
        )
        if model is None or model.domain_id != domain_id or version is None or version.semantic_model_id != semantic_model_id:
            raise SemanticModelPublicationError("MODEL_VERSION_NOT_FOUND")
        if int(version.row_version) != expected_row_version:
            raise SemanticModelPublicationError("ROW_VERSION_CONFLICT")
        if not can_transition(SemanticModelVersionStatus(version.status), SemanticModelVersionStatus.REVIEW):
            raise SemanticModelPublicationError("MODEL_VERSION_NOT_SUBMITTABLE")
        version.status = SemanticModelVersionStatus.REVIEW.value
        version.submitted_by = actor_id
        await uow.commit()


async def return_semantic_model_version_for_revision(
    *, uow_factory: Callable[[], DataQueryUnitOfWork], domain_id: int,
    actor_id: str, semantic_model_id: UUID, semantic_model_version_id: UUID,
    review_comment: str, expected_row_version: int,
) -> None:
    """把待审核版本退回为可编辑草稿，并保留本轮审核意见。"""
    async with uow_factory() as uow:
        assert uow.semantic_models and uow.semantic_model_versions
        model = await uow.semantic_models.get_by_id(semantic_model_id=semantic_model_id)
        version = await uow.semantic_model_versions.get_by_id(
            semantic_model_version_id=semantic_model_version_id, lock=True
        )
        if model is None or model.domain_id != domain_id or version is None or version.semantic_model_id != semantic_model_id:
            raise SemanticModelPublicationError("MODEL_VERSION_NOT_FOUND")
        if int(version.row_version) != expected_row_version:
            raise SemanticModelPublicationError("ROW_VERSION_CONFLICT")
        if not can_transition(SemanticModelVersionStatus(version.status), SemanticModelVersionStatus.DRAFT):
            raise SemanticModelPublicationError("MODEL_VERSION_NOT_RETURNABLE")
        version.status = SemanticModelVersionStatus.DRAFT.value
        version.review_comment = review_comment
        version.reviewed_by = actor_id
        await uow.commit()


async def retire_semantic_model_version(
    *, uow_factory: Callable[[], DataQueryUnitOfWork], domain_id: int,
    actor_id: str, semantic_model_id: UUID, semantic_model_version_id: UUID,
    expected_row_version: int,
) -> None:
    """废弃当前正式版本；历史定义保留，但不再允许新的问数运行选择它。"""
    async with uow_factory() as uow:
        assert uow.semantic_models and uow.semantic_model_versions
        model = await uow.semantic_models.get_by_id(semantic_model_id=semantic_model_id, lock=True)
        version = await uow.semantic_model_versions.get_by_id(
            semantic_model_version_id=semantic_model_version_id, lock=True,
        )
        if (
            model is None or model.domain_id != domain_id or version is None
            or version.semantic_model_id != semantic_model_id
        ):
            raise SemanticModelPublicationError("MODEL_VERSION_NOT_FOUND")
        if int(version.row_version) != expected_row_version:
            raise SemanticModelPublicationError("ROW_VERSION_CONFLICT")
        if not can_transition(
            SemanticModelVersionStatus(version.status), SemanticModelVersionStatus.RETIRED,
        ):
            raise SemanticModelPublicationError("MODEL_VERSION_NOT_RETIRABLE")
        if model.active_version != version.version_no:
            raise SemanticModelPublicationError("MODEL_ACTIVE_VERSION_MISMATCH")
        version.status = SemanticModelVersionStatus.RETIRED.value
        model.active_version = None
        model.updated_by = actor_id
        await uow.commit()
