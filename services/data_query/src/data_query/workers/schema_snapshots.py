"""可追溯的数据库结构发现与对象级采集 Worker。

Worker 只读取系统目录。首次任务发现表/视图并等待管理员选择；确认后逐对象采集，
因此单表失败不会丢失同批次其他对象的成果。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from loguru import logger

from data_query.adapters import DatabaseCredentialService
from data_query.connectors.schema_introspector import DatabaseSchemaIntrospector
from data_query.contracts import DataSourceEndpoint
from data_query.entities import SchemaSnapshotObjectEntity
from data_query.persistence import DataQueryUnitOfWork
from data_query.application.notifications import publish_data_query_notification


@dataclass(frozen=True)
class _SourceContext:
    snapshot_id: UUID
    data_source_id: UUID
    source_type: str
    endpoint: DataSourceEndpoint
    credential_id: UUID
    domain_id: int


@dataclass(frozen=True)
class _ObjectContext(_SourceContext):
    object_id: UUID
    schema_name: str
    object_name: str
    object_type: str


class SchemaSnapshotWorker:
    MAX_DISCOVERED_OBJECTS = 10_000
    def __init__(
        self, *, uow_factory: Callable[[], DataQueryUnitOfWork],
        credential_service: DatabaseCredentialService,
        introspector: DatabaseSchemaIntrospector,
    ) -> None:
        self._uow_factory = uow_factory
        self._credential_service = credential_service
        self._introspector = introspector

    async def process_one(self) -> bool:
        discovery = await self._claim_discovery()
        if discovery is not None:
            await self._process_discovery(discovery)
            return True
        selected_object = await self._claim_object()
        if selected_object is not None:
            await self._process_object(selected_object)
            return True
        return False

    async def run_forever(self, *, interval_seconds: float, stop: asyncio.Event) -> None:
        await self._recover_stale_work()
        while not stop.is_set():
            try:
                if await self.process_one():
                    continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Data Query 结构采集循环发生未处理异常；Worker 将继续处理后续任务")
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
            except TimeoutError:
                pass

    async def _recover_stale_work(self) -> None:
        """进程重启后重新排队超出任务硬超时的遗留工作。"""
        stale_before = datetime.now(UTC) - timedelta(minutes=2)
        async with self._uow_factory() as uow:
            assert uow.schema_snapshots and uow.schema_snapshot_objects
            snapshots = await uow.schema_snapshots.requeue_stale_discoveries(
                stale_before=stale_before
            )
            objects = await uow.schema_snapshot_objects.requeue_stale_captures(
                stale_before=stale_before
            )
            await uow.commit()
        if snapshots or objects:
            logger.warning(
                "已恢复遗留结构采集任务 | snapshots={} | objects={}",
                snapshots,
                objects,
            )

    async def _claim_discovery(self) -> _SourceContext | None:
        async with self._uow_factory() as uow:
            assert uow.schema_snapshots and uow.data_sources
            snapshot = await uow.schema_snapshots.claim_next_requested()
            if snapshot is None:
                await uow.commit()
                return None
            source = await uow.data_sources.get_by_id(data_source_id=snapshot.data_source_id)
            context = self._context(snapshot.schema_snapshot_id, source)
            if context is None:
                snapshot.status = "FAILED"
                snapshot.error_code = "DATA_SOURCE_CONFIGURATION_INVALID"
                snapshot.completed_at = datetime.now(UTC)
            await uow.commit()
            return context

    async def _claim_object(self) -> _ObjectContext | None:
        async with self._uow_factory() as uow:
            assert uow.schema_snapshot_objects and uow.schema_snapshots and uow.data_sources
            item = await uow.schema_snapshot_objects.claim_next_selected()
            if item is None:
                await uow.commit()
                return None
            snapshot = await uow.schema_snapshots.get_by_id(schema_snapshot_id=item.schema_snapshot_id)
            source = None if snapshot is None else await uow.data_sources.get_by_id(data_source_id=snapshot.data_source_id)
            base = None if snapshot is None else self._context(snapshot.schema_snapshot_id, source)
            if base is None:
                item.status = "FAILED"
                item.error_code = "DATA_SOURCE_CONFIGURATION_INVALID"
                item.completed_at = datetime.now(UTC)
                await uow.commit()
                return None
            item.started_at = datetime.now(UTC)
            await uow.commit()
            return _ObjectContext(
                **base.__dict__, object_id=item.schema_snapshot_object_id,
                schema_name=item.schema_name, object_name=item.object_name,
                object_type=item.object_type,
            )

    @staticmethod
    def _context(snapshot_id: UUID, source: Any) -> _SourceContext | None:
        if source is None:
            return None
        try:
            endpoint = DataSourceEndpoint.model_validate(source.configuration_json)
        except ValueError:
            return None
        return _SourceContext(
            snapshot_id,
            source.data_source_id,
            source.source_type,
            endpoint,
            source.credential_id,
            int(source.domain_id),
        )

    async def _process_discovery(self, context: _SourceContext) -> None:
        try:
            username, password = await self._credential_service.read_database_credentials(
                credential_id=context.credential_id,
                domain_id=context.domain_id,
                data_source_id=context.data_source_id,
            )
            objects, version = await asyncio.wait_for(
                self._introspector.discover(context, username, password), timeout=60
            )
            if len(objects) > self.MAX_DISCOVERED_OBJECTS:
                raise ValueError("SCHEMA_DISCOVERY_OBJECT_LIMIT_EXCEEDED")
            async with self._uow_factory() as uow:
                assert uow.schema_snapshots and uow.schema_snapshot_objects and uow.data_sources
                snapshot = await uow.schema_snapshots.get_by_id(schema_snapshot_id=context.snapshot_id)
                source = await uow.data_sources.get_by_id(data_source_id=context.data_source_id)
                if snapshot is not None and snapshot.status == "DISCOVERING":
                    await uow.schema_snapshot_objects.add_all([
                        SchemaSnapshotObjectEntity(
                            schema_snapshot_id=context.snapshot_id,
                            schema_name=schema, object_name=name, object_type=kind,
                        )
                        for schema, name, kind in objects
                    ])
                    snapshot.status = "WAITING_SELECTION" if objects else "FAILED"
                    snapshot.error_code = None if objects else "NO_DISCOVERED_OBJECTS"
                    snapshot.error_message = None if objects else "所选 Schema 中没有当前账号可访问的表或视图。"
                    snapshot.completed_at = None if objects else datetime.now(UTC)
                    snapshot.connector_version = version
                    snapshot.capabilities_json = {
                        "connector": context.source_type,
                        "database_version": version,
                        "discovered_count": len(objects),
                    }
                if source is not None:
                    source.status = "ACTIVE"
                    source.error_code = None
                    source.error_message = None
                if snapshot is not None and source is not None:
                    event_type = (
                        "data_query.schema.selection_required"
                        if objects else "data_query.schema.capture_failed"
                    )
                    await publish_data_query_notification(
                        uow=uow, event_type=event_type,
                        event_key=f"{snapshot.schema_snapshot_id}:discovery-terminal",
                        domain_id=int(source.domain_id), actor_id=source.created_by,
                        resource_type="schema_snapshot",
                        resource_id=str(snapshot.schema_snapshot_id),
                        resource_name=source.display_name,
                        correlation_id=str(snapshot.schema_snapshot_id),
                        operation_id=str(snapshot.schema_snapshot_id),
                        summary=(
                            "数据库对象发现完成，请选择采集范围。"
                            if objects else "数据库对象发现失败。"
                        ),
                        safe_data={
                            "status": snapshot.status,
                            "error_code": snapshot.error_code,
                            "object_count": len(objects),
                        },
                    )
                await uow.commit()
        except Exception as exc:
            logger.warning(
                "数据库对象发现失败 | snapshot_id={} | error_type={}",
                context.snapshot_id,
                type(exc).__name__,
            )
            await self._fail_snapshot(context.snapshot_id, exc)

    async def _process_object(self, context: _ObjectContext) -> None:
        try:
            username, password = await self._credential_service.read_database_credentials(
                credential_id=context.credential_id,
                domain_id=context.domain_id,
                data_source_id=context.data_source_id,
            )
            metadata = await asyncio.wait_for(
                self._introspector.capture_object(context, username, password), timeout=60
            )
        except Exception as exc:
            await self._complete_object(context.object_id, success=False, error=exc)
        else:
            await self._complete_object(context.object_id, success=True, metadata=metadata)
        await self._finalize_snapshot(context.snapshot_id)

    async def _complete_object(
        self, object_id: UUID, *, success: bool,
        metadata: dict[str, object] | None = None, error: Exception | None = None,
    ) -> None:
        async with self._uow_factory() as uow:
            assert uow.schema_snapshot_objects
            item = await uow.schema_snapshot_objects.get_by_id(schema_snapshot_object_id=object_id, lock=True)
            if item is not None and item.status == "CAPTURING":
                item.status = "READY" if success else "FAILED"
                item.metadata_json = metadata if success else None
                item.error_code = None if success else self._error_code(error)
                item.error_message = None if success else self._error_message(error)
                item.completed_at = datetime.now(UTC)
            await uow.commit()

    async def _finalize_snapshot(self, snapshot_id: UUID) -> None:
        async with self._uow_factory() as uow:
            assert uow.schema_snapshots and uow.schema_snapshot_objects and uow.data_sources
            snapshot = await uow.schema_snapshots.get_by_id(schema_snapshot_id=snapshot_id)
            rows = await uow.schema_snapshot_objects.list_by_snapshot(schema_snapshot_id=snapshot_id)
            selected = [row for row in rows if row.selected]
            if snapshot is None or snapshot.status != "CAPTURING":
                await uow.commit()
                return
            ready = [row for row in selected if row.status in {"READY", "MANUAL"}]
            failed = [row for row in selected if row.status == "FAILED"]
            pending = [row for row in selected if row.status in {"QUEUED", "CAPTURING"}]
            if pending:
                await uow.commit()
                return
            if ready:
                snapshot.status = "PARTIAL_READY" if failed else "READY"
                snapshot.objects_json = {"objects": [row.metadata_json for row in ready if row.metadata_json]}
                content_hash = hashlib.sha256(
                    json.dumps(snapshot.objects_json, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                snapshot.snapshot_hash = content_hash
                await uow.schema_snapshots.supersede_previous(
                    data_source_id=snapshot.data_source_id,
                    current_snapshot_id=snapshot.schema_snapshot_id,
                )
            else:
                snapshot.status = "FAILED"
                snapshot.error_code = "ALL_SELECTED_OBJECTS_FAILED"
            snapshot.capabilities_json = {
                **snapshot.capabilities_json,
                "selected_count": len(selected), "succeeded_count": len(ready),
                "failed_count": len(failed),
                "content_hash": content_hash if ready else None,
            }
            snapshot.completed_at = datetime.now(UTC)
            source = await uow.data_sources.get_by_id(
                data_source_id=snapshot.data_source_id
            )
            if source is not None:
                succeeded = bool(ready)
                await publish_data_query_notification(
                    uow=uow,
                    event_type=(
                        "data_query.schema.capture_completed"
                        if succeeded else "data_query.schema.capture_failed"
                    ),
                    event_key=f"{snapshot.schema_snapshot_id}:capture-terminal",
                    domain_id=int(source.domain_id), actor_id=source.created_by,
                    resource_type="schema_snapshot",
                    resource_id=str(snapshot.schema_snapshot_id),
                    resource_name=source.display_name,
                    correlation_id=str(snapshot.schema_snapshot_id),
                    operation_id=str(snapshot.schema_snapshot_id),
                    summary=(
                        "数据库结构采集完成。" if succeeded else "数据库结构采集失败。"
                    ),
                    safe_data={
                        "status": snapshot.status,
                        "succeeded_count": len(ready),
                        "failed_count": len(failed),
                        "error_code": snapshot.error_code,
                    },
                )
            await uow.commit()

    async def _fail_snapshot(self, snapshot_id: UUID, error: Exception) -> None:
        async with self._uow_factory() as uow:
            assert uow.schema_snapshots and uow.data_sources
            snapshot = await uow.schema_snapshots.get_by_id(schema_snapshot_id=snapshot_id)
            if snapshot is not None:
                snapshot.status = "FAILED"
                snapshot.error_code = self._error_code(error)
                snapshot.error_message = self._error_message(error)
                snapshot.completed_at = datetime.now(UTC)
                source = await uow.data_sources.get_by_id(data_source_id=snapshot.data_source_id)
                if source is not None:
                    source.status = "FAILED"
                    source.error_code = snapshot.error_code
                    source.error_message = snapshot.error_message
                    await publish_data_query_notification(
                        uow=uow,
                        event_type="data_query.schema.capture_failed",
                        event_key=f"{snapshot.schema_snapshot_id}:worker-failed",
                        domain_id=int(source.domain_id), actor_id=source.created_by,
                        resource_type="schema_snapshot",
                        resource_id=str(snapshot.schema_snapshot_id),
                        resource_name=source.display_name,
                        correlation_id=str(snapshot.schema_snapshot_id),
                        operation_id=str(snapshot.schema_snapshot_id),
                        summary="数据库结构采集失败。",
                        safe_data={"status": "FAILED", "error_code": snapshot.error_code},
                    )
            await uow.commit()

    @staticmethod
    def _error_code(error: Exception | None) -> str:
        return DatabaseSchemaIntrospector.error_code(error)

    @staticmethod
    def _error_message(error: Exception | None) -> str:
        return DatabaseSchemaIntrospector.error_message(error)
