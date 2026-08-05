"""跨服务资源组合编排；只保存恢复回执，不接管下游领域数据。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from main_api.entities import CompositionReceiptEntity
from platform_core.contracts import (
    AgentCompositionCreate,
    AgentCompositionUpdate,
    CollectionCompositionCreate,
    CollectionModelsCompositionUpdate,
    CompositionEdge,
    CompositionNode,
    CompositionReceipt,
    ResourceReferenceGraph,
    RunCompositionView,
    SemanticModelPublicationComposition,
)
from platform_core.identity import uuid7


class CompositionError(RuntimeError):
    """向公开 API 暴露稳定错误码的组合编排错误。"""

    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


Verify = Callable[[], Awaitable[dict[str, Any] | None]]
Command = Callable[[], Awaitable[None]]
Precheck = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class _ReceiptState:
    receipt_id: UUID
    domain_id: int
    actor_id: str
    operation: str
    idempotency_key: str
    request_hash: str
    status: str
    resource_type: str
    resource_id: str | None
    resource_version: str | None
    verification_json: dict[str, Any]
    error_code: str | None
    created_at: datetime
    updated_at: datetime


def _receipt_state(entity: CompositionReceiptEntity) -> _ReceiptState:
    """在关闭 UoW 前复制 Receipt，避免返回过期 ORM 实体。"""
    return _ReceiptState(
        receipt_id=entity.receipt_id,
        domain_id=int(entity.domain_id),
        actor_id=entity.actor_id,
        operation=entity.operation,
        idempotency_key=entity.idempotency_key,
        request_hash=entity.request_hash,
        status=entity.status,
        resource_type=entity.resource_type,
        resource_id=entity.resource_id,
        resource_version=entity.resource_version,
        verification_json=dict(entity.verification_json or {}),
        error_code=entity.error_code,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode()
    return sha256(encoded).hexdigest()


def _version(payload: dict[str, Any]) -> str | None:
    value = payload.get("row_version") or payload.get("version")
    return str(value) if value is not None else None


def _node(
    node_type: str,
    resource_id: object,
    source_service: str,
    payload: dict[str, Any] | None,
    *,
    availability: str = "AVAILABLE",
    attributes: dict[str, Any] | None = None,
) -> CompositionNode:
    body = payload or {}
    resolved_availability = (
        "STALE" if payload is None and availability == "AVAILABLE"
        else availability
    )
    return CompositionNode(
        node_type=node_type,
        resource_id=str(resource_id),
        source_service=source_service,
        source_version=_version(body),
        observed_at=_now(),
        availability=resolved_availability,
        status=str(body.get("status")) if body.get("status") is not None else None,
        attributes=attributes or {},
    )


def _ids(value: Any, names: set[str]) -> set[UUID]:
    found: set[UUID] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in names:
                candidates = item if isinstance(item, (list, tuple)) else [item]
                for candidate in candidates:
                    try:
                        found.add(UUID(str(candidate)))
                    except (TypeError, ValueError):
                        pass
            found.update(_ids(item, names))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_ids(item, names))
    return found


class ResourceCompositionService:
    """执行 PRECHECK → COMMAND → VERIFY，并持久化可恢复状态。"""

    def __init__(
        self, *, uow_factory, agent_client, knowledge_client,
        data_query_client, model_clients: Iterable[Any], notification_center,
    ):
        self._uow_factory = uow_factory
        self._agent = agent_client
        self._knowledge = knowledge_client
        self._data_query = data_query_client
        self._models = tuple(model_clients)
        self._notifications = notification_center

    @staticmethod
    def _receipt(entity: _ReceiptState, *, replay: bool) -> CompositionReceipt:
        return CompositionReceipt(
            receipt_id=entity.receipt_id,
            operation=entity.operation,
            idempotency_key=entity.idempotency_key,
            status=entity.status,
            resource_type=entity.resource_type,
            resource_id=entity.resource_id,
            resource_version=entity.resource_version,
            error_code=entity.error_code,
            verification=dict(entity.verification_json or {}),
            idempotent_replay=replay,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

    async def get_receipt(
        self, *, receipt_id: UUID, domain_id: int, actor_id: str,
    ) -> CompositionReceipt:
        async with self._uow_factory() as uow:
            entity = await uow.compositions.get(
                receipt_id=receipt_id, domain_id=domain_id, actor_id=actor_id,
            )
            if entity is None:
                raise CompositionError("COMPOSITION_RECEIPT_NOT_FOUND", "组合回执不存在", 404)
            return self._receipt(_receipt_state(entity), replay=False)

    async def _begin(
        self, *, domain_id: int, actor_id: str, operation: str,
        idempotency_key: str, request_hash: str, resource_type: str,
        resource_id: str | None,
    ) -> tuple[_ReceiptState, bool]:
        try:
            async with self._uow_factory() as uow:
                entity = await uow.compositions.get_by_idempotency(
                    domain_id=domain_id, actor_id=actor_id, operation=operation,
                    idempotency_key=idempotency_key, lock=True,
                )
                if entity is not None:
                    if entity.request_hash != request_hash:
                        raise CompositionError(
                            "COMPOSITION_IDEMPOTENCY_CONFLICT",
                            "Idempotency-Key 已用于不同的组合请求", 409,
                        )
                    return _receipt_state(entity), True
                entity = CompositionReceiptEntity(
                    receipt_id=uuid7(), domain_id=domain_id, actor_id=actor_id,
                    operation=operation, idempotency_key=idempotency_key,
                    request_hash=request_hash, status="PRECHECKING",
                    resource_type=resource_type, resource_id=resource_id,
                    verification_json={}, error_code=None,
                )
                await uow.compositions.add(entity)
                await uow.commit()
                if getattr(uow, "session", None) is not None:
                    await uow.session.refresh(entity)
                return _receipt_state(entity), False
        except IntegrityError:
            # 并发请求由数据库自然键裁决；失败方读取胜出 Receipt。
            async with self._uow_factory() as uow:
                winner = await uow.compositions.get_by_idempotency(
                    domain_id=domain_id, actor_id=actor_id,
                    operation=operation, idempotency_key=idempotency_key,
                )
                if winner is None:
                    raise CompositionError(
                        "COMPOSITION_RECEIPT_CONFLICT",
                        "组合回执并发创建后暂时不可见", 409,
                    )
                if winner.request_hash != request_hash:
                    raise CompositionError(
                        "COMPOSITION_IDEMPOTENCY_CONFLICT",
                        "Idempotency-Key 已用于不同的组合请求", 409,
                    )
                return _receipt_state(winner), True

    async def _reserved_uuid(
        self, *, domain_id: int, actor_id: str,
        operation: str, idempotency_key: str,
    ) -> UUID:
        """重放创建命令时复用 Receipt 中预分配的资源 ID。"""
        async with self._uow_factory() as uow:
            entity = await uow.compositions.get_by_idempotency(
                domain_id=domain_id, actor_id=actor_id,
                operation=operation, idempotency_key=idempotency_key,
            )
            if entity is not None and entity.resource_id:
                return UUID(entity.resource_id)
        return uuid7()

    async def _transition(
        self, *, receipt_id: UUID, domain_id: int, actor_id: str, status: str,
        resource_id: str | None = None, resource_version: str | None = None,
        verification: dict[str, Any] | None = None, error_code: str | None = None,
    ) -> _ReceiptState:
        async with self._uow_factory() as uow:
            entity = await uow.compositions.get(
                receipt_id=receipt_id, domain_id=domain_id, actor_id=actor_id,
            )
            if entity is None:
                raise RuntimeError("组合回执在状态迁移时不存在")
            await uow.compositions.transition(
                entity, status=status, resource_id=resource_id,
                resource_version=resource_version, verification=verification,
                error_code=error_code,
            )
            await uow.commit()
            if getattr(uow, "session", None) is not None:
                await uow.session.refresh(entity)
            return _receipt_state(entity)

    async def _execute(
        self, *, domain_id: int, actor_id: str, operation: str,
        idempotency_key: str, request_payload: Any, resource_type: str,
        resource_id: str | None, precheck: Precheck, command: Command,
        verify: Verify, resource_ref: list[UUID] | None = None,
    ) -> CompositionReceipt:
        entity, replay = await self._begin(
            domain_id=domain_id, actor_id=actor_id, operation=operation,
            idempotency_key=idempotency_key, request_hash=_hash(request_payload),
            resource_type=resource_type, resource_id=resource_id,
        )
        if resource_ref is not None and entity.resource_id is not None:
            resource_ref[0] = UUID(entity.resource_id)
        if entity.status == "SUCCEEDED":
            return self._receipt(entity, replay=True)
        if (
            replay
            and entity.status == "PRECHECKING"
            and entity.updated_at >= _now() - timedelta(minutes=5)
        ):
            raise CompositionError(
                "COMPOSITION_IN_PROGRESS",
                "相同幂等请求正在执行前置检查", 409,
            )

        # COMMAND_SUBMITTED/COMPENSATION_REQUIRED 表示命令结果不确定；重放只能验证。
        if replay and entity.status in {"COMMAND_SUBMITTED", "COMPENSATION_REQUIRED"}:
            try:
                result = await verify()
            except Exception:
                result = None
            if result is None:
                raise CompositionError(
                    "COMPOSITION_RECOVERY_REQUIRED",
                    "下游命令结果仍无法确认，需要人工恢复", 409,
                )
            entity = await self._transition(
                receipt_id=entity.receipt_id, domain_id=domain_id,
                actor_id=actor_id, status="SUCCEEDED",
                resource_version=str(result.get("resource_version") or "") or None,
                verification=result,
            )
            return self._receipt(entity, replay=True)

        try:
            await precheck()
        except CompositionError as exc:
            await self._transition(
                receipt_id=entity.receipt_id, domain_id=domain_id,
                actor_id=actor_id, status="FAILED_PRECHECK", error_code=exc.code,
            )
            raise
        except Exception as exc:
            await self._transition(
                receipt_id=entity.receipt_id, domain_id=domain_id,
                actor_id=actor_id, status="FAILED_PRECHECK",
                error_code="COMPOSITION_PRECHECK_UNAVAILABLE",
            )
            raise CompositionError(
                "COMPOSITION_PRECHECK_UNAVAILABLE", "组合前置检查暂时不可用", 503,
            ) from exc

        entity = await self._transition(
            receipt_id=entity.receipt_id, domain_id=domain_id,
            actor_id=actor_id, status="COMMAND_SUBMITTED",
        )
        command_error: Exception | None = None
        try:
            await command()
        except Exception as exc:
            command_error = exc
        try:
            result = await verify()
        except Exception:
            result = None
        if result is None:
            error_code = (
                str(getattr(command_error, "code", "COMPOSITION_VERIFICATION_FAILED"))
                if command_error else "COMPOSITION_VERIFICATION_FAILED"
            )
            entity = await self._transition(
                receipt_id=entity.receipt_id, domain_id=domain_id,
                actor_id=actor_id, status="COMPENSATION_REQUIRED",
                error_code=error_code,
            )
            raise CompositionError(
                "COMPOSITION_COMPENSATION_REQUIRED",
                "组合命令结果无法验证，已记录恢复回执且不会自动重复命令", 409,
            ) from command_error
        entity = await self._transition(
            receipt_id=entity.receipt_id, domain_id=domain_id,
            actor_id=actor_id, status="SUCCEEDED",
            resource_version=str(result.get("resource_version") or "") or None,
            verification=result,
        )
        return self._receipt(entity, replay=replay)

    async def _model(self, model_id: UUID) -> dict[str, Any]:
        unavailable = 0
        for client in self._models:
            try:
                return await client.get_model(model_id)
            except LookupError:
                continue
            except Exception:
                unavailable += 1
        if unavailable == len(self._models):
            raise CompositionError("MODEL_CATALOG_UNAVAILABLE", "模型目录暂时不可用", 503)
        raise CompositionError("MODEL_NOT_FOUND", f"模型 {model_id} 不存在", 422)

    async def _active_models(self, models: dict[str, UUID]) -> None:
        for model_id in models.values():
            model = await self._model(model_id)
            if model.get("status") != "ACTIVE":
                raise CompositionError("MODEL_NOT_ACTIVE", f"模型 {model_id} 未启用", 409)

    async def _active_collections(self, domain_id: int, ids: Iterable[UUID], context) -> None:
        for collection_id in ids:
            collection = await self._knowledge.get_collection(
                domain_id=domain_id, collection_id=collection_id, auth_context=context,
            )
            if collection.get("status") != "ACTIVE":
                raise CompositionError(
                    "COLLECTION_NOT_ACTIVE", f"知识库 {collection_id} 未启用", 409,
                )

    async def _sync_collections(
        self, *, domain_id: int, agent_id: UUID,
        desired: set[UUID], context,
    ) -> None:
        response = await self._knowledge.list_agent_bindings(
            domain_id=domain_id, agent_id=agent_id, auth_context=context,
        )
        current = {
            UUID(str(item["collection_id"])) for item in response.get("bindings", [])
            if item.get("status") == "ACTIVE"
        }
        for collection_id in sorted(desired - current, key=str):
            await self._knowledge.bind_collection(
                domain_id=domain_id, agent_id=agent_id,
                collection_id=collection_id, note="由 Main API 组合编排创建",
                auth_context=context,
            )
        for collection_id in sorted(current - desired, key=str):
            await self._knowledge.unbind_collection(
                domain_id=domain_id, agent_id=agent_id,
                collection_id=collection_id, auth_context=context,
            )

    async def _validate_data_query_binding(self, binding, context) -> None:
        semantic = await self._data_query.management_get(
            resource="semantic-models", resource_id=binding.semantic_model_id,
            auth_context=context,
        )
        if not any(version.get("status") == "PUBLISHED" for version in semantic.get("versions", [])):
            raise CompositionError(
                "SEMANTIC_MODEL_NOT_PUBLISHED", "问数绑定需要已发布的语义模型", 409,
            )
        policy = await self._data_query.management_get(
            resource="policy-bindings", resource_id=binding.policy_binding_id,
            auth_context=context,
        )
        if str(binding.semantic_model_id) not in {
            str(item) for item in policy.get("semantic_model_ids", [])
        }:
            raise CompositionError(
                "DATA_QUERY_POLICY_MISMATCH", "问数策略未包含指定语义模型", 409,
            )

    async def _data_query_binding_exists(self, *, agent_id: UUID, binding, context) -> bool:
        page = await self._data_query.management_list(
            resource="agent-bindings", cursor=None, limit=200,
            auth_context=context,
        )
        return any(
            str(row.get("agent_id")) == str(agent_id)
            and str(row.get("semantic_model_id")) == str(binding.semantic_model_id)
            and str(row.get("policy_binding_id")) == str(binding.policy_binding_id)
            and row.get("status") == "ACTIVE"
            for row in page.get("items", [])
        )

    async def create_agent(
        self, *, body: AgentCompositionCreate, domain_id: int,
        actor_id: str, idempotency_key: str, context,
    ) -> CompositionReceipt:
        agent_id = await self._reserved_uuid(
            domain_id=domain_id, actor_id=actor_id,
            operation="AGENT_CREATE", idempotency_key=idempotency_key,
        )
        resource_ref = [agent_id]
        payload = body.agent.model_dump(mode="json")
        desired = set(body.collection_ids)
        created: dict[str, Any] = {}

        async def precheck() -> None:
            await self._active_models(body.agent.models)
            await self._active_collections(domain_id, desired, context)
            if body.agent.data_query_mode == "SEMANTIC":
                if body.data_query_binding is None:
                    raise CompositionError(
                        "SEMANTIC_BINDING_REQUIRED",
                        "SEMANTIC Agent 必须同时提供问数绑定", 409,
                    )
                await self._validate_data_query_binding(
                    body.data_query_binding, context,
                )
            elif body.data_query_binding is not None:
                raise CompositionError(
                    "DATA_QUERY_BINDING_MODE_MISMATCH",
                    "只有 SEMANTIC 模式可以提供问数绑定", 422,
                )

        async def command() -> None:
            create_payload = dict(payload)
            if body.agent.status == "ACTIVE" and body.data_query_binding is not None:
                create_payload["status"] = "DRAFT"
            created.update(await self._agent.create_agent(
                payload={**create_payload, "agent_id": str(resource_ref[0])}, auth_context=context,
            ))
            if body.data_query_binding is not None:
                await self._data_query.management_create(
                    resource="agent-bindings",
                    payload={
                        "agent_id": str(resource_ref[0]),
                        "semantic_model_id": str(body.data_query_binding.semantic_model_id),
                        "policy_binding_id": str(body.data_query_binding.policy_binding_id),
                    }, auth_context=context,
                )
            if body.agent.status == "ACTIVE" and create_payload["status"] == "DRAFT":
                updated = await self._agent.update_agent(
                    agent_id=resource_ref[0],
                    payload={
                        "expected_row_version": int(created["row_version"]),
                        "status": "ACTIVE",
                    }, auth_context=context,
                )
                created.update(updated)
            await self._sync_collections(
                domain_id=domain_id, agent_id=resource_ref[0],
                desired=desired, context=context,
            )

        async def verify() -> dict[str, Any] | None:
            agent = await self._agent.get_agent(agent_id=resource_ref[0], auth_context=context)
            bindings = await self._knowledge.list_agent_bindings(
                domain_id=domain_id, agent_id=resource_ref[0], auth_context=context,
            )
            actual = {
                UUID(str(item["collection_id"])) for item in bindings.get("bindings", [])
                if item.get("status") == "ACTIVE"
            }
            if actual != desired:
                return None
            if body.data_query_binding is not None and not await self._data_query_binding_exists(
                agent_id=resource_ref[0], binding=body.data_query_binding, context=context,
            ):
                return None
            return {"resource_version": agent.get("row_version"), "collection_ids": sorted(map(str, actual))}

        return await self._execute(
            domain_id=domain_id, actor_id=actor_id, operation="AGENT_CREATE",
            idempotency_key=idempotency_key,
            request_payload=body.model_dump(mode="json"), resource_type="agent",
            resource_id=str(agent_id), precheck=precheck, command=command,
            verify=verify, resource_ref=resource_ref,
        )

    async def update_agent(
        self, *, agent_id: UUID, body: AgentCompositionUpdate,
        domain_id: int, actor_id: str, idempotency_key: str,
        expected_version: int, context,
    ) -> CompositionReceipt:
        if body.agent.expected_row_version != expected_version:
            raise CompositionError("IF_MATCH_MISMATCH", "If-Match 与请求版本不一致", 400)
        desired = set(body.collection_ids or ()) if body.collection_ids is not None else None
        current_holder: dict[str, Any] = {}

        async def precheck() -> None:
            current = await self._agent.get_agent(agent_id=agent_id, auth_context=context)
            current_holder.update(current)
            if int(current.get("row_version", 0)) != expected_version:
                raise CompositionError("AGENT_VERSION_CONFLICT", "Agent 已被其他请求修改", 409)
            await self._active_models(body.agent.models or current.get("models", {}))
            if desired is not None:
                await self._active_collections(domain_id, desired, context)
            effective_mode = body.agent.data_query_mode or current.get("data_query_mode")
            effective_status = body.agent.status or current.get("status")
            if body.data_query_binding is not None:
                if effective_mode != "SEMANTIC":
                    raise CompositionError(
                        "DATA_QUERY_BINDING_MODE_MISMATCH",
                        "只有 SEMANTIC 模式可以提供问数绑定", 422,
                    )
                await self._validate_data_query_binding(body.data_query_binding, context)
            if effective_mode == "SEMANTIC" and effective_status == "ACTIVE":
                if body.data_query_binding is None:
                    page = await self._data_query.management_list(
                        resource="agent-bindings", cursor=None, limit=200,
                        auth_context=context,
                    )
                    if not any(str(row.get("agent_id")) == str(agent_id) and row.get("status") == "ACTIVE" for row in page.get("items", [])):
                        raise CompositionError(
                            "SEMANTIC_BINDING_REQUIRED",
                            "启用 SEMANTIC Agent 前必须建立问数绑定", 409,
                        )

        async def command() -> None:
            update_payload = body.agent.model_dump(mode="json", exclude_unset=True)
            requested_status = update_payload.pop("status", None)
            if body.data_query_binding is not None and not await self._data_query_binding_exists(
                agent_id=agent_id, binding=body.data_query_binding, context=context,
            ):
                await self._data_query.management_create(
                    resource="agent-bindings",
                    payload={
                        "agent_id": str(agent_id),
                        "semantic_model_id": str(body.data_query_binding.semantic_model_id),
                        "policy_binding_id": str(body.data_query_binding.policy_binding_id),
                    }, auth_context=context,
                )
            if requested_status is not None:
                update_payload["status"] = requested_status
            await self._agent.update_agent(
                agent_id=agent_id,
                payload=update_payload,
                auth_context=context,
            )
            if desired is not None:
                await self._sync_collections(
                    domain_id=domain_id, agent_id=agent_id,
                    desired=desired, context=context,
                )

        async def verify() -> dict[str, Any] | None:
            agent = await self._agent.get_agent(agent_id=agent_id, auth_context=context)
            if int(agent.get("row_version", 0)) <= expected_version:
                return None
            if desired is not None:
                response = await self._knowledge.list_agent_bindings(
                    domain_id=domain_id, agent_id=agent_id, auth_context=context,
                )
                actual = {UUID(str(item["collection_id"])) for item in response.get("bindings", []) if item.get("status") == "ACTIVE"}
                if actual != desired:
                    return None
            if body.data_query_binding is not None and not await self._data_query_binding_exists(
                agent_id=agent_id, binding=body.data_query_binding, context=context,
            ):
                return None
            return {"resource_version": agent.get("row_version")}

        return await self._execute(
            domain_id=domain_id, actor_id=actor_id, operation="AGENT_UPDATE",
            idempotency_key=idempotency_key,
            request_payload=body.model_dump(mode="json"), resource_type="agent",
            resource_id=str(agent_id), precheck=precheck, command=command, verify=verify,
        )

    async def create_collection(
        self, *, body: CollectionCompositionCreate, domain_id: int,
        actor_id: str, idempotency_key: str, context,
    ) -> CompositionReceipt:
        collection_id = await self._reserved_uuid(
            domain_id=domain_id, actor_id=actor_id,
            operation="COLLECTION_CREATE", idempotency_key=idempotency_key,
        )
        resource_ref = [collection_id]
        definition = body.collection

        async def precheck() -> None:
            await self._active_models(definition.models)

        async def command() -> None:
            await self._knowledge.create_collection(
                domain_id=domain_id,
                payload={**definition.model_dump(mode="json"), "collection_id": str(resource_ref[0])},
                auth_context=context,
            )

        async def verify() -> dict[str, Any] | None:
            row = await self._knowledge.get_collection(
                domain_id=domain_id, collection_id=resource_ref[0], auth_context=context,
            )
            if {key: str(value) for key, value in definition.models.items()} != {
                key: str(value) for key, value in row.get("models", {}).items()
            }:
                return None
            return {"resource_version": row.get("row_version")}

        return await self._execute(
            domain_id=domain_id, actor_id=actor_id, operation="COLLECTION_CREATE",
            idempotency_key=idempotency_key,
            request_payload=body.model_dump(mode="json"), resource_type="collection",
            resource_id=str(collection_id), precheck=precheck, command=command,
            verify=verify, resource_ref=resource_ref,
        )

    async def update_collection_models(
        self, *, collection_id: UUID, body: CollectionModelsCompositionUpdate,
        domain_id: int, actor_id: str, idempotency_key: str,
        expected_version: int, context,
    ) -> CompositionReceipt:
        async def precheck() -> None:
            current = await self._knowledge.get_collection(
                domain_id=domain_id, collection_id=collection_id, auth_context=context,
            )
            if int(current.get("row_version", 0)) != expected_version:
                raise CompositionError("COLLECTION_VERSION_CONFLICT", "知识库已被其他请求修改", 409)
            await self._active_models(body.models)

        async def command() -> None:
            await self._knowledge.update_collection_models(
                domain_id=domain_id, collection_id=collection_id,
                payload={
                    **body.model_dump(mode="json"),
                    "expected_row_version": expected_version,
                }, auth_context=context,
            )

        async def verify() -> dict[str, Any] | None:
            row = await self._knowledge.get_collection(
                domain_id=domain_id, collection_id=collection_id, auth_context=context,
            )
            expected = {key: str(value) for key, value in body.models.items()}
            actual = {key: str(value) for key, value in row.get("models", {}).items()}
            if expected != actual or int(row.get("row_version", 0)) <= expected_version:
                return None
            return {"resource_version": row.get("row_version")}

        return await self._execute(
            domain_id=domain_id, actor_id=actor_id, operation="COLLECTION_MODELS_UPDATE",
            idempotency_key=idempotency_key, request_payload=body.model_dump(mode="json"),
            resource_type="collection", resource_id=str(collection_id),
            precheck=precheck, command=command, verify=verify,
        )

    async def publish_semantic_model(
        self, *, semantic_model_id: UUID, semantic_model_version_id: UUID,
        body: SemanticModelPublicationComposition, domain_id: int,
        actor_id: str, idempotency_key: str, expected_version: int, context,
    ) -> CompositionReceipt:
        async def precheck() -> None:
            detail = await self._data_query.management_get(
                resource="semantic-models", resource_id=semantic_model_id,
                auth_context=context,
            )
            versions = detail.get("versions", [])
            version = next((row for row in versions if str(row.get("semantic_model_version_id")) == str(semantic_model_version_id)), None)
            if version is None:
                raise CompositionError("SEMANTIC_MODEL_VERSION_NOT_FOUND", "语义模型版本不存在", 404)
            if int(version.get("row_version", 0)) != expected_version:
                raise CompositionError("SEMANTIC_MODEL_VERSION_CONFLICT", "语义模型版本已变化", 409)
            await self._active_models({"validation": body.validation_model_id})
            if body.binding is not None:
                await self._agent.get_agent(agent_id=body.binding.agent_id, auth_context=context)
                await self._data_query.management_get(
                    resource="policy-bindings", resource_id=body.binding.policy_binding_id,
                    auth_context=context,
                )

        async def command() -> None:
            await self._data_query.management_publish_model(
                semantic_model_id=semantic_model_id,
                semantic_model_version_id=semantic_model_version_id,
                payload={"schema_snapshot_id": str(body.schema_snapshot_id), "expected_row_version": expected_version},
                auth_context=context,
            )
            if body.binding is not None:
                await self._data_query.management_create(
                    resource="agent-bindings",
                    payload={
                        "agent_id": str(body.binding.agent_id),
                        "semantic_model_id": str(semantic_model_id),
                        "policy_binding_id": str(body.binding.policy_binding_id),
                    }, auth_context=context,
                )

        async def verify() -> dict[str, Any] | None:
            detail = await self._data_query.management_get(
                resource="semantic-models", resource_id=semantic_model_id,
                auth_context=context,
            )
            versions = detail.get("versions", [])
            version = next((row for row in versions if str(row.get("semantic_model_version_id")) == str(semantic_model_version_id)), None)
            if version is None or version.get("status") != "PUBLISHED":
                return None
            if body.binding is not None:
                bindings = await self._data_query.management_list(
                    resource="agent-bindings", cursor=None, limit=200, auth_context=context,
                )
                if not any(
                    str(row.get("agent_id")) == str(body.binding.agent_id)
                    and str(row.get("semantic_model_id")) == str(semantic_model_id)
                    and row.get("status") == "ACTIVE"
                    for row in bindings.get("items", [])
                ):
                    return None
            return {"resource_version": version.get("row_version"), "semantic_model_version_id": str(semantic_model_version_id)}

        return await self._execute(
            domain_id=domain_id, actor_id=actor_id, operation="SEMANTIC_MODEL_PUBLISH",
            idempotency_key=idempotency_key,
            request_payload=body.model_dump(mode="json"), resource_type="semantic_model",
            resource_id=str(semantic_model_id), precheck=precheck, command=command, verify=verify,
        )

    async def reference_graph(
        self, *, resource_type: str, resource_id: UUID,
        domain_id: int, context,
    ) -> ResourceReferenceGraph:
        observed = _now()
        nodes: list[CompositionNode] = []
        edges: list[CompositionEdge] = []
        partial = False
        try:
            if resource_type == "agent":
                agent = await self._agent.get_agent(agent_id=resource_id, auth_context=context)
                nodes.append(_node("agent", resource_id, "agent-runtime", agent))
                for role, model_id in agent.get("models", {}).items():
                    nodes.append(_node("model", model_id, "model-serving", None, attributes={"role": role}))
                    edges.append(CompositionEdge(source_type="agent", source_id=str(resource_id), target_type="model", target_id=str(model_id), relation=f"USES_MODEL:{role}", blocking=True))
                bindings = await self._knowledge.list_agent_bindings(domain_id=domain_id, agent_id=resource_id, auth_context=context)
                for binding in bindings.get("bindings", []):
                    cid = binding["collection_id"]
                    nodes.append(_node("collection", cid, "knowledge-core", binding))
                    edges.append(CompositionEdge(source_type="agent", source_id=str(resource_id), target_type="collection", target_id=str(cid), relation="USES_COLLECTION", blocking=True))
                dq_bindings = await self._data_query.management_list(
                    resource="agent-bindings", cursor=None, limit=200,
                    auth_context=context,
                )
                for binding in dq_bindings.get("items", []):
                    if str(binding.get("agent_id")) != str(resource_id) or binding.get("status") != "ACTIVE":
                        continue
                    semantic_id = str(binding.get("semantic_model_id"))
                    nodes.append(_node("semantic_model", semantic_id, "data-query", binding))
                    edges.append(CompositionEdge(source_type="agent", source_id=str(resource_id), target_type="semantic_model", target_id=semantic_id, relation="USES_SEMANTIC_MODEL", blocking=True))
            elif resource_type == "collection":
                row = await self._knowledge.get_collection(domain_id=domain_id, collection_id=resource_id, auth_context=context)
                nodes.append(_node("collection", resource_id, "knowledge-core", row))
                for role, model_id in row.get("models", {}).items():
                    nodes.append(_node("model", model_id, "model-serving", None, attributes={"role": role}))
                    edges.append(CompositionEdge(source_type="collection", source_id=str(resource_id), target_type="model", target_id=str(model_id), relation=f"USES_MODEL:{role}", blocking=True))
                bindings = await self._knowledge.list_collection_bindings(
                    domain_id=domain_id, collection_id=resource_id,
                    auth_context=context,
                )
                for binding in bindings.get("bindings", []):
                    consumer_type = str(binding.get("consumer_type", "resource")).lower()
                    consumer_id = str(binding.get("consumer_id"))
                    nodes.append(_node(consumer_type, consumer_id, "knowledge-core", binding))
                    edges.append(CompositionEdge(
                        source_type=consumer_type, source_id=consumer_id,
                        target_type="collection", target_id=str(resource_id),
                        relation="USES_COLLECTION", blocking=True,
                    ))
            elif resource_type == "model":
                model = await self._model(resource_id)
                nodes.append(_node("model", resource_id, "model-serving", model))
                references: list[dict[str, Any]] = []
                for client in (self._agent, self._knowledge, self._data_query):
                    try:
                        references.extend(await client.list_model_references(model_id=resource_id, auth_context=context))
                    except Exception:
                        partial = True
                for reference in references:
                    kind = str(reference.get("resource_type") or reference.get("reference_type") or "resource")
                    rid = str(reference.get("resource_id") or reference.get("id") or "unknown")
                    nodes.append(_node(kind.lower(), rid, str(reference.get("service") or "unknown"), reference))
                    edges.append(CompositionEdge(source_type=kind.lower(), source_id=rid, target_type="model", target_id=str(resource_id), relation="REFERENCES", blocking=True))
            elif resource_type in {"semantic_model", "data_source"}:
                resource = "semantic-models" if resource_type == "semantic_model" else "data-sources"
                row = await self._data_query.management_get(resource=resource, resource_id=resource_id, auth_context=context)
                nodes.append(_node(resource_type, resource_id, "data-query", row))
                if resource_type == "semantic_model":
                    for version in row.get("versions", []):
                        data_source_id = version.get("data_source_id")
                        if data_source_id:
                            nodes.append(_node("data_source", data_source_id, "data-query", None))
                            edges.append(CompositionEdge(source_type="semantic_model", source_id=str(resource_id), target_type="data_source", target_id=str(data_source_id), relation="USES_DATA_SOURCE", blocking=True))
                    bindings = await self._data_query.management_list(
                        resource="agent-bindings", cursor=None, limit=200,
                        auth_context=context,
                    )
                    for binding in bindings.get("items", []):
                        if str(binding.get("semantic_model_id")) != str(resource_id) or binding.get("status") != "ACTIVE":
                            continue
                        agent_id = str(binding.get("agent_id"))
                        nodes.append(_node("agent", agent_id, "data-query", binding))
                        edges.append(CompositionEdge(source_type="agent", source_id=agent_id, target_type="semantic_model", target_id=str(resource_id), relation="USES_SEMANTIC_MODEL", blocking=True))
                else:
                    page = await self._data_query.management_list(
                        resource="semantic-models", cursor=None, limit=200,
                        auth_context=context,
                    )
                    for summary in page.get("items", []):
                        semantic_id = summary.get("semantic_model_id")
                        if not semantic_id:
                            continue
                        detail = await self._data_query.management_get(
                            resource="semantic-models",
                            resource_id=UUID(str(semantic_id)),
                            auth_context=context,
                        )
                        if not any(str(version.get("data_source_id")) == str(resource_id) for version in detail.get("versions", [])):
                            continue
                        nodes.append(_node("semantic_model", semantic_id, "data-query", summary))
                        edges.append(CompositionEdge(source_type="semantic_model", source_id=str(semantic_id), target_type="data_source", target_id=str(resource_id), relation="USES_DATA_SOURCE", blocking=True))
            elif resource_type == "run":
                view = await self.run_composition(run_id=resource_id, domain_id=domain_id, actor_id=str(context.asserted_user_id or ""), context=context)
                nodes.extend([view.run, view.agent, *view.models, *view.collections, *view.semantic_models, *view.data_sources, *view.data_query_runs, *view.knowledge_evidence, *view.artifacts])
            else:
                raise CompositionError("COMPOSITION_RESOURCE_TYPE_INVALID", "不支持的资源类型", 422)
        except CompositionError:
            raise
        except Exception:
            partial = True
            nodes.append(_node(resource_type, resource_id, "unknown", None, availability="UNAVAILABLE"))
        blockers = tuple(edge for edge in edges if edge.blocking)
        return ResourceReferenceGraph(
            root_type=resource_type, root_id=str(resource_id), observed_at=observed,
            nodes=tuple(nodes), edges=tuple(edges), blockers=blockers, partial=partial,
        )

    async def run_composition(
        self, *, run_id: UUID, domain_id: int, actor_id: str, context,
    ) -> RunCompositionView:
        debug = await self._agent.get_debug_run(run_id=run_id, auth_context=context)
        run = debug.get("run", debug)
        snapshot = run.get("config_snapshot") or debug.get("config_snapshot") or {}
        agent_id = UUID(str(run.get("agent_id") or snapshot.get("agent_id")))
        agent = await self._agent.get_agent(agent_id=agent_id, auth_context=context)
        model_ids = _ids(snapshot, {"model_id", "model_ids"}) | {UUID(str(value)) for value in agent.get("models", {}).values()}
        collection_ids = _ids(snapshot, {"collection_id", "collection_ids"})
        semantic_ids = _ids(debug, {"semantic_model_id", "semantic_model_ids"})
        source_ids = _ids(debug, {"data_source_id", "data_source_ids"})
        dq_run_ids = _ids(debug, {"data_query_run_id", "data_query_run_ids"})
        evidence_ids = _ids(debug, {"evidence_id", "evidence_ids", "chunk_id", "chunk_ids"})
        partial = False
        model_nodes: list[CompositionNode] = []
        for model_id in sorted(model_ids, key=str):
            try:
                model_nodes.append(_node("model", model_id, "model-serving", await self._model(model_id)))
            except Exception:
                partial = True
                model_nodes.append(_node("model", model_id, "model-serving", None, availability="UNAVAILABLE"))
        collection_nodes: list[CompositionNode] = []
        for collection_id in sorted(collection_ids, key=str):
            try:
                row = await self._knowledge.get_collection(domain_id=domain_id, collection_id=collection_id, auth_context=context)
                collection_nodes.append(_node("collection", collection_id, "knowledge-core", row))
            except Exception:
                partial = True
                collection_nodes.append(_node("collection", collection_id, "knowledge-core", None, availability="UNAVAILABLE"))
        semantic_nodes: list[CompositionNode] = []
        for semantic_id in sorted(semantic_ids, key=str):
            try:
                semantic_nodes.append(_node("semantic_model", semantic_id, "data-query", await self._data_query.management_get(resource="semantic-models", resource_id=semantic_id, auth_context=context)))
            except Exception:
                partial = True
                semantic_nodes.append(_node("semantic_model", semantic_id, "data-query", None, availability="UNAVAILABLE"))
        source_nodes: list[CompositionNode] = []
        for source_id in sorted(source_ids, key=str):
            try:
                source_nodes.append(_node(
                    "data_source", source_id, "data-query",
                    await self._data_query.management_get(
                        resource="data-sources", resource_id=source_id,
                        auth_context=context,
                    ),
                ))
            except Exception:
                partial = True
                source_nodes.append(_node(
                    "data_source", source_id, "data-query", None,
                    availability="UNAVAILABLE",
                ))
        dq_nodes: list[CompositionNode] = []
        for dq_run_id in sorted(dq_run_ids, key=str):
            try:
                payload = await self._data_query.get_run(data_query_run_id=dq_run_id, auth_context=context)
                result = await self._data_query.get_result(data_query_run_id=dq_run_id, auth_context=context)
                safe = {key: result.get(key) for key in ("columns", "row_count", "truncated", "provenance") if key in result}
                dq_nodes.append(_node("data_query_run", dq_run_id, "data-query", payload, attributes=safe))
            except Exception:
                partial = True
                dq_nodes.append(_node("data_query_run", dq_run_id, "data-query", None, availability="UNAVAILABLE"))
        artifacts = []
        for item in debug.get("artifacts", []):
            artifact_id = item.get("artifact_id")
            if artifact_id:
                artifacts.append(_node("artifact", artifact_id, str(item.get("producer") or "agent-runtime"), item, attributes={key: item.get(key) for key in ("artifact_type", "schema_version", "content_hash", "provenance") if key in item}))
        notifications = await self._notifications.resource_notifications(
            domain_id=domain_id, actor_id=actor_id,
            resource_type="agent_run", resource_id=str(run_id), limit=100,
        )
        return RunCompositionView(
            run_id=run_id, observed_at=_now(),
            run=_node("run", run_id, "agent-runtime", run),
            agent=_node("agent", agent_id, "agent-runtime", agent),
            models=tuple(model_nodes), collections=tuple(collection_nodes),
            semantic_models=tuple(semantic_nodes), data_sources=tuple(source_nodes),
            data_query_runs=tuple(dq_nodes),
            knowledge_evidence=tuple(_node("knowledge_evidence", item, "knowledge-core", None) for item in sorted(evidence_ids, key=str)),
            artifacts=tuple(artifacts), notifications=tuple(notifications),
            tasks=tuple(debug.get("tasks", [])), partial=partial,
        )
