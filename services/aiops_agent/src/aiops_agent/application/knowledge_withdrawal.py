"""案例精确版本的持久化撤回命令、来源围栏和确认回执。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.entities import OutboxEntity
from platform_core.identity import uuid7
from platform_core.contracts.knowledge_withdrawal import KnowledgeWithdrawalReceipt

WITHDRAWAL_EVENT = "OPS_CASE_WITHDRAWAL_REQUESTED"


class CaseWithdrawalCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: UUID
    domain_id: str = Field(min_length=1, max_length=64)
    tenant_id: UUID | None
    asset_id: UUID
    asset_version_id: UUID
    index_ref_id: UUID
    collection_id: UUID
    bundle_id: UUID
    bundle_revision_id: UUID


def withdrawal_key(index_id) -> str:
    return f"case-withdrawal:{index_id}"


def payload_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def invalid_withdrawal() -> AIOpsApplicationError:
    return AIOpsApplicationError(
        code="AIOPS_CASE_WITHDRAWAL_INVALID",
        message="案例撤回命令或接收方回执与精确版本不匹配",
        status_code=409, retryable=False,
    )


async def queue_case_withdrawals(*, uow, asset, version) -> None:
    """与案例停用同事务保存；只处理案例，不删除历史正文。"""
    if asset.asset_kind != "DIAGNOSIS_CASE":
        return
    relations = await uow.operations_knowledge.get_version_relations(
        asset_version_id=version.asset_version_id,
    )
    for index in relations["indexes"]:
        key = withdrawal_key(index.index_ref_id)
        if await uow.outbox.get_by_idempotency(idempotency_key=key) is not None:
            continue
        command = CaseWithdrawalCommand(
            request_id=uuid7(), domain_id=str(asset.domain_id),
            tenant_id=getattr(asset, "tenant_id", None),
            asset_id=asset.asset_id, asset_version_id=version.asset_version_id,
            index_ref_id=index.index_ref_id, collection_id=index.collection_id,
            bundle_id=index.bundle_id, bundle_revision_id=index.bundle_revision_id,
        )
        payload = command.model_dump(mode="json")
        await uow.outbox.add(OutboxEntity(
            outbox_id=command.request_id, aggregate_type="KNOWLEDGE_VERSION",
            aggregate_id=version.asset_version_id, event_type=WITHDRAWAL_EVENT,
            idempotency_key=key, payload_json=payload, payload_hash=payload_hash(payload),
            status="PENDING", available_at=datetime.now(UTC), attempt_count=0,
            max_attempts=12, trace_id=str(command.request_id),
        ))


async def assert_case_not_withdrawn(*, uow, indexes) -> None:
    for index in indexes:
        if await uow.outbox.get_by_idempotency(
            idempotency_key=withdrawal_key(index.index_ref_id),
        ) is not None:
            raise AIOpsApplicationError(
                code="AIOPS_CASE_REVISION_WITHDRAWN",
                message="该案例版本已发出永久撤回请求，请创建并审核新版本",
                status_code=409,
            )


async def validate_command_source(*, uow, command, domain_id):
    """读取确切版本及其索引，不相信消息里的资源归属。"""
    version = await uow.operations_knowledge.get_version(
        domain_id=domain_id, asset_version_id=command.asset_version_id, for_update=True,
    )
    if version is None or version.asset_id != command.asset_id or version.status == "PUBLISHED":
        raise invalid_withdrawal()
    asset = await uow.operations_knowledge.get_asset(
        domain_id=domain_id, asset_id=command.asset_id, for_update=True,
    )
    if (
        asset is None or asset.asset_kind != "DIAGNOSIS_CASE"
        or str(asset.domain_id) != command.domain_id
        or getattr(asset, "tenant_id", None) != command.tenant_id
        or asset.current_version_id == command.asset_version_id
    ):
        raise invalid_withdrawal()
    relations = await uow.operations_knowledge.get_version_relations(
        asset_version_id=command.asset_version_id,
    )
    match = next((item for item in relations["indexes"]
        if item.index_ref_id == command.index_ref_id), None)
    if match is None or any(
        getattr(match, field) != getattr(command, field)
        for field in ("collection_id", "bundle_id", "bundle_revision_id")
    ):
        raise invalid_withdrawal()
    outbox = await uow.outbox.get_by_idempotency(
        idempotency_key=withdrawal_key(command.index_ref_id),
    )
    payload = command.model_dump(mode="json")
    if (
        outbox is None or outbox.outbox_id != command.request_id
        or outbox.aggregate_id != command.asset_version_id
        or outbox.event_type != WITHDRAWAL_EVENT
        or outbox.payload_json != payload or outbox.payload_hash != payload_hash(payload)
    ):
        raise invalid_withdrawal()
    return asset, version


def validate_receipt(command, response) -> KnowledgeWithdrawalReceipt:
    try:
        receipt = KnowledgeWithdrawalReceipt.model_validate(response)
    except ValidationError as exc:
        raise invalid_withdrawal() from exc
    if any(
        getattr(receipt, name) != getattr(command, name)
        for name in ("request_id", "collection_id", "bundle_id", "bundle_revision_id")
    ):
        raise invalid_withdrawal()
    if receipt.withdrawn_at > datetime.now(UTC) + timedelta(minutes=5):
        raise invalid_withdrawal()
    return receipt


async def withdrawal_views(*, uow, indexes, reviews) -> list[dict]:
    """撤回进度与接收方确认分开；Outbox已投递也不能替代回执。"""
    result = []
    for index in indexes:
        message = await uow.outbox.get_by_idempotency(
            idempotency_key=withdrawal_key(index.index_ref_id),
        )
        if message is None:
            continue
        receipt = None
        state = "PENDING"
        try:
            command = CaseWithdrawalCommand.model_validate(message.payload_json)
            review = next((item for item in reviews
                if item.review_id == command.request_id and item.decision == "EXTERNAL_WITHDRAWAL_ACK"), None)
            if review is not None:
                receipt = validate_receipt(command, json.loads(review.comment_text))
                state = "ACKNOWLEDGED"
            elif message.status == "FAILED":
                state = "FAILED"
        except (ValueError, AIOpsApplicationError):
            state = "FAILED"
        result.append({
            "index_ref_id": str(index.index_ref_id), "consumer": "KNOWLEDGE_CORE",
            "request_id": str(message.outbox_id), "status": state,
            "delivery_status": message.status, "attempt_count": int(message.attempt_count),
            "error_code": message.error_code,
            "receipt": receipt.model_dump(mode="json") if receipt else None,
        })
    return result
