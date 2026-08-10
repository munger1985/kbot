"""Knowledge Core 事务内通知 Outbox 发布器。"""

from typing import Protocol
from uuid import UUID

from platform_core.notifications import NotificationEnvelope, publish_notification


class KnowledgeNotificationPublisher(Protocol):
    async def publish(
        self, *, uow, event_type: str, actor_id: str,
        resource_id: str, payload: dict[str, object],
    ) -> None: ...


class KnowledgeOutboxPublisher:
    async def publish(
        self, *, uow, event_type: str, actor_id: str,
        resource_id: str, payload: dict[str, object],
    ) -> None:
        if uow.collections is None:
            raise RuntimeError("Knowledge Core Collection Repository 未初始化")
        collection = await uow.collections.get_by_id(
            collection_id=UUID(resource_id),
        )
        if collection is None:
            raise RuntimeError("通知资源所属 Collection 不存在")
        initiator = actor_id.strip() or None
        if initiator and initiator.startswith("kbot-"):
            initiator = None
        display_name = str(
            payload.get("display_name") or collection.display_name or resource_id
        )
        summary = {
            "knowledge.ingestion.started": "知识内容已进入解析与索引流程。",
            "knowledge.collection.purge_completed": "知识库已安全删除。",
            "knowledge.collection.purge_failed": "知识库删除失败，请检查任务状态。",
            "knowledge.ingestion.completed": "知识内容处理完成。",
            "knowledge.ingestion.partial": "部分知识内容处理失败，需要处理。",
            "knowledge.ingestion.failed": "知识内容处理失败。",
        }[event_type]
        await publish_notification(
            uow=uow,
            producer_service="knowledge-core",
            event_key=(
                f"{payload.get('event_key') or payload.get('job_id') or resource_id}:"
                f"{event_type}"
            ),
            envelope=NotificationEnvelope(
                domain_id=int(collection.domain_id),
                event_type=event_type,
                resource_type="knowledge_collection",
                resource_id=resource_id,
                resource_name=display_name,
                initiator_actor_id=initiator,
                recipient_actor_ids=[initiator] if initiator else [],
                summary=summary,
                correlation_id=str(
                    payload.get("correlation_id")
                    or payload.get("job_id")
                    or resource_id
                ),
                operation_id=str(
                    payload.get("operation_id")
                    or payload.get("job_id")
                    or resource_id
                ),
                safe_data={
                    "error_code": payload.get("error_code"),
                    "objects_deleted": payload.get("objects_deleted"),
                    "progress_current": payload.get("progress_current"),
                    "progress_total": payload.get("progress_total"),
                },
            ),
        )
