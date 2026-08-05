"""Model Serving 目录阻塞事件通知发布器。"""

from platform_core.notifications import NotificationEnvelope, publish_notification


class ModelCatalogNotificationPublisher:
    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def publish_blocked(
        self, *, event_type: str, model_id, model_name: str,
        auth_context, references,
    ) -> None:
        if auth_context.domain_id is None:
            return
        actor_id = str(auth_context.asserted_user_id or "").strip() or None
        async with self._uow_factory() as uow:
            await publish_notification(
                uow=uow,
                producer_service="model-serving",
                event_key=f"{model_id}:{event_type}",
                envelope=NotificationEnvelope(
                    domain_id=int(auth_context.domain_id),
                    event_type=event_type,
                    resource_type="ai_model",
                    resource_id=str(model_id),
                    resource_name=model_name,
                    initiator_actor_id=actor_id,
                    recipient_actor_ids=[actor_id] if actor_id else [],
                    summary="模型仍存在业务引用，当前操作已阻止。",
                    correlation_id=str(auth_context.trace_id),
                    safe_data={
                        "reference_count": len(references.references),
                        "unavailable_service_count": len(references.unavailable_services),
                    },
                ),
            )
            await uow.commit()

    async def publish_reload_failed(
        self, *, model, auth_context, error_code: str,
    ) -> None:
        if auth_context.domain_id is None:
            return
        actor_id = str(auth_context.asserted_user_id or "").strip() or None
        async with self._uow_factory() as uow:
            await publish_notification(
                uow=uow,
                producer_service="model-serving",
                event_key=(
                    f"{model['model_id']}:{model['row_version']}:"
                    "model.runtime.reload_failed"
                ),
                envelope=NotificationEnvelope(
                    domain_id=int(auth_context.domain_id),
                    event_type="model.runtime.reload_failed",
                    resource_type="ai_model",
                    resource_id=str(model["model_id"]),
                    resource_name=str(model["display_name"]),
                    initiator_actor_id=actor_id,
                    recipient_actor_ids=[actor_id] if actor_id else [],
                    summary="模型目录已更新，但运行时缓存重新加载失败。",
                    correlation_id=str(auth_context.trace_id),
                    safe_data={"error_code": error_code[:128]},
                ),
            )
            await uow.commit()
