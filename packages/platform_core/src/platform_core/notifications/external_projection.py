"""外发任务的共同投影规则；无网络调用、无提交，不依赖 Inbox。"""

from datetime import datetime, timedelta, timezone
from uuid import UUID
from platform_core.notifications.delivery import EXTERNAL_EVENTS, GROUP_EVENTS, DeliveryStatus as Status, payload_hash, render_card, stable_id
from platform_core.contracts.aiops.notification_projection import NotificationResourceRequest


def now_utc():
    return datetime.now(timezone.utc)


def noise_controlled(event_type, severity):
    """仅低等级新告警与恢复允许降噪；升级、待处理和结果事件不受影响。"""
    return event_type in {"aiops.situation.detected", "aiops.situation.recovered"} and severity in {"INFO", "WARNING"}


def after_quiet(config, now):
    """计算半开 UTC 静默时段的结束，不改变消息期限。"""
    start, end = config.quiet_start_utc_hour, config.quiet_end_utc_hour
    if start is None:
        return now
    utc = now.astimezone(timezone.utc)
    inside = start <= utc.hour < end if start < end else utc.hour >= start or utc.hour < end
    if not inside:
        return now
    ready = utc.replace(hour=end, minute=0, second=0, microsecond=0)
    return ready + timedelta(days=1) if ready <= utc else ready


def noise_schedule(config, *, now, event_type, severity):
    """固定合并窗口避免持续新事件无限延迟；静默仅影响外部任务。"""
    if not noise_controlled(event_type, severity):
        return now, None
    bucket = None
    ready = now
    if config.coalesce_seconds:
        bucket = int(now.timestamp()) // config.coalesce_seconds
        ready = datetime.fromtimestamp((bucket + 1) * config.coalesce_seconds, timezone.utc)
    return after_quiet(config, ready), bucket


async def new_delivery_job(repo, *, scope, connection, binding, source_event_id, event_type, content,
                   entry_id, expires_at, purpose, actor_id, now, severity=None, available_at=None, coalesce_key=None):
    job_id = stable_id(source_event_id, connection["id"], binding["target_type"], binding["external_id"], purpose, 1, 0)
    values = {"id": job_id, "source_event_id": source_event_id, "event_type": event_type,
        "original_severity": severity, "coalesce_key": coalesce_key,
        "connection_id": connection["id"], "connection_version": connection["credential_version"],
        "binding_id": binding["id"], "binding_version": binding["row_version"],
        "recipient_actor_id": binding["actor_id"], "target_type": binding["target_type"],
        "external_id": binding["external_id"], "purpose": purpose, "template_version": 1, "generation": 0,
        "entry_id": entry_id, "content": content, "content_hash": payload_hash(content),
        "provider_uuid": str(job_id), "status": Status.PENDING, "attempt_count": 0, "retry_count": 0,
        "available_at": available_at or now, "expires_at": expires_at}
    created = await repo.add_once("job", scope=scope, actor_id=actor_id, now=now, values=values)
    return values | {"created": created}

async def enqueue_external_event(*, config, repo, scope, source_event_id, event_type, route, legitimate_actors, occurred_at, severity=None):
    if not config.enabled or event_type not in EXTERNAL_EVENTS:
        return
    severity = severity if isinstance(severity, str) and severity in {"INFO", "WARNING", "HIGH", "CRITICAL"} else None
    NotificationResourceRequest(event_type=event_type, domain_id=scope.domain_id,
        tenant_id=scope.tenant_id, route_key=route["route_key"], route_params=route["route_params"])
    now = now_utc()
    expiry = occurred_at + timedelta(seconds=config.job_ttl_seconds)
    if expiry <= now:
        return
    for policy in await repo.active_policies(scope=scope):
        if event_type not in policy["event_types"]:
            continue
        # 仅开启合并时用父连接行锁序列化同一目标投影，默认路径不增加连接行锁。
        connection = await repo.get("connection", scope=scope, object_id=policy["connection_id"],
            lock=bool(config.coalesce_seconds))
        if not connection:
            continue
        targets = [await repo.get("binding", scope=scope, object_id=UUID(str(raw_id))) for raw_id in policy["binding_ids"]]
        existing_bindings = await repo.bindings_for_connection(scope=scope, connection_id=connection["id"])
        for actor_id in policy.get("actor_ids") or []:
            if actor_id in legitimate_actors:
                candidate = next((item for item in existing_bindings if item["actor_id"] == actor_id and item["status"] != "DISABLED"), None)
                targets.append(candidate or {"id": None, "connection_id": connection["id"], "actor_id": actor_id,
                    "external_id": "unmapped:" + str(stable_id(actor_id)), "target_type": "open_id", "row_version": 0, "status": "MISSING"})
        for binding in targets:
            if not binding or binding["connection_id"] != connection["id"] or binding["status"] == "DISABLED":
                continue
            group = binding["target_type"] == "chat_id"
            if (group and event_type not in GROUP_EVENTS) or (not group and binding["actor_id"] not in legitimate_actors):
                continue
            entry_id = stable_id(source_event_id, connection["id"], binding["external_id"], "entry")
            await repo.add_once("entry", scope=scope, actor_id="system:external-delivery", now=now, values={
                "id": entry_id, "recipient_actor_id": None if group else binding["actor_id"],
                "audience": "GROUP" if group else "PERSONAL", "event_type": event_type,
                "route_key": route["route_key"], "route_params": route["route_params"],
                "expires_at": now + timedelta(days=config.entry_ttl_days)})
            content = render_card(event_type=event_type, severity=severity, group=group,
                url=config.public_origin + "/notifications/open/" + str(entry_id))
            ready, bucket = noise_schedule(config, now=now, event_type=event_type, severity=severity)
            merge_key = payload_hash([scope.model_dump(mode="json"), str(connection["id"]),
                connection["credential_version"], str(binding["id"]), binding["row_version"],
                binding["target_type"], binding["external_id"], event_type, severity, route, bucket, 1]) if bucket is not None else None
            job = await new_delivery_job(repo, scope=scope, connection=connection, binding=binding,
                source_event_id=source_event_id, event_type=event_type, content=content, entry_id=entry_id,
                expires_at=expiry, purpose="BUSINESS", actor_id="system:external-delivery", now=now,
                severity=severity, available_at=min(ready, expiry), coalesce_key=merge_key)
            if not job["created"]:
                continue
            if ready >= expiry:
                await repo.change("job", scope=scope, object_id=job["id"], expected_version=1,
                    values={"status": Status.EXPIRED, "last_error_code": "NOTIFICATION_QUIET_EXCEEDS_TTL"},
                    actor_id="system:external-delivery", now=now)
            elif binding["status"] != "VERIFIED" or not connection["enabled"]:
                await repo.change("job", scope=scope, object_id=job["id"], expected_version=1,
                    values={"status": Status.BLOCKED, "last_error_code":
                        "NOTIFICATION_CONNECTION_DISABLED" if not connection["enabled"] else
                        "NOTIFICATION_BINDING_MISSING" if binding["status"] == "MISSING" else "NOTIFICATION_BINDING_UNVERIFIED"},
                    actor_id="system:external-delivery", now=now)
            elif merge_key:
                # 仅合并从未进入网络的同目标、同事件、同资源、同等级任务；旧记录和入口保留。
                for previous in await repo.coalesce_candidates(scope=scope, coalesce_key=merge_key, exclude_id=job["id"]):
                    changed = await repo.change("job", scope=scope, object_id=previous["id"],
                        expected_version=previous["row_version"],
                        values={"status": Status.CANCELLED, "last_error_code": "NOTIFICATION_COALESCED"},
                        actor_id="system:external-delivery", now=now)
                    if changed:
                        await repo.audit(scope=scope, action="delivery.coalesce", object_id=previous["id"],
                            actor_id="system:external-delivery", outcome="SUPERSEDED_BY:" + str(job["id"]), now=now)
