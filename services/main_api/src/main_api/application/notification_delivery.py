"""渠道、身份验证、独立入口与可靠投递的应用编排。"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from time import monotonic
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4
from platform_core.notifications.external_projection import after_quiet, new_delivery_job, noise_controlled

from platform_core.notifications.delivery import (
    ACTIONABLE_EVENTS, EXTERNAL_EVENTS, GROUP_EVENTS, DeliveryScope, DeliveryStatus as Status,
    SendResult, can_retry_uncertain, payload_hash, render_card, stable_id,
)


class ExternalDeliveryError(ValueError):
    def __init__(self, code, status_code=409):
        super().__init__(code)
        self.code, self.status_code = code, status_code


def now_utc():
    return datetime.now(timezone.utc)


def public_record(kind, record):
    fields = {
        "connection": ("id", "tenant_id", "name", "provider", "app_id", "enabled",
                       "health", "last_error_code", "checked_at", "row_version"),
        "binding": ("id", "connection_id", "actor_id", "target_type", "external_id", "status",
                    "verified_at", "row_version"),
        "policy": ("id", "connection_id", "event_types", "binding_ids", "actor_ids", "enabled", "row_version"),
        "job": ("id", "source_event_id", "event_type", "original_severity", "connection_id", "recipient_actor_id", "purpose",
                "binding_id", "target_type",
                "status", "attempt_count", "created_at", "available_at", "expires_at",
                "message_id", "last_error_code", "row_version", "generation", "parent_job_id"),
        "attempt": ("id", "job_id", "attempt_number", "started_at", "finished_at", "elapsed_ms", "status",
                    "error_code", "request_id", "message_id"),
        "audit": ("id", "action", "object_id", "outcome", "created_at", "created_by"),
    }[kind]
    return {key: record[key] for key in fields if key in record}


class ExternalNotificationService:
    def __init__(self, *, uow_factory, config, vault, sender, authority):
        self._uow_factory, self.config = uow_factory, config
        self.vault, self.sender, self.authority = vault, sender, authority

    def enabled(self):
        if not self.config.enabled:
            raise ExternalDeliveryError("NOTIFICATION_EXTERNAL_DISABLED", 503)

    async def list(self, kind, *, scope, limit=100, offset=0, status=None):
        async with self._uow_factory() as uow:
            return [public_record(kind, row) for row in await uow.external_delivery.list(
                kind, scope=scope, limit=limit, offset=offset, status=status)]

    async def create_connection(self, *, scope, actor_id, body):
        self.enabled()
        object_id, now = uuid4(), now_utc()
        reference = await self.vault.write(scope=scope, object_id=uuid4(), values={"app_secret": body.app_secret.get_secret_value()})
        try:
            async with self._uow_factory() as uow:
                await uow.external_delivery.lock_configuration()
                if await uow.external_delivery.connection_for_provider(scope=scope, provider=body.provider):
                    raise ExternalDeliveryError("NOTIFICATION_CONNECTION_ALREADY_CONFIGURED")
                record = await uow.external_delivery.add("connection", scope=scope, actor_id=actor_id, now=now,
                    values={"id": object_id, "name": body.name, "provider": body.provider, "app_id": body.app_id,
                        "credential_ref": reference, "credential_version": 1, "enabled": body.enabled, "health": "UNCHECKED"})
                await uow.external_delivery.audit(scope=scope, action="connection.create", object_id=object_id,
                    actor_id=actor_id, outcome="CREATED", now=now)
                await uow.commit()
                return public_record("connection", record)
        except BaseException:
            await self.vault.delete(scope=scope, reference=reference)
            raise

    async def update_connection(self, *, scope, actor_id, object_id, body):
        self.enabled()
        now, reference = now_utc(), None
        if body.app_secret is not None:
            reference = await self.vault.write(scope=scope, object_id=uuid4(), values={"app_secret": body.app_secret.get_secret_value()})
        try:
            async with self._uow_factory() as uow:
                repo = uow.external_delivery
                row = await repo.get("connection", scope=scope, object_id=object_id, lock=True)
                if row is None:
                    raise ExternalDeliveryError("NOTIFICATION_CONNECTION_NOT_FOUND", 404)
                values = {"name": body.name, "enabled": body.enabled}
                if reference:
                    values.update(credential_ref=reference, credential_version=int(row["credential_version"]) + 1,
                                  health="UNCHECKED", last_error_code=None)
                if not await repo.change("connection", scope=scope, object_id=object_id,
                    expected_version=body.expected_row_version, values=values, actor_id=actor_id, now=now):
                    raise ExternalDeliveryError("NOTIFICATION_VERSION_CONFLICT")
                await repo.audit(scope=scope, action="connection.update", object_id=object_id,
                    actor_id=actor_id, outcome="UPDATED", now=now)
                await uow.commit()
                return public_record("connection", {**row, **values, "row_version": body.expected_row_version + 1})
        except BaseException:
            if reference:
                await self.vault.delete(scope=scope, reference=reference)
            raise

    async def check_connection(self, *, scope, actor_id, object_id):
        self.enabled()
        async with self._uow_factory() as uow:
            row = await uow.external_delivery.get("connection", scope=scope, object_id=object_id)
        if row is None:
            raise ExternalDeliveryError("NOTIFICATION_CONNECTION_NOT_FOUND", 404)
        try:
            secret = (await self.vault.read(scope=scope, reference=row["credential_ref"]))["app_secret"]
            result = await self.sender.check(connection=row, secret=secret)
        except Exception:
            result = SendResult(Status.BLOCKED, "NOTIFICATION_CREDENTIAL_UNAVAILABLE")
        now = now_utc()
        async with self._uow_factory() as uow:
            changed = await uow.external_delivery.change("connection", scope=scope, object_id=object_id,
                expected_version=row["row_version"], values={"health": "TOKEN_VALID" if result.status == Status.ACCEPTED else "ERROR",
                    "checked_at": now, "last_error_code": result.error_code}, actor_id=actor_id, now=now)
            await uow.external_delivery.audit(scope=scope, action="connection.check", object_id=object_id,
                actor_id=actor_id, outcome=(result.error_code or "TOKEN_VALID_NOT_DELIVERED")
                    if changed else "STALE_VERSION_NOT_CHECKED", now=now)
            await uow.commit()
        if not changed:
            raise ExternalDeliveryError("NOTIFICATION_VERSION_CONFLICT")
        return {"health": "TOKEN_VALID" if result.status == Status.ACCEPTED else "ERROR", "error_code": result.error_code,
                "message": "令牌检查不代表消息已投递"}

    async def create_binding(self, *, scope, actor_id, body):
        self.enabled()
        if body.actor_id and not await self.authority.actor_active(scope=scope, actor_id=body.actor_id):
            raise ExternalDeliveryError("NOTIFICATION_ACTOR_UNAVAILABLE", 422)
        now = now_utc()
        # 连接父行锁序列化同一应用下的身份登记；同一身份不得绑定两个本地用户。
        async with self._uow_factory() as uow:
            repo = uow.external_delivery
            connection = await repo.get("connection", scope=scope, object_id=body.connection_id, lock=True)
            if connection is None:
                raise ExternalDeliveryError("NOTIFICATION_CONNECTION_NOT_FOUND", 404)
            bindings = await repo.bindings_for_connection(scope=scope, connection_id=body.connection_id)
            if any(row["status"] != "DISABLED" and
                   (row["external_id"] == body.external_id or (body.actor_id and row["actor_id"] == body.actor_id))
                   for row in bindings):
                raise ExternalDeliveryError("NOTIFICATION_BINDING_CONFLICT")
            row = await repo.add("binding", scope=scope, actor_id=actor_id, now=now, values={
                "id": uuid4(), **body.model_dump(), "status": "VERIFIED" if body.target_type == "chat_id" else "UNVERIFIED",
                "verified_at": now if body.target_type == "chat_id" else None, "challenge_attempts": 0})
            await repo.audit(scope=scope, action="binding.create", object_id=row["id"], actor_id=actor_id,
                outcome=row["status"], now=now)
            await uow.commit()
            return public_record("binding", row)

    async def disable_binding(self, *, scope, actor_id, object_id, expected_version):
        async with self._uow_factory() as uow:
            repo = uow.external_delivery
            if not await repo.change("binding", scope=scope, object_id=object_id, expected_version=expected_version,
                values={"status": "DISABLED", "challenge_hash": None, "challenge_ref": None}, actor_id=actor_id, now=now_utc()):
                raise ExternalDeliveryError("NOTIFICATION_VERSION_CONFLICT")
            await repo.audit(scope=scope, action="binding.disable", object_id=object_id, actor_id=actor_id,
                outcome="DISABLED", now=now_utc())
            await uow.commit()

    async def challenge_mine(self, *, scope, actor_id, object_id):
        async with self._uow_factory() as uow:
            row = await uow.external_delivery.get("binding", scope=scope, object_id=object_id)
        if row is None or row["actor_id"] != actor_id:
            raise ExternalDeliveryError("NOTIFICATION_BINDING_NOT_FOUND", 404)
        return await self.challenge(scope=scope, actor_id=actor_id, object_id=object_id)

    async def challenge(self, *, scope, actor_id, object_id):
        self.enabled()
        code, salt, now = f"{secrets.randbelow(1000000):06}", secrets.token_hex(16), now_utc()
        digest = hashlib.scrypt(code.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
        reference = await self.vault.write(scope=scope, object_id=uuid4(), values={"verification_code": code})
        try:
            async with self._uow_factory() as uow:
                repo = uow.external_delivery
                binding = await repo.get("binding", scope=scope, object_id=object_id, lock=True)
                if binding is None or binding["target_type"] != "open_id" or binding["status"] == "DISABLED":
                    raise ExternalDeliveryError("NOTIFICATION_BINDING_NOT_FOUND", 404)
                if binding["challenge_sent_at"] and now < binding["challenge_sent_at"] + timedelta(seconds=self.config.verification_cooldown_seconds):
                    raise ExternalDeliveryError("NOTIFICATION_VERIFICATION_RATE_LIMIT", 429)
                connection = await repo.get("connection", scope=scope, object_id=binding["connection_id"])
                if connection is None or not connection["enabled"]:
                    raise ExternalDeliveryError("NOTIFICATION_CONNECTION_DISABLED")
                version = binding["row_version"] + 1
                expiry = now + timedelta(seconds=self.config.verification_ttl_seconds)
                await repo.change("binding", scope=scope, object_id=object_id, expected_version=binding["row_version"],
                    values={"status": "UNVERIFIED", "challenge_hash": digest, "challenge_salt": salt,
                        "challenge_ref": reference, "challenge_expires_at": expiry, "challenge_sent_at": now,
                        "challenge_attempts": 0, "verified_at": None}, actor_id=actor_id, now=now)
                binding = {**binding, "row_version": version}
                content = {"verification_ref": reference, "challenge_hash": digest}
                job = await self._new_job(repo, scope=scope, connection=connection, binding=binding,
                    source_event_id=uuid4(), event_type="identity.verification", content=content,
                    entry_id=None, expires_at=expiry, purpose="VERIFY", actor_id=actor_id, now=now)
                await repo.audit(scope=scope, action="binding.challenge", object_id=object_id,
                    actor_id=actor_id, outcome="QUEUED", now=now)
                await uow.commit()
                return {"job_id": job["id"], "row_version": version, "expires_at": expiry}
        except BaseException:
            await self.vault.delete(scope=scope, reference=reference)
            raise

    async def confirm(self, *, scope, actor_id, object_id, body):
        self.enabled()
        now, valid = now_utc(), False
        async with self._uow_factory() as uow:
            repo = uow.external_delivery
            row = await repo.get("binding", scope=scope, object_id=object_id, lock=True)
            if row is None or row["actor_id"] != actor_id:
                raise ExternalDeliveryError("NOTIFICATION_BINDING_NOT_FOUND", 404)
            if row["row_version"] != body.expected_row_version:
                raise ExternalDeliveryError("NOTIFICATION_VERSION_CONFLICT")
            if (row["status"] != "UNVERIFIED" or not row["challenge_hash"] or not row["challenge_expires_at"]
                or row["challenge_expires_at"] <= now or row["challenge_attempts"] >= self.config.verification_max_attempts):
                raise ExternalDeliveryError("NOTIFICATION_VERIFICATION_EXPIRED", 422)
            digest = hashlib.scrypt(body.code.encode(), salt=bytes.fromhex(row["challenge_salt"]), n=16384, r=8, p=1).hex()
            valid = hmac.compare_digest(digest, row["challenge_hash"])
            values = {"challenge_attempts": row["challenge_attempts"] + 1}
            if valid:
                values.update(status="VERIFIED", verified_at=now, challenge_hash=None, challenge_ref=None)
            if valid:
                await repo.change("binding", scope=scope, object_id=object_id, expected_version=row["row_version"],
                    values=values, actor_id=actor_id, now=now)
            else:
                # 错误尝试计数不能使尚未发送的验证码版本失效。
                await repo.record_verification_failure(scope=scope, binding=row, now=now)
            await repo.audit(scope=scope, action="binding.confirm", object_id=object_id, actor_id=actor_id,
                outcome="VERIFIED" if valid else "INVALID_CODE", now=now)
            await uow.commit()
        if not valid:
            raise ExternalDeliveryError("NOTIFICATION_VERIFICATION_INVALID", 422)
        return {"status": "VERIFIED"}

    async def create_policy(self, *, scope, actor_id, body):
        self.enabled()
        now = now_utc()
        async with self._uow_factory() as uow:
            repo = uow.external_delivery
            if not await repo.get("connection", scope=scope, object_id=body.connection_id, lock=True):
                raise ExternalDeliveryError("NOTIFICATION_CONNECTION_NOT_FOUND", 404)
            for binding_id in body.binding_ids:
                binding = await repo.get("binding", scope=scope, object_id=binding_id)
                if not binding or binding["connection_id"] != body.connection_id:
                    raise ExternalDeliveryError("NOTIFICATION_BINDING_SCOPE_INVALID", 422)
                if binding["target_type"] == "chat_id" and not set(body.event_types) <= GROUP_EVENTS:
                    raise ExternalDeliveryError("NOTIFICATION_GROUP_EVENT_DENIED", 422)
            values = body.model_dump(mode="json") | {"connection_id": body.connection_id, "id": uuid4()}
            row = await repo.add("policy", scope=scope, actor_id=actor_id, now=now, values=values)
            await repo.audit(scope=scope, action="policy.create", object_id=row["id"], actor_id=actor_id,
                outcome="CREATED", now=now)
            await uow.commit()
            return public_record("policy", row)

    @staticmethod
    async def _new_job(repo, *, scope, connection, binding, source_event_id, event_type, content,
                       entry_id, expires_at, purpose, actor_id, now):
        return await new_delivery_job(repo, scope=scope, connection=connection, binding=binding,
            source_event_id=source_event_id, event_type=event_type, content=content,
            entry_id=entry_id, expires_at=expires_at, purpose=purpose, actor_id=actor_id, now=now)

    async def set_policy_enabled(self, *, scope, actor_id, object_id, expected_version, enabled):
        self.enabled()
        now = now_utc()
        async with self._uow_factory() as uow:
            repo = uow.external_delivery
            if not await repo.change("policy", scope=scope, object_id=object_id, expected_version=expected_version,
                values={"enabled": enabled}, actor_id=actor_id, now=now):
                raise ExternalDeliveryError("NOTIFICATION_VERSION_CONFLICT")
            await repo.audit(scope=scope, action="policy.state", object_id=object_id, actor_id=actor_id,
                outcome="ENABLED" if enabled else "PAUSED", now=now)
            await uow.commit()


    async def open_entry(self, *, scope, actor_id, entry_id):
        async with self._uow_factory() as uow:
            entry = await uow.external_delivery.get("entry", scope=scope, object_id=entry_id)
        if entry is None or (entry["recipient_actor_id"] and entry["recipient_actor_id"] != actor_id):
            raise ExternalDeliveryError("NOTIFICATION_ENTRY_NOT_FOUND", 404)
        if entry["expires_at"] <= now_utc():
            raise ExternalDeliveryError("NOTIFICATION_ENTRY_EXPIRED", 410)
        scope = DeliveryScope(tenant_id=entry.get("tenant_id"), domain_id=str(entry["domain_id"]))
        decision = await self.authority.check(scope=scope, actor_id=actor_id, entry=entry, actionable=False)
        if decision is not None:
            raise ExternalDeliveryError(decision, 403)
        return {"route_key": entry["route_key"], "route_params": entry["route_params"]}

    async def resolve_entry(self, *, actor_id, entry_id):
        async with self._uow_factory() as uow:
            entry = await uow.external_delivery.entry_by_id(entry_id)
        if not entry or (entry["recipient_actor_id"] and entry["recipient_actor_id"] != actor_id):
            raise ExternalDeliveryError("NOTIFICATION_ENTRY_NOT_FOUND", 404)
        scope = DeliveryScope(domain_id=str(entry["domain_id"]), tenant_id=entry.get("tenant_id"))
        route = await self.open_entry(scope=scope, actor_id=actor_id, entry_id=entry_id)
        return {**route, **scope.model_dump(mode="json")}

    async def my_bindings(self, *, scope, actor_id):
        async with self._uow_factory() as uow:
            return {"items": [public_record("binding", row) for row in
                await uow.external_delivery.my_bindings(scope=scope, actor_id=actor_id)]}

    async def queue_test(self, *, scope, actor_id, object_id, binding_id):
        self.enabled()
        now = now_utc()
        async with self._uow_factory() as uow:
            repo = uow.external_delivery
            connection = await repo.get("connection", scope=scope, object_id=object_id)
            binding = await repo.get("binding", scope=scope, object_id=binding_id)
            if not connection or not connection["enabled"] or not binding or binding["connection_id"] != object_id or binding["status"] != "VERIFIED":
                raise ExternalDeliveryError("NOTIFICATION_TEST_TARGET_INVALID", 422)
            content = {"elements": [{"tag": "div", "text": {"tag": "plain_text",
                "content": "这是经管理员明确发起的通知渠道试发，不代表告警、审批或执行结果。"}}]}
            job = await self._new_job(repo, scope=scope, connection=connection, binding=binding,
                source_event_id=uuid4(), event_type="connection.test", content=content, entry_id=None,
                expires_at=now + timedelta(minutes=5), purpose="TEST", actor_id=actor_id, now=now)
            await repo.audit(scope=scope, action="connection.test", object_id=job["id"], actor_id=actor_id,
                outcome="EXPLICIT_TEST_QUEUED", now=now)
            await uow.commit()
        return {"job_id": job["id"], "status": "PENDING"}

    async def attempt_history(self, *, scope, job_id):
        async with self._uow_factory() as uow:
            return [public_record("attempt", row) for row in await uow.external_delivery.attempts(scope=scope, job_id=job_id)]

    async def health(self, *, scope):
        async with self._uow_factory() as uow:
            return {"items": await uow.external_delivery.health(scope=scope),
                    "metrics": await uow.external_delivery.metrics(scope=scope, now=now_utc())}

    async def retry(self, *, scope, actor_id, job_id, body):
        self.enabled()
        now = now_utc()
        async with self._uow_factory() as uow:
            repo = uow.external_delivery
            row = await repo.get("job", scope=scope, object_id=job_id, lock=True)
            if row is None:
                raise ExternalDeliveryError("NOTIFICATION_DELIVERY_NOT_FOUND", 404)
            scope = DeliveryScope(tenant_id=row.get("tenant_id"),
                domain_id=str(row["domain_id"]) if row["domain_id"] is not None else None)
            if row["row_version"] != body.expected_row_version or row["status"] not in {Status.UNKNOWN, Status.BLOCKED, Status.DEAD}:
                raise ExternalDeliveryError("NOTIFICATION_VERSION_CONFLICT")
            if row["purpose"] == "VERIFY" or row["expires_at"] <= now:
                raise ExternalDeliveryError("NOTIFICATION_DELIVERY_EXPIRED", 422)
            connection = await repo.get("connection", scope=scope, object_id=row["connection_id"])
            binding = await repo.get("binding", scope=scope, object_id=row["binding_id"])
            if (not connection or not binding or binding["status"] != "VERIFIED"
                or binding["actor_id"] != row["recipient_actor_id"] or binding["external_id"] != row["external_id"]):
                raise ExternalDeliveryError("NOTIFICATION_RETRY_TARGET_CHANGED", 422)
            refreshed_versions = {"connection_version": connection["credential_version"], "binding_version": binding["row_version"]}
            generation = int(row["generation"])
            new_generation = row["first_sent_at"] is not None and not can_retry_uncertain(row["first_sent_at"], now,
                window_seconds=self.config.dedupe_safe_seconds)
            if new_generation and not body.acknowledge_duplicate_risk:
                raise ExternalDeliveryError("NOTIFICATION_DUPLICATE_RISK_ACK_REQUIRED", 422)
            if not new_generation:
                await repo.change("job", scope=scope, object_id=job_id, expected_version=row["row_version"],
                    values={"status": Status.RETRY_WAIT, "available_at": now, "last_error_code": None,
                        "retry_count": 0, **refreshed_versions},
                    actor_id=actor_id, now=now)
                target_id = job_id
            else:
                generation += 1
                target_id = stable_id(row["id"], generation)
                values = {key: value for key, value in row.items()
                    if key not in {"created_at", "created_by", "updated_at", "updated_by", "row_version", "domain_id", "tenant_id"}}
                values.update(id=target_id, parent_job_id=job_id, generation=generation, provider_uuid=str(target_id),
                    status=Status.PENDING, attempt_count=0, retry_count=0, first_sent_at=None, network_started_at=None,
                    lease_token=None, lease_expires_at=None, message_id=None, last_error_code=None, available_at=now)
                values.update(refreshed_versions)
                if not await repo.add_once("job", scope=scope, actor_id=actor_id, now=now, values=values):
                    raise ExternalDeliveryError("NOTIFICATION_RETRY_ALREADY_CREATED")
                await repo.change("job", scope=scope, object_id=job_id, expected_version=row["row_version"],
                    values={"status": Status.UNKNOWN}, actor_id=actor_id, now=now)
            await repo.audit(scope=scope, action="delivery.retry", object_id=target_id, actor_id=actor_id,
                outcome="NEW_GENERATION_DUPLICATE_POSSIBLE" if new_generation else "SAME_PROVIDER_UUID", now=now)
            await uow.commit()
        return {"job_id": target_id}

    async def dispatch_once(self):
        if not self.config.enabled:
            return False
        now = now_utc()
        async with self._uow_factory() as uow:
            job = await uow.external_delivery.claim(now=now, lease_seconds=self.config.lease_seconds,
                dedupe_seconds=self.config.dedupe_safe_seconds)
            await uow.commit()
        if job is None:
            return False
        scope = DeliveryScope(domain_id=str(job["domain_id"]) if job["domain_id"] is not None else None,
                              tenant_id=job.get("tenant_id"))
        try:
            result, available = await self._deliver(scope=scope, job=job)
        except Exception:
            # 仅可能进入外部调用的异常标 UNKNOWN，不打印可能含凭据的异常对象。
            async with self._uow_factory() as uow:
                current = await uow.external_delivery.get("job", scope=scope, object_id=job["id"])
            uncertain = current is not None and current["network_started_at"] is not None
            result, available = SendResult(Status.UNKNOWN if uncertain else Status.BLOCKED,
                "NOTIFICATION_WORKER_UNCERTAIN" if uncertain else "NOTIFICATION_PREFLIGHT_FAILED"), now_utc()
        now = now_utc()
        attempt_status = result.status
        local_wait = result.error_code in {"NOTIFICATION_LOCAL_RATE_LIMIT", "NOTIFICATION_QUIET_HOURS"}
        retry_count = int(job.get("retry_count", 0)) + (0 if local_wait else 1)
        if result.status == Status.RETRY_WAIT and not local_wait and retry_count >= self.config.max_attempts:
            result = replace(result, status=Status.DEAD)
        elif result.status == Status.UNKNOWN and retry_count < self.config.max_attempts and can_retry_uncertain(
            job["first_sent_at"] or now, now, window_seconds=self.config.dedupe_safe_seconds):
            # UNKNOWN 仍保留尝试证据；同 uuid、冻结内容在安全窗口内重试。
            result = replace(result, status=Status.RETRY_WAIT)
        async with self._uow_factory() as uow:
            await uow.external_delivery.finish(job=job, result=result, now=now, available_at=available,
                attempt_status=attempt_status, retry_count=retry_count)
            await uow.commit()
        return True

    async def _deliver(self, *, scope, job):
        now = now_utc()
        async with self._uow_factory() as uow:
            repo = uow.external_delivery
            connection = await repo.get("connection", scope=scope, object_id=job["connection_id"])
            binding = await repo.get("binding", scope=scope, object_id=job["binding_id"])
            entry = await repo.get("entry", scope=scope, object_id=job["entry_id"]) if job["entry_id"] else None
            policies = await repo.active_policies(scope=scope) if job["purpose"] == "BUSINESS" else []
        if job["purpose"] == "BUSINESS" and not await self.authority.delivery_available(scope=scope):
            return SendResult(Status.BLOCKED, "NOTIFICATION_DOMAIN_UNAVAILABLE"), now
        if not connection or not connection["enabled"] or connection["credential_version"] != job["connection_version"]:
            return SendResult(Status.BLOCKED, "NOTIFICATION_CONNECTION_CHANGED"), now
        if not binding:
            return SendResult(Status.BLOCKED, "NOTIFICATION_BINDING_MISSING"), now
        if (binding["status"] == "DISABLED" or binding["row_version"] != job["binding_version"]
            or binding["external_id"] != job["external_id"] or binding["actor_id"] != job["recipient_actor_id"]):
            return SendResult(Status.CANCELLED, "NOTIFICATION_BINDING_CHANGED"), now
        if job["purpose"] == "BUSINESS":
            if not any(policy["connection_id"] == job["connection_id"]
                and job["event_type"] in policy["event_types"]
                and (str(job["binding_id"]) in {str(value) for value in policy["binding_ids"]}
                     or job["recipient_actor_id"] in (policy.get("actor_ids") or []))
                for policy in policies):
                return SendResult(Status.BLOCKED, "NOTIFICATION_POLICY_PAUSED"), now
            if binding["status"] != "VERIFIED":
                return SendResult(Status.BLOCKED, "NOTIFICATION_BINDING_UNVERIFIED"), now
            if not entry or entry["expires_at"] <= now:
                return SendResult(Status.EXPIRED, "NOTIFICATION_ENTRY_EXPIRED"), now
            decision = await self.authority.check(scope=scope, actor_id=binding["actor_id"], entry=entry,
                actionable=job["event_type"] in ACTIONABLE_EVENTS)
            if decision:
                return SendResult(Status.CANCELLED if decision == "NOTIFICATION_TASK_CLOSED" else Status.BLOCKED, decision), now
            if noise_controlled(job["event_type"], job.get("original_severity")):
                ready = after_quiet(self.config, now)
                if ready >= job["expires_at"]:
                    return SendResult(Status.EXPIRED, "NOTIFICATION_QUIET_EXCEEDS_TTL"), now
                if ready > now:
                    return SendResult(Status.RETRY_WAIT, "NOTIFICATION_QUIET_HOURS"), ready
        elif job["purpose"] == "VERIFY":
            if (not await self.authority.actor_active(scope=scope, actor_id=binding["actor_id"])
                or binding["challenge_expires_at"] is None or binding["challenge_expires_at"] <= now):
                return SendResult(Status.EXPIRED, "NOTIFICATION_VERIFICATION_EXPIRED"), now
        elif job["purpose"] == "TEST":
            if binding["status"] != "VERIFIED" or (binding["actor_id"] is not None and
                not await self.authority.actor_active(scope=scope, actor_id=binding["actor_id"])):
                return SendResult(Status.BLOCKED, "NOTIFICATION_TEST_TARGET_INVALID"), now
        else:
            return SendResult(Status.BLOCKED, "NOTIFICATION_PURPOSE_INVALID"), now
        content = job["content"]
        if payload_hash(content) != job["content_hash"]:
            return SendResult(Status.BLOCKED, "NOTIFICATION_PAYLOAD_HASH_MISMATCH"), now
        try:
            secret = (await self.vault.read(scope=scope, reference=connection["credential_ref"]))["app_secret"]
        except Exception:
            return SendResult(Status.BLOCKED, "NOTIFICATION_CREDENTIAL_UNAVAILABLE"), now
        if job["purpose"] == "VERIFY":
            code = (await self.vault.read(scope=scope, reference=content["verification_ref"]))["verification_code"]
            content = {"elements": [{"tag": "div", "text": {"tag": "plain_text",
                "content": f"飞书身份绑定验证码：{code}。请在本人已登录的系统中确认；不要转发验证码。"}}]}
        async with self._uow_factory() as uow:
            repo = uow.external_delivery
            error, ready_at = await repo.reserve_rate(scope=scope, connection=connection, binding=binding, now=now_utc(), config=self.config)
            if error:
                return SendResult(Status.BLOCKED, error), now
            if ready_at > now_utc():
                return SendResult(Status.RETRY_WAIT, "NOTIFICATION_LOCAL_RATE_LIMIT"), ready_at
            if not await repo.mark_network(job=job, now=now_utc()):
                return SendResult(Status.UNKNOWN, "NOTIFICATION_LEASE_LOST"), now
            await uow.commit()
        started = monotonic()
        result = await self.sender.send(connection=connection, secret=secret, target_type=job["target_type"],
            external_id=job["external_id"], content=content, provider_uuid=job["provider_uuid"])
        result = replace(result, elapsed_ms=max(0, int((monotonic() - started) * 1000)))
        delay = min(self.config.retry_max_seconds, self.config.retry_base_seconds * (2 ** min(int(job.get("retry_count", 0)), 10)))
        delay *= 1 + secrets.randbelow(101) / 100
        delay = max(delay, result.retry_after_seconds or 0)
        # 超长厂商等待提示不溢出 datetime，也不能截短后提前发送。
        now = now_utc()
        remaining = max(0, (job["expires_at"] - now).total_seconds())
        return result, now + timedelta(seconds=min(delay, remaining))
