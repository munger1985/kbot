"""外部通知的持久化、主键幂等与租约 CAS；事务由应用层提交。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sqlalchemy import and_, func, insert, or_, select, update
from sqlalchemy.exc import IntegrityError

from main_api.entities.notification_delivery import external_tables
from platform_core.notifications.delivery import DeliveryScope, DeliveryStatus as Status, stable_id


class ExternalDeliveryRepository:
    def __init__(self, session):
        self._session = session
        self.tables = external_tables

    def scope_values(self, scope, table=None):
        values = {}
        if "tenant_id" in self.tables["connection"].c:
            if scope.tenant_id is None:
                raise ValueError("NOTIFICATION_TENANT_REQUIRED")
            values["tenant_id"] = scope.tenant_id
        elif scope.tenant_id is not None:
            raise ValueError("NOTIFICATION_TENANT_UNEXPECTED")
        if table is not None and "domain_id" in table.c:
            values["domain_id"] = (None if scope.domain_id is None else
                UUID(scope.domain_id) if "tenant_id" in self.tables["connection"].c else int(scope.domain_id))
        return values

    def scoped(self, table, scope):
        return [table.c[key] == value for key, value in self.scope_values(scope, table).items()
                if key != "domain_id" or scope.domain_id is not None]

    async def settings(self, *, scope):
        from platform_core.notifications.delivery import notification_settings_id
        return await self.get("settings", scope=scope, object_id=notification_settings_id(scope))

    async def lock_configuration(self):
        """用部署初始化的全局频控行串行化渠道创建，不依赖业务唯一索引。"""
        table = self.tables["throttle"]
        row = (await self._session.execute(select(table.c.id).where(
            table.c.id == "external").with_for_update())).scalar_one_or_none()
        if row is None:
            raise ValueError("NOTIFICATION_THROTTLE_NOT_INITIALIZED")

    async def connection_for_provider(self, *, scope, provider="FEISHU", lock=False):
        table = self.tables["connection"]
        query = select(table).where(*self.scoped(table, scope), table.c.provider == provider)
        if lock:
            query = query.with_for_update()
        rows = (await self._session.execute(query)).mappings().all()
        if len(rows) > 1:
            raise ValueError("NOTIFICATION_CONNECTION_SCOPE_CONFLICT")
        return dict(rows[0]) if rows else None

    async def get(self, kind, *, scope, object_id, lock=False):
        table = self.tables[kind]
        query = select(table).where(table.c.id == object_id, *self.scoped(table, scope))
        if lock:
            query = query.with_for_update()
        row = (await self._session.execute(query)).mappings().one_or_none()
        return dict(row) if row else None

    async def list(self, kind, *, scope, limit=100, offset=0, status=None,
                   connection_id=None, event_type=None, recipient_actor_id=None,
                   created_from=None, created_to=None):
        table = self.tables[kind]
        query = select(table).where(*self.scoped(table, scope))
        if status is not None:
            query = query.where(table.c.status == status)
        for field, value in (("connection_id", connection_id), ("event_type", event_type),
                             ("recipient_actor_id", recipient_actor_id)):
            if value is not None:
                query = query.where(table.c[field] == value)
        if created_from is not None:
            query = query.where(table.c.created_at >= created_from)
        if created_to is not None:
            query = query.where(table.c.created_at < created_to)
        rows = (await self._session.execute(query
            .order_by(table.c.created_at.desc(), table.c.id).offset(offset).limit(limit))).mappings().all()
        return [dict(row) for row in rows]

    async def add(self, kind, *, scope, values, actor_id, now):
        table = self.tables[kind]
        record = {**values, **self.scope_values(scope, table), "created_by": actor_id, "updated_by": actor_id,
                  "created_at": now, "updated_at": now, "row_version": 1}
        await self._session.execute(insert(table).values(**record))
        return record

    async def add_once(self, kind, *, scope, values, actor_id, now):
        # 保存点只吸收已存在同一主键；外键／NOT NULL 错误仍上抛。
        try:
            async with self._session.begin_nested():
                await self.add(kind, scope=scope, values=values, actor_id=actor_id, now=now)
            return True
        except IntegrityError as exc:
            origin = exc.orig
            oracle_error = origin.args[0] if getattr(origin, "args", ()) else None
            duplicate = (getattr(origin, "sqlstate", None) == "23505"
                or getattr(origin, "sqlite_errorcode", None) in {1555, 2067}
                or getattr(oracle_error, "code", None) == 1)
            if not duplicate:
                raise
            if await self.get(kind, scope=scope, object_id=values["id"]) is None:
                raise
            return False

    async def change(self, kind, *, scope, object_id, expected_version, values, actor_id, now):
        table = self.tables[kind]
        result = await self._session.execute(update(table).where(table.c.id == object_id,
            table.c.row_version == expected_version, *self.scoped(table, scope)).values(
                **values, row_version=expected_version + 1, updated_by=actor_id, updated_at=now))
        return result.rowcount == 1

    async def audit(self, *, scope, action, object_id, actor_id, outcome, now):
        await self.add("audit", scope=scope, actor_id=actor_id, now=now,
            values={"id": uuid4(), "action": action, "object_id": object_id, "outcome": outcome})

    async def attempts(self, *, scope, job_id):
        table = self.tables["attempt"]
        rows = (await self._session.execute(select(table).where(table.c.job_id == job_id,
            *self.scoped(table, scope)).order_by(table.c.attempt_number))).mappings().all()
        return [dict(row) for row in rows]

    async def entry_by_id(self, entry_id):
        table = self.tables["entry"]
        row = (await self._session.execute(select(table).where(table.c.id == entry_id))).mappings().one_or_none()
        return dict(row) if row else None

    async def my_bindings(self, *, scope, actor_id):
        table = self.tables["binding"]
        rows = (await self._session.execute(select(table).where(
            *self.scoped(table, scope), table.c.actor_id == actor_id))).mappings().all()
        return [dict(row) for row in rows]

    async def health(self, *, scope):
        table = self.tables["job"]
        rows = (await self._session.execute(select(table.c.status, table.c.last_error_code, func.count().label("count"),
            func.min(table.c.created_at).label("oldest_created_at")).where(*self.scoped(table, scope))
            .group_by(table.c.status, table.c.last_error_code))).mappings().all()
        return [dict(row) for row in rows]

    async def metrics(self, *, scope, now):
        """仅按固定状态及安全错误码聚合，避免用户和资源 ID 高基数。"""
        attempt = self.tables["attempt"]
        rows = (await self._session.execute(select(attempt.c.status, attempt.c.error_code,
            func.count().label("count"), func.avg(attempt.c.elapsed_ms).label("average_elapsed_ms"))
            .where(*self.scoped(attempt, scope), attempt.c.started_at >= now - timedelta(days=1))
            .group_by(attempt.c.status, attempt.c.error_code))).mappings().all()
        total = sum(int(row["count"]) for row in rows if row["status"] != Status.SENDING)
        accepted = sum(int(row["count"]) for row in rows if row["status"] == Status.ACCEPTED)
        return {"window_seconds": 86400, "attempts": [dict(row) for row in rows],
                "accepted_attempts": accepted, "finished_attempts": total,
                "acceptance_rate": accepted / total if total else None}

    async def record_verification_failure(self, *, scope, binding, now):
        table = self.tables["binding"]
        await self._session.execute(update(table).where(table.c.id == binding["id"],
            *self.scoped(table, scope), table.c.row_version == binding["row_version"]).values(
                challenge_attempts=table.c.challenge_attempts + 1, updated_at=now))

    async def bindings_for_connection(self, *, scope, connection_id):
        table = self.tables["binding"]
        rows = (await self._session.execute(select(table).where(
            *self.scoped(table, scope), table.c.connection_id == connection_id))).mappings().all()
        return [dict(row) for row in rows]


    async def active_policies(self, *, scope):
        table = self.tables["policy"]
        rows = (await self._session.execute(select(table).where(
            *self.scoped(table, scope), table.c.enabled.is_(True))
            .order_by(table.c.connection_id, table.c.id))).mappings().all()
        return [dict(row) for row in rows]

    async def coalesce_candidates(self, *, scope, coalesce_key, exclude_id):
        table = self.tables["job"]
        rows = (await self._session.execute(select(table).where(
            *self.scoped(table, scope), table.c.coalesce_key == coalesce_key,
            table.c.id != exclude_id, table.c.purpose == "BUSINESS",
            table.c.first_sent_at.is_(None), table.c.status.in_((Status.PENDING, Status.RETRY_WAIT)))
            .with_for_update())).mappings().all()
        return [dict(row) for row in rows]

    async def claim(self, *, now, lease_seconds, dedupe_seconds):
        """先选候选再原子条件更新；不使用 Oracle 不支持的 FETCH + FOR UPDATE。"""
        table = self.tables["job"]
        due = and_(table.c.status.in_((Status.PENDING, Status.RETRY_WAIT)), table.c.available_at <= now)
        lost = and_(table.c.status == Status.SENDING, table.c.lease_expires_at <= now)
        rows = (await self._session.execute(select(table).where(or_(due, lost))
            .order_by(table.c.available_at, table.c.id).limit(20))).mappings().all()
        for row in rows:
            scope = DeliveryScope(domain_id=str(row["domain_id"]) if row["domain_id"] is not None else None,
                                  tenant_id=row.get("tenant_id"))
            version = int(row["row_version"])
            if row["status"] == Status.SENDING:
                # 租约失效不说明网络调用未发生；已记录网络起点的任务先进入 UNKNOWN。
                status = Status.UNKNOWN if row["network_started_at"] else Status.RETRY_WAIT
                changed = await self.change("job", scope=scope, object_id=row["id"], expected_version=version,
                    values={"status": status, "lease_token": None, "lease_expires_at": None,
                            "last_error_code": "NOTIFICATION_LEASE_LOST", "available_at": now},
                    actor_id="system:external-delivery", now=now)
                if changed:
                    attempt = self.tables["attempt"]
                    await self._session.execute(update(attempt).where(attempt.c.job_id == row["id"],
                        attempt.c.lease_token == row["lease_token"], attempt.c.finished_at.is_(None)).values(
                            status=status, error_code="NOTIFICATION_LEASE_LOST", finished_at=now))
                continue
            if row["expires_at"] <= now:
                await self.change("job", scope=scope, object_id=row["id"], expected_version=version,
                    values={"status": Status.EXPIRED}, actor_id="system:external-delivery", now=now)
                continue
            first = row["first_sent_at"]
            if first is not None and (first > now or now >= first + timedelta(seconds=dedupe_seconds)):
                await self.change("job", scope=scope, object_id=row["id"], expected_version=version,
                    values={"status": Status.UNKNOWN, "last_error_code": "NOTIFICATION_DEDUPE_WINDOW_CLOSED"},
                    actor_id="system:external-delivery", now=now)
                continue
            lease, attempt_number = uuid4(), int(row["attempt_count"]) + 1
            changed = await self.change("job", scope=scope, object_id=row["id"], expected_version=version,
                values={"status": Status.SENDING, "lease_token": lease,
                    "lease_expires_at": now + timedelta(seconds=lease_seconds),
                    "network_started_at": None, "attempt_count": attempt_number},
                actor_id="system:external-delivery", now=now)
            if not changed:
                continue
            await self.add("attempt", scope=scope, actor_id="system:external-delivery", now=now,
                values={"id": stable_id(row["id"], attempt_number), "job_id": row["id"],
                    "attempt_number": attempt_number, "lease_token": lease,
                    "started_at": now, "status": Status.SENDING})
            return {**dict(row), "lease_token": lease, "attempt_count": attempt_number, "updated_at": now,
                    "row_version": version + 1, "status": Status.SENDING}
        return None

    async def mark_network(self, *, job, now):
        table = self.tables["job"]
        result = await self._session.execute(update(table).where(table.c.id == job["id"],
            table.c.status == Status.SENDING, table.c.row_version == job["row_version"],
            table.c.lease_token == job["lease_token"], table.c.lease_expires_at > now,
            table.c.expires_at > now).values(
                network_started_at=now, first_sent_at=job["first_sent_at"] or now))
        return result.rowcount == 1

    async def finish(self, *, job, result, now, available_at, attempt_status=None, retry_count=None):
        table = self.tables["job"]
        changed = await self._session.execute(update(table).where(table.c.id == job["id"],
            table.c.status == Status.SENDING, table.c.row_version == job["row_version"],
            table.c.lease_token == job["lease_token"], table.c.lease_expires_at > now).values(
                status=result.status, last_error_code=result.error_code, message_id=result.message_id,
                available_at=available_at, lease_token=None, lease_expires_at=None,
                updated_at=now, row_version=table.c.row_version + 1))
        if changed.rowcount != 1:
            return False
        attempt = self.tables["attempt"]
        await self._session.execute(update(attempt).where(attempt.c.job_id == job["id"],
            attempt.c.lease_token == job["lease_token"]).values(status=attempt_status or result.status,
                error_code=result.error_code, message_id=result.message_id, request_id=result.request_id,
                finished_at=now, elapsed_ms=result.elapsed_ms))
        if retry_count is not None:
            await self._session.execute(update(table).where(table.c.id == job["id"]).values(retry_count=retry_count))
        return True

    async def reserve_rate(self, *, scope, connection, binding, now, config):
        # 所有 Worker 都按全局→连接→绑定的固定锁顺序预留，避免死锁和进程级 QPS 漏洞。
        throttle = self.tables["throttle"]
        global_row = (await self._session.execute(select(throttle).where(
            throttle.c.id == "external").with_for_update())).mappings().one_or_none()
        if global_row is None:
            return "NOTIFICATION_THROTTLE_NOT_INITIALIZED", None
        locked_connection = await self.get("connection", scope=scope, object_id=connection["id"], lock=True)
        locked_binding = await self.get("binding", scope=scope, object_id=binding["id"], lock=True)
        if (locked_connection is None or locked_binding is None
            or not locked_connection["enabled"] or locked_connection["row_version"] != connection["row_version"]
            or locked_binding["row_version"] != binding["row_version"]):
            return "NOTIFICATION_CONFIGURATION_CHANGED", None
        waits = [global_row["next_send_at"], locked_connection["next_send_at"], locked_binding["next_send_at"]]
        ready = max([value for value in waits if value is not None] + [now])
        if ready > now:
            return None, ready
        await self._session.execute(update(throttle).where(throttle.c.id == "external").values(
            next_send_at=now + timedelta(seconds=config.global_interval_seconds)))
        for table, record, interval in ((self.tables["connection"], connection, config.connection_interval_seconds),
                                       (self.tables["binding"], binding, config.target_interval_seconds)):
            await self._session.execute(update(table).where(table.c.id == record["id"]).values(
                next_send_at=now + timedelta(seconds=interval)))
        return None, now
