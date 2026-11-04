"""系统案例资源的持久化事件与租约；Collection 写入仍归 KC 所有。"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select, text

from platform_core.app_resources.entity import CaseResourceEntity
from platform_core.identity import uuid7


def now():
    return datetime.now(timezone.utc)


SCOPE_SQL = """
SELECT domain.DOMAIN_ID AS "domain_id",
 CASE WHEN domain.STATUS='ACTIVE' AND EXISTS (SELECT 1 FROM KBOT_APP_DOMAIN binding
 WHERE binding.DOMAIN_ID=domain.DOMAIN_ID AND binding.APP_ID='aiops' AND binding.STATUS='ACTIVE')
 AND EXISTS (SELECT 1 FROM KBOT_PLATFORM_APP app WHERE app.APP_ID='aiops' AND app.STATUS='ACTIVE')
 AND NOT EXISTS (SELECT 1 FROM KBOT_APP_DOMAIN other_app WHERE other_app.DOMAIN_ID=domain.DOMAIN_ID
 AND other_app.APP_ID<>'aiops' AND other_app.STATUS='ACTIVE') THEN 1 ELSE 0 END AS "eligible"
 FROM KBOT_PLATFORM_DOMAIN domain WHERE domain.DOMAIN_ID=:domain_id
 AND EXISTS (SELECT 1 FROM KBOT_APP_DOMAIN binding WHERE binding.DOMAIN_ID=domain.DOMAIN_ID AND binding.APP_ID='aiops')
"""

LOCK_SQL = """
 FOR UPDATE OF domain.DOMAIN_ID
"""

ALL_SQL = """
SELECT domain.DOMAIN_ID AS "domain_id", COALESCE(domain.UPDATED_BY, domain.CREATED_BY) AS "actor_id"
 FROM KBOT_PLATFORM_DOMAIN domain WHERE EXISTS (SELECT 1 FROM KBOT_APP_DOMAIN binding
 WHERE binding.DOMAIN_ID=domain.DOMAIN_ID AND binding.APP_ID='aiops') ORDER BY domain.DOMAIN_ID
"""

DISCOVER_SQL = """
SELECT domain.DOMAIN_ID AS "domain_id", COALESCE(domain.UPDATED_BY, domain.CREATED_BY) AS "actor_id"
 FROM KBOT_PLATFORM_DOMAIN domain WHERE domain.DOMAIN_ID>:after_id
 AND EXISTS (SELECT 1 FROM KBOT_APP_DOMAIN binding WHERE binding.DOMAIN_ID=domain.DOMAIN_ID AND binding.APP_ID='aiops')
 ORDER BY domain.DOMAIN_ID FETCH FIRST :batch_limit ROWS ONLY
"""


class CaseResourceRepository:
    def __init__(self, session):
        self._session = session

    @staticmethod
    def domain_value(value):
        return int(value)

    def scope_params(self, *, domain_id, tenant_id=None):
        return {"domain_id": int(domain_id)}

    async def scope(self, *, domain_id, tenant_id=None, lock=False):
        params = self.scope_params(domain_id=domain_id, tenant_id=tenant_id)
        row = (await self._session.execute(text(SCOPE_SQL + (LOCK_SQL if lock else "")), params)).mappings().one_or_none()
        return row

    async def get(self, *, domain_id, tenant_id=None):
        row = await self._session.get(CaseResourceEntity, self.domain_value(domain_id))
        return row

    async def register(self, *, domain_id, tenant_id=None, actor_id, force=False):
        await self._session.flush()
        scope = await self.scope(domain_id=domain_id, tenant_id=tenant_id, lock=True)
        if scope is None:
            return
        row = await self.get(domain_id=domain_id, tenant_id=tenant_id)
        if actor_id is None:
            if row is None:
                raise ValueError("案例资源事件缺少审计操作者")
            actor_id = row.actor_id
        active = bool(scope["eligible"])
        if row is None:
            row = CaseResourceEntity(domain_id=self.domain_value(domain_id), actor_id=str(actor_id),
                status="QUEUED" if active else "PAUSED", configuration_status="UNKNOWN",
                missing_model_roles=[], attempt_count=0, next_attempt_at=now(), creation_allowed=1)
            # Oracle 的 Domain 本身就是隔离边界。
            self._session.add(row)
        elif (not active and row.status != "PAUSED") or (active and row.status == "PAUSED") or force:
            row.status = "QUEUED" if active else "PAUSED"
            row.lease_token, row.lease_until = None, None
            row.next_attempt_at = now()
            row.actor_id = str(actor_id)
        await self._session.flush()

    async def register_all(self, *, tenant_id=None, actor_id=None):
        # App 启停事务内登记所有自有 Domain，不受后台分页大小限制。
        rows = (await self._session.execute(text(ALL_SQL), {})).mappings().all()
        for row in rows:
            values = dict(row)
            if actor_id is not None:
                values["actor_id"] = actor_id
            await self.register(**values, force=True)

    async def discover(self, *, after=None, limit=100):
        query = text(DISCOVER_SQL)
        rows = (await self._session.execute(query,
            {"after_id": self.domain_value(after) if after is not None else 0, "batch_limit": limit})).mappings().all()
        return [dict(row) for row in rows]

    async def claim(self):
        instant = now()
        candidates = (await self._session.scalars(select(CaseResourceEntity.domain_id)
            .where(CaseResourceEntity.next_attempt_at <= instant,
                or_(CaseResourceEntity.lease_until.is_(None), CaseResourceEntity.lease_until <= instant))
            .order_by(CaseResourceEntity.next_attempt_at, CaseResourceEntity.domain_id).limit(20))).all()
        for domain_id in candidates:
            scope = await self.scope(domain_id=domain_id, tenant_id=None, lock=True)
            row = (await self._session.scalars(select(CaseResourceEntity)
                .where(CaseResourceEntity.domain_id == domain_id)
                .with_for_update(skip_locked=True).execution_options(populate_existing=True))).one_or_none()
            if row is None or row.next_attempt_at > instant or (row.lease_until and row.lease_until > instant):
                continue
            if scope is None or not scope["eligible"]:
                row.status = "PAUSED"
                row.lease_token, row.lease_until = None, None
                row.next_attempt_at = instant + timedelta(seconds=60)
                continue
            row.status = "RUNNING"
            row.lease_token, row.lease_until = str(uuid7()), instant + timedelta(seconds=60)
            row.attempt_count += 1
            await self._session.flush()
            return {"domain_id": row.domain_id, "tenant_id": None,
                "actor_id": row.actor_id, "collection_id": row.collection_id,
                "allow_create": bool(row.creation_allowed), "lease_token": row.lease_token}
        return None

    async def complete(self, item, *, result=None, error=None, retryable=True):
        scope = await self.scope(domain_id=item["domain_id"], tenant_id=item["tenant_id"], lock=True)
        row = (await self._session.scalars(select(CaseResourceEntity)
            .where(CaseResourceEntity.domain_id == item["domain_id"])
            .with_for_update().execution_options(populate_existing=True))).one_or_none()
        if row is None:
            return
        # 即使已被新事件撤销，也保留成功的资源身份；绝不让迟到失败覆盖新租约。
        if result is not None:
            if row.collection_id is not None and row.collection_id != result.collection_id:
                error, result, retryable = "CASE_RESOURCE_CONFLICT", None, False
            else:
                row.collection_id = result.collection_id
                row.creation_allowed = 0
        if row.lease_token != item["lease_token"]:
            await self._session.flush()
            return
        row.lease_token, row.lease_until = None, None
        row.error_code = error
        if error in {"CASE_RESOURCE_CONFLICT", "CASE_RESOURCE_KNOWN_COLLECTION_MISSING",
            "CASE_RESOURCE_BINDING_CONFLICT"}:
            # 资源冲突曾被确认后，即使后续超时或原库被删，也不能再走首次创建。
            row.creation_allowed = 0
        row.status = "READY" if result else ("QUEUED" if retryable else "FAILED")
        if scope is None or not scope["eligible"]:
            row.status = "PAUSED"
        if result:
            row.configuration_status = result.configuration_status
            row.missing_model_roles = result.missing_model_roles
            row.attempt_count = 0
        elif error:
            row.configuration_status = "UNKNOWN"
        delay = 60 if result or row.status == "PAUSED" else (
            min(2 ** min(int(row.attempt_count), 8), 300) if retryable else 300)
        row.next_attempt_at = now() + timedelta(seconds=delay)
        await self._session.flush()
