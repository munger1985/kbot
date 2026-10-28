"""有界短期访问观测；缓存只减少重复拒绝，不能授权或代替当前成功取证。"""

from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import secrets
import time

from sqlglot import exp, parse_one

from aiops_agent.diagnostics.grants import canonical_sha256


CACHED_ACCESS_DENIAL = "PRIVILEGE_PROBE_CACHED_DENIAL"


def catalog_object_group(sql, database_type):
    """只记录查询引用的对象组；联合查询失败不推断其中每个对象都无权限。"""
    try:
        expression = parse_one(sql, read={"POSTGRESQL": "postgres", "MYSQL": "mysql", "ORACLE": "oracle"}[database_type])
        aliases = {item.alias for item in expression.find_all(exp.CTE)}
        objects = sorted({item.sql() for item in expression.find_all(exp.Table)
                          if item.name not in aliases})
        if len(objects) > 64 or any(len(item) > 256 for item in objects):
            return ()
        return tuple(objects)
    except Exception:
        # 诊断模板已在目录校验；元数据提取失败不影响主查询或扩大权限。
        return ()


@dataclass(frozen=True)
class AccessObservation:
    state: str
    checked_at: datetime
    expires_at: datetime
    monotonic_expires_at: float
    objects: tuple[str, ...]

    def public_metadata(self):
        return {"schema_version": "QUERY_ACCESS_OBSERVATION.v1", "state": self.state,
                "checked_at": self.checked_at.isoformat(), "expires_at": self.expires_at.isoformat(),
                "object_group": list(self.objects), "scope": "EXACT_QUERY_AND_PARAMETERS",
                "object_group_is_exhaustive": False, "grants_authority": False}


class QueryAccessObservations:
    """每进程最多保存 1024 项；成功有效期五分钟，拒绝仅保留三十秒。"""

    def __init__(self, *, capacity=1024, monotonic=time.monotonic, clock=None):
        if not 1 <= capacity <= 4096:
            raise ValueError("访问观测容量超出范围")
        self._capacity = capacity
        self._monotonic = monotonic
        self._clock = clock or (lambda: datetime.now(UTC))
        self._entries = OrderedDict()
        self._credential_key = secrets.token_bytes(32)

    def key(self, grant, *, issued, query_sha256):
        # 只在当前 Grant 兑换成功后使用。随机 HMAC 防止保留可离线枚举的口令摘要；
        # 同一凭据 ID 的实际内容变化也立即失效，不把口令、连接串或参数保存在缓存中。
        credential = hmac.new(self._credential_key,
            (issued.username + "\0" + issued.password).encode("utf-8"), hashlib.sha256).hexdigest()
        return canonical_sha256({"authorization_scope": grant.model_dump(mode="json", include={"tenant_id", "domain_id"}),
            "target": str(grant.target_id), "target_version": grant.target_row_version,
            "credential_id": str(grant.diagnostic_credential_id), "credential": credential,
            "database_type": grant.db_type,
            "profile": grant.connection_profile.model_dump(mode="json"),
            "capabilities": grant.capability_snapshot_hash,
            "query": query_sha256, "parameters": grant.parameters_sha256,
            "tool": (grant.tool_id, grant.tool_version, grant.variant),
            "query_policy": grant.model_dump(mode="json", include={"policy_sha256"})})

    def get(self, key):
        observed = self._entries.get(key)
        if observed is None:
            return None
        now = self._clock()
        if (self._monotonic() >= observed.monotonic_expires_at
                or not observed.checked_at <= now < observed.expires_at):
            self._entries.pop(key, None)
            return None
        self._entries.move_to_end(key)
        return observed

    def remember(self, key, *, state, objects=()):
        if state not in {"SUCCEEDED", "DENIED"}:
            raise ValueError("仅保存实际成功或明确权限拒绝，不缓存超时、认证或对象缺失")
        seconds = 30 if state == "DENIED" else 300
        now = self._clock()
        bounded_objects = tuple(item for item in objects[:64] if isinstance(item, str) and len(item) <= 256)
        observed = AccessObservation(state, now, now + timedelta(seconds=seconds),
                                     self._monotonic() + seconds, bounded_objects)
        self._entries[key] = observed
        self._entries.move_to_end(key)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)
        return observed
