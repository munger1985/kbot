"""有界短期访问观测；缓存只减少重复拒绝，不能授权或代替当前成功取证。"""

from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import secrets
import time

from sqlglot import exp, parse
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

from platform_core.contracts.aiops.executor import ObjectAccessObservation

from aiops_agent.diagnostics.grants import canonical_sha256


CACHED_ACCESS_DENIAL = "PRIVILEGE_PROBE_CACHED_DENIAL"
OBJECT_CACHED_ACCESS_DENIAL = "OBJECT_ACCESS_CACHED_DENIAL"


@dataclass(frozen=True)
class ObjectReadScope:
    object_ref: str
    columns: tuple[str, ...]
    is_probe: bool


def object_read_scope(sql, database_type):
    """保守提取单对象显式列；复杂语句仍可正常执行，但不据此复用对象权限。"""
    try:
        dialect = {"POSTGRESQL": "postgres", "MYSQL": "mysql", "ORACLE": "oracle"}[database_type]
        statements = parse(sql, read=dialect)
        if len(statements) != 1 or not isinstance(statements[0], exp.Select):
            return None
        expression = statements[0]
        if any(expression.find(kind) is not None for kind in (
                exp.CTE, exp.Subquery, exp.Join, exp.Star, exp.Func, exp.Window, exp.Into, exp.Lock)):
            return None
        if len(tuple(expression.find_all(exp.Select))) != 1:
            return None
        tables = tuple(expression.find_all(exp.Table))
        if len(tables) != 1 or not isinstance(tables[0].this, exp.Identifier):
            return None
        table = tables[0]
        # 别名列重命名、时间旅行、分区选择等不是普通对象读取，不跨查询推断。
        if any(value for key, value in table.args.items() if key not in {"this", "db", "catalog", "alias"}):
            return None
        alias = table.args.get("alias")
        if alias is not None and alias.args.get("columns"):
            return None
        if database_type != "MYSQL":
            expression = normalize_identifiers(expression, dialect=dialect)
            table = next(expression.find_all(exp.Table))
        columns = tuple(expression.find_all(exp.Column))
        if not columns:
            return None
        allowed_qualifiers = {table.name, table.alias_or_name}
        aliases = {item.alias for item in expression.expressions if isinstance(item, exp.Alias)}
        projection_columns = {id(column) for item in expression.expressions for column in item.find_all(exp.Column)}
        for column in columns:
            if (not isinstance(column.this, exp.Identifier)
                    or column.table and column.table not in allowed_qualifiers
                    or column.db and column.db != table.db
                    or column.catalog and column.catalog != table.catalog):
                return None
            if not column.table and column.name in aliases and id(column) not in projection_columns:
                return None
        # MySQL 列名不区分大小写；表名依赖服务器配置，保留原拼写以免误合并。
        names = tuple(sorted({column.name.casefold() if database_type == "MYSQL"
                              else column.this.sql(dialect=dialect, comments=False) for column in columns}))
        reference = table.copy()
        reference.set("alias", None)
        object_ref = reference.sql(dialect=dialect, comments=False)
        if len(object_ref) > 256 or len(names) > 128 or any(len(name) > 256 or name == "*" for name in names):
            return None
        where = expression.args.get("where")
        condition = where.this.unnest() if where is not None else None
        is_false = isinstance(condition, exp.Boolean) and condition.this is False
        if isinstance(condition, exp.EQ):
            left, right = condition.left, condition.right
            is_false = (isinstance(left, exp.Literal) and isinstance(right, exp.Literal)
                        and not left.is_string and not right.is_string
                        and {left.this, right.this} == {"0", "1"})
        # 探测只有直接列投影；不以表达式错误或额外谓词权限推断对象拒绝。
        simple = not any(value for key, value in expression.args.items()
                         if key not in {"expressions", "from_", "from", "where", "limit", "offset"})
        is_probe = simple and is_false and all(isinstance(item.unalias(), exp.Column) for item in expression.expressions)
        return ObjectReadScope(object_ref, names, is_probe)
    except Exception:
        return None


def catalog_object_group(sql, database_type):
    """只记录查询引用的对象组；联合查询失败不推断其中每个对象都无权限。"""
    try:
        expression = parse(sql, read={"POSTGRESQL": "postgres", "MYSQL": "mysql", "ORACLE": "oracle"}[database_type])[0]
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
    object_access: tuple[ObjectAccessObservation, ...] = ()


    def public_metadata(self):
        return {"schema_version": "QUERY_ACCESS_OBSERVATION.v1", "state": self.state,
                "checked_at": self.checked_at.isoformat(), "expires_at": self.expires_at.isoformat(),
                "object_group": list(self.objects), "scope": "EXACT_QUERY_AND_PARAMETERS",
                "object_group_is_exhaustive": False, "grants_authority": False}


@dataclass(frozen=True)
class CachedObjectAccess:
    scope_key: str
    observation: ObjectAccessObservation
    monotonic_expires_at: float

    @property
    def checked_at(self):
        return self.observation.checked_at

    @property
    def expires_at(self):
        return self.observation.expires_at


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
        return canonical_sha256({"scope": self.scope_key(grant, issued=issued),
            "query": query_sha256, "parameters": grant.parameters_sha256,
            "tool": (grant.tool_id, grant.tool_version, grant.variant)})

    def scope_key(self, grant, *, issued):
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
            "query_policy": grant.model_dump(mode="json", include={"policy_sha256"})})

    def get(self, key, *, touch=True):
        observed = self._entries.get(key)
        if observed is None:
            return None
        now = self._clock()
        if (self._monotonic() >= observed.monotonic_expires_at
                or not observed.checked_at <= now < observed.expires_at):
            self._entries.pop(key, None)
            return None
        if touch:
            self._entries.move_to_end(key)
        return observed

    def _put(self, key, observed):
        self._entries[key] = observed
        self._entries.move_to_end(key)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)
        return observed

    def remember(self, key, *, state, objects=(), object_access=()):
        if state not in {"SUCCEEDED", "DENIED"}:
            raise ValueError("仅保存实际成功或明确权限拒绝，不缓存超时、认证或对象缺失")
        seconds = 30 if state == "DENIED" else 300
        now = self._clock()
        bounded_objects = tuple(item for item in objects[:64] if isinstance(item, str) and len(item) <= 256)
        observed = AccessObservation(state, now, now + timedelta(seconds=seconds),
                                     self._monotonic() + seconds, bounded_objects, object_access)
        return self._put(key, observed)

    def object_observation(self, scope_key, *, scope, state, request_id, query_sha256, database_type):
        """只记录实际零行探测；不可见对象不作为明确拒绝缓存。"""
        if not scope.is_probe:
            raise ValueError("对象访问观察只能来自显式零行探测")
        now = self._clock()
        seconds = 300 if state == "READABLE" else 30
        observation = ObjectAccessObservation(db_type=database_type, object_ref=scope.object_ref,
            read_columns=scope.columns, state=state, checked_at=now,
            expires_at=now + timedelta(seconds=seconds), source_executor_request_id=request_id,
            source_query_sha256=query_sha256)
        if state in {"READABLE", "DENIED"}:
            # 后来的成功不能被旧拒绝覆盖；多列拒绝仍不猜测其中哪一列受限。
            if state == "READABLE":
                for key in tuple(self._entries):
                    cached = self.get(key, touch=False)
                    if (isinstance(cached, CachedObjectAccess) and cached.scope_key == scope_key
                            and cached.observation.object_ref == scope.object_ref
                            and cached.observation.state == "DENIED"
                            and set(cached.observation.read_columns) <= set(scope.columns)):
                        self._entries.pop(key, None)
            key = ("OBJECT", scope_key, scope.object_ref, scope.columns)
            self._put(key, CachedObjectAccess(scope_key, observation, self._monotonic() + seconds))
        return observation

    def find_object(self, scope_key, scope):
        """拒绝集合必须被新查询完整需要；成功历史提示不能替代现场查询。"""
        if scope is None:
            return None
        readable = None
        readable_key = None
        for key in reversed(tuple(self._entries)):
            cached = self.get(key, touch=False)
            if (not isinstance(cached, CachedObjectAccess) or cached.scope_key != scope_key
                    or cached.observation.object_ref != scope.object_ref):
                continue
            observed = cached.observation
            if observed.state == "DENIED" and set(observed.read_columns) <= set(scope.columns):
                self._entries.move_to_end(key)
                return observed.model_copy(update={"reused": True})
            if readable is None and observed.state == "READABLE" and set(scope.columns) <= set(observed.read_columns):
                readable = observed.model_copy(update={"reused": True})
                readable_key = key
        if readable_key is not None:
            self._entries.move_to_end(readable_key)
        return readable
