"""PromQL 与 LogQL 临时取证查询的确定性安全策略。"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import timedelta

import promql_parser
from pydantic import BaseModel, ConfigDict, Field


_EXTERNAL_SENTINEL = "__KBOT_EXTERNAL_TARGET__"
_HOST_SENTINEL = "__KBOT_HOST_TARGET__"
_PROM_PLACEHOLDERS = {
    "${external_target}": _EXTERNAL_SENTINEL,
    "${host_target}": _HOST_SENTINEL,
}
_LOG_FILTER = re.compile(r'^\s*(\|=|!=)\s*"((?:[^"\\]|\\.)*)"\s*')


class MonitoringQueryRejected(ValueError):
    """临时监控查询未通过确定性策略。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PromQueryPolicySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "PROM_QUERY_POLICY.v1"
    max_query_chars: int = Field(default=2000, ge=1, le=20_000)
    max_vector_selectors: int = Field(default=12, ge=1, le=64)
    max_range_seconds: int = Field(default=3600, ge=60, le=86_400)
    max_window_seconds: int = Field(default=3600, ge=60, le=86_400)
    min_step_seconds: int = Field(default=30, ge=1, le=3600)
    max_points: int = Field(default=240, ge=2, le=5000)
    max_series: int = Field(default=100, ge=1, le=1000)


class ValidatedPromQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "VALIDATED_PROM_QUERY.v1"
    normalized_query: str
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_scopes: tuple[str, ...]
    metric_names: tuple[str, ...]
    window_seconds: int
    step_seconds: int
    max_series: int


class PromQueryPolicy:
    """使用 Prometheus 语法 AST 验证每个选择器都被 Target 约束。"""

    def __init__(self, snapshot: PromQueryPolicySnapshot) -> None:
        self.snapshot = snapshot

    def validate(
        self, query: str, *, window_seconds: int | None = None
    ) -> ValidatedPromQuery:
        if not query.strip() or len(query) > self.snapshot.max_query_chars:
            raise MonitoringQueryRejected(
                "PROMQL_LENGTH_INVALID", "PromQL 为空或超过长度限制"
            )
        substituted = query.strip()
        for placeholder, sentinel in _PROM_PLACEHOLDERS.items():
            substituted = substituted.replace(placeholder, sentinel)
        if "${" in substituted:
            raise MonitoringQueryRejected(
                "PROMQL_PLACEHOLDER_INVALID", "PromQL 包含未知占位符"
            )
        try:
            expression = promql_parser.parse(substituted)
        except ValueError as exc:
            raise MonitoringQueryRejected(
                "PROMQL_PARSE_FAILED", "PromQL 语法解析失败"
            ) from exc
        vectors = []
        ranges = []

        def collect(node):
            if isinstance(node, promql_parser.VectorSelector):
                vectors.append(node)
            elif isinstance(node, promql_parser.MatrixSelector):
                ranges.append(node.range)

        promql_parser.walk(expression, pre_visit=collect)
        if not vectors or len(vectors) > self.snapshot.max_vector_selectors:
            raise MonitoringQueryRejected(
                "PROMQL_SELECTOR_COUNT_INVALID",
                "PromQL 向量选择器数量不符合策略",
            )
        scopes: set[str] = set()
        metrics: set[str] = set()
        for vector in vectors:
            metric_name = str(vector.name or "")
            if not re.fullmatch(r"[a-zA-Z_:][a-zA-Z0-9_:]*", metric_name):
                raise MonitoringQueryRejected(
                    "PROMQL_METRIC_INVALID", "PromQL 指标名称无效"
                )
            metrics.add(metric_name)
            if vector.at is not None or vector.offset is not None:
                raise MonitoringQueryRejected(
                    "PROMQL_TIME_MODIFIER_FORBIDDEN",
                    "PromQL 禁止 @ 和 offset 时间逃逸",
                )
            exact = {
                (matcher.name, matcher.value)
                for matcher in vector.matchers.matchers
                if matcher.op == promql_parser.MatchOp.Equal
            }
            if ("instance", _EXTERNAL_SENTINEL) in exact:
                scopes.add("DATABASE")
            elif ("target_key", _HOST_SENTINEL) in exact:
                scopes.add("HOST")
            else:
                raise MonitoringQueryRejected(
                    "PROMQL_TARGET_SCOPE_REQUIRED",
                    "每个 PromQL 向量选择器必须精确绑定数据库或主机 Target",
                )
        if any(
            item > timedelta(seconds=self.snapshot.max_range_seconds)
            for item in ranges
        ):
            raise MonitoringQueryRejected(
                "PROMQL_RANGE_EXCEEDED", "PromQL Range Selector 超过策略"
            )
        window = window_seconds or self.snapshot.max_window_seconds
        if not 60 <= window <= self.snapshot.max_window_seconds:
            raise MonitoringQueryRejected(
                "PROMQL_WINDOW_INVALID", "PromQL 查询时间窗超过策略"
            )
        step = max(
            self.snapshot.min_step_seconds,
            max(1, window // self.snapshot.max_points),
        )
        normalized = str(expression)
        normalized = normalized.replace(
            _EXTERNAL_SENTINEL, "${external_target}"
        ).replace(_HOST_SENTINEL, "${host_target}")
        return ValidatedPromQuery(
            normalized_query=normalized,
            query_sha256=self._sha256(normalized),
            policy_sha256=self._policy_hash(),
            target_scopes=tuple(sorted(scopes)),
            metric_names=tuple(sorted(metrics)),
            window_seconds=window,
            step_seconds=step,
            max_series=self.snapshot.max_series,
        )

    def _policy_hash(self) -> str:
        return self._sha256(
            json.dumps(
                self.snapshot.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    @staticmethod
    def _sha256(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


class LogQueryPolicySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "LOG_QUERY_POLICY.v1"
    max_query_chars: int = Field(default=2000, ge=1, le=10_000)
    max_filters: int = Field(default=4, ge=0, le=16)
    max_filter_chars: int = Field(default=256, ge=1, le=2000)
    max_window_seconds: int = Field(default=3600, ge=60, le=86_400)
    max_entries: int = Field(default=200, ge=1, le=5000)


class ValidatedLogQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "VALIDATED_LOG_QUERY.v1"
    normalized_query: str
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    filters: tuple[tuple[str, str], ...]
    window_seconds: int
    max_entries: int


class LogQueryPolicy:
    """只接受冻结 Binding selector 加字面量包含或排除过滤。"""

    def __init__(self, snapshot: LogQueryPolicySnapshot) -> None:
        self.snapshot = snapshot

    def validate(
        self, query: str, *, window_seconds: int | None = None
    ) -> ValidatedLogQuery:
        if not query.strip() or len(query) > self.snapshot.max_query_chars:
            raise MonitoringQueryRejected(
                "LOGQL_LENGTH_INVALID", "LogQL 为空或超过长度限制"
            )
        text = query.strip()
        if not text.startswith("${binding_selector}"):
            raise MonitoringQueryRejected(
                "LOGQL_BINDING_SELECTOR_REQUIRED",
                "LogQL 必须以冻结的 Binding selector 开始",
            )
        remaining = text[len("${binding_selector}") :]
        filters: list[tuple[str, str]] = []
        while remaining.strip():
            match = _LOG_FILTER.match(remaining)
            if match is None:
                raise MonitoringQueryRejected(
                    "LOGQL_PIPELINE_FORBIDDEN",
                    "LogQL 只允许字面量包含或排除过滤",
                )
            operator, escaped = match.groups()
            try:
                value = json.loads(f'"{escaped}"')
            except json.JSONDecodeError as exc:
                raise MonitoringQueryRejected(
                    "LOGQL_FILTER_INVALID", "LogQL 字面量格式无效"
                ) from exc
            if not value or len(value) > self.snapshot.max_filter_chars:
                raise MonitoringQueryRejected(
                    "LOGQL_FILTER_INVALID", "LogQL 过滤文本为空或过长"
                )
            if any(ord(character) < 32 for character in value):
                raise MonitoringQueryRejected(
                    "LOGQL_FILTER_INVALID", "LogQL 过滤文本包含控制字符"
                )
            filters.append((operator, value))
            remaining = remaining[match.end() :]
        if len(filters) > self.snapshot.max_filters:
            raise MonitoringQueryRejected(
                "LOGQL_FILTER_COUNT_INVALID", "LogQL 过滤数量超过策略"
            )
        window = window_seconds or self.snapshot.max_window_seconds
        if not 60 <= window <= self.snapshot.max_window_seconds:
            raise MonitoringQueryRejected(
                "LOGQL_WINDOW_INVALID", "LogQL 查询时间窗超过策略"
            )
        normalized = "${binding_selector}" + "".join(
            f" {operator} {json.dumps(value, ensure_ascii=False)}"
            for operator, value in filters
        )
        policy_hash = hashlib.sha256(
            json.dumps(
                self.snapshot.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return ValidatedLogQuery(
            normalized_query=normalized,
            query_sha256=hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest(),
            policy_sha256=policy_hash,
            filters=tuple(filters),
            window_seconds=window,
            max_entries=self.snapshot.max_entries,
        )
