"""版本化实时监控 Profile 目录。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aiops_agent.contracts.evidence import MetricDefinition
from platform_core.contracts.aiops.monitoring import (
    GrafanaDashboardUid,
    MonitoringProfileSummary,
)
from aiops_agent.monitoring.query_policy import (
    PromQueryPolicy,
    PromQueryPolicySnapshot,
)


class MonitoringProfileDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    version: str = Field(min_length=1, max_length=32)
    display_name: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=1000)
    supported_db_types: tuple[
        Literal["ORACLE", "MYSQL", "POSTGRESQL"], ...
    ]
    metric_codes: tuple[str, ...] = Field(min_length=1, max_length=8)
    grafana_dashboard_uid: GrafanaDashboardUid | None = None

    @model_validator(mode="after")
    def validate_uniqueness(self):
        if len(set(self.supported_db_types)) != len(self.supported_db_types):
            raise ValueError("Profile 数据库类型不能重复")
        if len(set(self.metric_codes)) != len(self.metric_codes):
            raise ValueError("Profile 指标不能重复")
        return self

    def summary(self) -> MonitoringProfileSummary:
        return MonitoringProfileSummary.model_validate(
            self.model_dump(mode="python")
        )


class MonitoringProfileDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    catalog_version: str = Field(min_length=1, max_length=64)
    profiles: tuple[MonitoringProfileDefinition, ...]

    @model_validator(mode="after")
    def validate_unique_ids(self):
        ids = [item.profile_id for item in self.profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("Monitoring Profile ID 不能重复")
        return self


class MonitoringProfileCatalog:
    def __init__(self, document: MonitoringProfileDocument, metric_catalog):
        self.version = document.catalog_version
        self._profiles = {
            item.profile_id: item for item in document.profiles
        }
        self._validate_metrics(metric_catalog)
        canonical = json.dumps(
            document.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.manifest_hash = hashlib.sha256(canonical).hexdigest()

    def _validate_metrics(self, metric_catalog) -> None:
        for profile in self._profiles.values():
            for code in profile.metric_codes:
                definition = metric_catalog.get(code)
                if not set(profile.supported_db_types).intersection(
                    definition.supported_db_types
                ):
                    raise ValueError(
                        f"Profile {profile.profile_id} 的指标 {code} 不支持声明的数据库"
                    )
                prometheus = definition.providers.get("PROMETHEUS")
                zabbix = definition.providers.get("ZABBIX")
                if prometheus is None or zabbix is None:
                    raise ValueError(f"指标 {code} 尚未完成 Prometheus/Zabbix 对齐")
                if not (
                    zabbix.exact_item_key
                    and zabbix.value_type
                    and zabbix.unit == definition.unit
                ):
                    raise ValueError(f"指标 {code} 的 Zabbix 定义不完整")

    def get(self, profile_id: str) -> MonitoringProfileDefinition:
        try:
            return self._profiles[profile_id]
        except KeyError as exc:
            raise KeyError(f"未知 Monitoring Profile：{profile_id}") from exc

    def list_for_db_types(
        self, db_types: tuple[str, ...]
    ) -> tuple[MonitoringProfileSummary, ...]:
        requested = {item.upper() for item in db_types}
        return tuple(
            profile.summary()
            for profile in sorted(
                self._profiles.values(), key=lambda item: item.profile_id
            )
            if requested
            and requested.issubset(set(profile.supported_db_types))
        )


def load_monitoring_profile_catalog(
    metric_catalog, path: Path | None = None
) -> MonitoringProfileCatalog:
    resolved = path or (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "monitoring"
        / "dashboard_profiles.v1.json"
    )
    document = MonitoringProfileDocument.model_validate_json(
        resolved.read_text(encoding="utf-8")
    )
    return MonitoringProfileCatalog(document, metric_catalog)


def resolve_metric_definitions(snapshot: dict) -> tuple[MetricDefinition, ...]:
    """应用 Binding 中受控的 Prometheus 与 Zabbix 精确映射覆盖。"""
    overrides = dict(snapshot.get("mapping_overrides") or {})
    prometheus_queries = overrides.get("prometheus_queries") or {}
    zabbix_item_keys = overrides.get("zabbix_item_keys") or {}
    if not isinstance(prometheus_queries, dict):
        raise ValueError("prometheus_queries 必须是对象")
    if not isinstance(zabbix_item_keys, dict):
        raise ValueError("zabbix_item_keys 必须是对象")
    definitions = []
    for item in snapshot["metrics"]:
        definition = MetricDefinition.model_validate(item)
        query = prometheus_queries.get(definition.metric_code)
        if query is not None:
            if (
                not isinstance(query, str)
                or not query.strip()
                or len(query) > 2000
                or "${" in query.replace("${external_target}", "").replace(
                    "${host_target}", ""
                )
            ):
                raise ValueError("Prometheus 指标查询覆盖格式无效")
            provider = definition.providers.get("PROMETHEUS")
            if provider is None:
                raise ValueError("指标不支持 Prometheus 查询覆盖")
            definition = definition.model_copy(
                update={
                    "providers": {
                        **definition.providers,
                        "PROMETHEUS": provider.model_copy(
                            update={
                                "template_id": f"binding.{definition.metric_code}",
                                "template_version": str(snapshot["binding_version"]),
                                "query_template": query.strip(),
                                "fallback_query_templates": (
                                    provider.fallback_query_templates
                                    if query.strip() == provider.query_template
                                    else ()
                                ),
                            }
                        ),
                    }
                }
            )
        item_key = zabbix_item_keys.get(definition.metric_code)
        if item_key is not None:
            if (
                not isinstance(item_key, str)
                or not item_key.strip()
                or len(item_key) > 512
            ):
                raise ValueError("Zabbix Item Key 覆盖格式无效")
            provider = definition.providers.get("ZABBIX")
            if provider is None:
                raise ValueError("指标不支持 Zabbix Item Key 覆盖")
            definition = definition.model_copy(
                update={
                    "providers": {
                        **definition.providers,
                        "ZABBIX": provider.model_copy(
                            update={
                                "template_id": f"binding.{definition.metric_code}",
                                "template_version": str(snapshot["binding_version"]),
                                "exact_item_key": item_key.strip(),
                            }
                        ),
                    }
                }
            )
        provider = definition.providers.get("PROMETHEUS")
        if provider is not None and provider.query_template is not None:
            policy = PromQueryPolicy(PromQueryPolicySnapshot())
            window = min(
                definition.default_window_seconds,
                PromQueryPolicySnapshot().max_window_seconds,
            )
            checked = policy.validate(
                provider.query_template, window_seconds=window
            )
            definition = definition.model_copy(
                update={
                    "providers": {
                        **definition.providers,
                        "PROMETHEUS": provider.model_copy(
                            update={
                                "query_template": checked.normalized_query,
                                "fallback_query_templates": tuple(
                                    policy.validate(
                                        fallback, window_seconds=window
                                    ).normalized_query
                                    for fallback in provider.fallback_query_templates
                                ),
                            }
                        ),
                    }
                }
            )
        definitions.append(definition)
    return tuple(definitions)
