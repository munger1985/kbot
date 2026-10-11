"""按数据库类型与版本精确解析的统一监控 Profile 目录。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aiops_agent.contracts.evidence import MetricDefinition
from platform_core.contracts.aiops.monitoring import MonitoringProfileSummary
from aiops_agent.monitoring.query_policy import (
    PromQueryPolicy,
    PromQueryPolicySnapshot,
)


DatabaseType = Literal["ORACLE", "MYSQL", "POSTGRESQL"]


class MonitoringProfileVariant(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    variant_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    db_type: DatabaseType
    version_codes: tuple[str, ...] = Field(min_length=1)
    metric_codes: tuple[str, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_uniqueness(self):
        if len(set(self.version_codes)) != len(self.version_codes):
            raise ValueError("Profile Variant 数据库版本不能重复")
        if len(set(self.metric_codes)) != len(self.metric_codes):
            raise ValueError("Profile Variant 指标不能重复")
        return self


class MonitoringProfileDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    version: str = Field(min_length=1, max_length=32)
    purpose: Literal["OVERVIEW", "REPLICATION", "CAPACITY", "HOST"]
    display_name: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=1000)
    variants: tuple[MonitoringProfileVariant, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_variants(self):
        variant_ids = [item.variant_id for item in self.variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("Profile Variant ID 不能重复")
        selectors = [
            (item.db_type, version_code)
            for item in self.variants
            for version_code in item.version_codes
        ]
        if len(selectors) != len(set(selectors)):
            raise ValueError("Profile 的数据库类型与版本必须唯一匹配 Variant")
        return self


class ResolvedMonitoringProfile(BaseModel):
    """一次请求中已经按 Target 类型与版本收敛的 Profile。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    version: str
    display_name: str
    description: str
    supported_db_types: tuple[DatabaseType, ...]
    supported_database_versions: dict[str, tuple[str, ...]]
    resolved_variant_ids: tuple[str, ...]
    metric_codes: tuple[str, ...]

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
            for variant in profile.variants:
                for code in variant.metric_codes:
                    definition = metric_catalog.get(code)
                    if variant.db_type not in definition.supported_db_types:
                        raise ValueError(
                            f"Profile {profile.profile_id} 的指标 {code} "
                            f"不支持 {variant.db_type}"
                        )
                    prometheus = definition.providers.get("PROMETHEUS")
                    zabbix = definition.providers.get("ZABBIX")
                    if prometheus is None or zabbix is None:
                        raise ValueError(
                            f"指标 {code} 尚未完成 Prometheus/Zabbix 对齐"
                        )
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

    def resolve(
        self,
        profile_id: str,
        databases: tuple[tuple[str, str], ...],
    ) -> ResolvedMonitoringProfile:
        if not databases:
            raise LookupError("Monitoring Profile 缺少数据库类型与版本")
        profile = self.get(profile_id)
        selected: list[MonitoringProfileVariant] = []
        versions: dict[str, set[str]] = {}
        for raw_db_type, version_code in databases:
            db_type = str(raw_db_type).upper()
            matches = [
                variant
                for variant in profile.variants
                if variant.db_type == db_type
                and version_code in variant.version_codes
            ]
            if len(matches) != 1:
                raise LookupError(
                    f"Profile {profile_id} 不支持数据库 {db_type} {version_code}"
                )
            selected.append(matches[0])
            versions.setdefault(db_type, set()).add(version_code)
        metric_sets = {variant.metric_codes for variant in selected}
        if len(metric_sets) != 1:
            raise LookupError("所选实例的 Profile Variant 指标口径不一致")
        return ResolvedMonitoringProfile(
            profile_id=profile.profile_id,
            version=profile.version,
            display_name=profile.display_name,
            description=profile.description,
            supported_db_types=tuple(sorted(versions)),
            supported_database_versions={
                db_type: tuple(sorted(items))
                for db_type, items in sorted(versions.items())
            },
            resolved_variant_ids=tuple(
                sorted({variant.variant_id for variant in selected})
            ),
            metric_codes=selected[0].metric_codes,
        )

    def overview_for_database(
        self, db_type: str, version_code: str
    ) -> ResolvedMonitoringProfile:
        matches = []
        for profile in self._profiles.values():
            if profile.purpose != "OVERVIEW":
                continue
            try:
                matches.append(
                    self.resolve(
                        profile.profile_id,
                        ((db_type, version_code),),
                    )
                )
            except LookupError:
                continue
        if len(matches) != 1:
            raise LookupError(
                f"数据库 {db_type} {version_code} 没有唯一概览 Monitoring Profile"
            )
        return matches[0]

    def list_for_databases(
        self, databases: tuple[tuple[str, str], ...]
    ) -> tuple[MonitoringProfileSummary, ...]:
        resolved = []
        for profile_id in self._profiles:
            try:
                profile = self.resolve(profile_id, databases)
            except LookupError:
                continue
            resolved.append(profile.summary())
        return tuple(resolved)


def load_monitoring_profile_catalog(
    metric_catalog, path: Path | None = None
) -> MonitoringProfileCatalog:
    resolved = path or (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "monitoring"
        / "monitoring_profiles.v1.json"
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
