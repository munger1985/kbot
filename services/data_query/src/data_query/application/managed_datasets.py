"""系统托管数据集的确定性调和。"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from data_query.entities import (
    DataSourceEntity,
    PolicyBindingEntity,
    SchemaSnapshotEntity,
    SchemaSnapshotObjectEntity,
    SemanticModelEntity,
    SemanticModelVersionEntity,
)
from platform_core.identity import uuid7
from data_query.contracts import SemanticModelDefinition


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def km_asset_definition(*, schema_name: str) -> dict[str, Any]:
    dataset = "assets"
    dimensions = [
        {"name": "asset_id", "display_name": "Asset ID", "physical_column": "ASSET_ID"},
        {"name": "title", "display_name": "Asset 标题", "physical_column": "ASSET_TITLE", "synonyms": ["asset name", "标题"]},
        {
            "name": "bundle_id",
            "display_name": "Knowledge Bundle ID",
            "physical_column": "KC_BUNDLE_ID",
            "synonyms": ["knowledge bundle id"],
        },
        {
            "name": "bundle_revision_id",
            "display_name": "Knowledge Bundle Revision ID",
            "physical_column": "KC_BUNDLE_REVISION_ID",
            "synonyms": ["knowledge bundle revision id"],
        },
        {
            "name": "author",
            "display_name": "作者邮箱或邮箱用户名",
            "physical_column": "AUTHOR_MAIL_NORM",
            "filter_alias_columns": ["AUTHOR_LOCAL_PART"],
            "value_normalization": "LOWER_TRIM",
            "allowed_filter_operators": ["EQ", "IN"],
            "synonyms": ["author email", "email address", "creator", "contributor", "作者", "作者邮箱", "创建者"],
        },
        {"name": "product", "display_name": "产品", "physical_column": "ASSET_PRODUCT"},
        {"name": "solution", "display_name": "解决方案或主题", "physical_column": "ASSET_SOLUTION"},
        {"name": "industry", "display_name": "行业", "physical_column": "INDUSTRY_ID"},
        {"name": "category", "display_name": "内容分类", "physical_column": "CONTENT_CATEGORY"},
        {
            "name": "asset_date",
            "display_name": "Asset 日期（优先发布日期，缺失时使用创建日期）",
            "physical_column": "ASSET_DATE_VALUE",
            "value_type": "DATE",
            "allowed_filter_operators": ["EQ", "BETWEEN", "GT", "GTE", "LT", "LTE"],
            "synonyms": ["created date", "creation date", "publish date", "published date", "创建日期", "发布日期"],
        },
    ]
    return {
        "datasets": [{"name": dataset, "display_name": "KM Asset", "physical_schema": schema_name, "physical_object": "KBOT_V_KM_ASSET_SEARCHABLE", "primary_time_dimension": None, "scope_column": "DOMAIN_ID"}],
        "dimensions": [{
            "dataset": dataset,
            "value_type": "STRING",
            "groupable": True,
            "filterable": True,
            "sensitivity": "INTERNAL",
            "synonyms": [],
            **item,
        } for item in dimensions],
        "measures": [
            {"name": "asset_count", "display_name": "Asset 数量", "dataset": dataset, "physical_column": None, "aggregation": "COUNT", "value_type": "INTEGER", "sensitivity": "INTERNAL"},
            {"name": "author_count", "display_name": "作者数量", "dataset": dataset, "physical_column": "AUTHOR_MAIL_NORM", "aggregation": "COUNT_DISTINCT", "value_type": "INTEGER", "sensitivity": "INTERNAL"},
        ],
    }


class ManagedDatasetError(ValueError):
    pass


class ManagedDatasetService:
    SOURCE_NAME = "__managed__:km_asset"
    MODEL_NAME = "KM Asset 元数据（系统托管）"

    def __init__(self, *, uow_factory, credential_service, database_config):
        self._uow_factory = uow_factory
        self._credentials = credential_service
        self._database = database_config

    async def reconcile_km_asset(self, *, domain_id: int, actor_id: str) -> dict[str, Any]:
        schema_name = self._database.oracle.username.upper()
        definition = SemanticModelDefinition.model_validate(
            km_asset_definition(schema_name=schema_name)
        ).model_dump(mode="json")
        catalog_hash = _hash(definition)
        columns = ["DOMAIN_ID", "SOURCE_ID", "KM_ASSET_ID", "ASSET_ID", "SOURCE_REVISION", "SOURCE_STATUS", "ASSET_TITLE", "AUTHOR_MAIL", "AUTHOR_MAIL_NORM", "AUTHOR_LOCAL_PART", "ASSET_PRODUCT", "ASSET_SOLUTION", "INDUSTRY_ID", "CONTENT_CATEGORY", "ASSET_STATUS", "PUBLISH_DATE", "ASSET_DATE_VALUE", "LAST_UPDATE_TIME", "KC_BUNDLE_ID", "KC_BUNDLE_REVISION_ID", "RAW_METADATA_JSON", "NORMALIZED_METADATA_JSON", "SYNCED_AT", "COMPLETED_AT"]
        async with self._uow_factory() as uow:
            source = await uow.data_sources.find_by_name(domain_id=domain_id, display_name=self.SOURCE_NAME, lock=True)
            if source is None:
                source_id = uuid7()
                credential = await self._credentials.create(uow=uow, domain_id=domain_id, data_source_id=source_id, username=self._database.oracle.username, password=self._database.oracle.require_password(), actor_id=actor_id)
                configuration = {"host": self._database.oracle.host, "port": self._database.oracle.port, "database": self._database.oracle.service_name, "allowed_schemas": [schema_name], "tls_enabled": False}
                source = DataSourceEntity(data_source_id=source_id, domain_id=domain_id, display_name=self.SOURCE_NAME, source_type="ORACLE", status="ACTIVE", current_version=1, configuration_json=configuration, configuration_hash=_hash(configuration), credential_id=credential.credential_id, capabilities_json={"managed": True, "snapshot": "CODE_OWNED"}, created_by=actor_id, updated_by=actor_id)
                await uow.data_sources.add(source)
            model = await uow.semantic_models.find_by_name(domain_id=domain_id, display_name=self.MODEL_NAME, lock=True)
            active = None if model is None else await uow.semantic_model_versions.get_active(semantic_model_id=model.semantic_model_id, lock=True)
            if model is None or active is None or active.definition_hash != catalog_hash:
                snapshot = SchemaSnapshotEntity(data_source_id=source.data_source_id, source_version=source.current_version, status="READY", snapshot_hash=catalog_hash, connector_type="ORACLE", connector_version="managed-v1", capabilities_json={"managed": True}, objects_json={"objects": [{"schema": schema_name, "name": "KBOT_V_KM_ASSET_SEARCHABLE", "type": "VIEW", "columns": columns}]}, requested_by=actor_id, completed_at=datetime.now(timezone.utc))
                await uow.schema_snapshots.add(snapshot)
                await uow.schema_snapshot_objects.add(SchemaSnapshotObjectEntity(schema_snapshot_id=snapshot.schema_snapshot_id, schema_name=schema_name, object_name="KBOT_V_KM_ASSET_SEARCHABLE", object_type="VIEW", selected=1, status="READY", attempt_count=1, metadata_source="MANUAL", metadata_json={"columns": columns}, completed_at=datetime.now(timezone.utc)))
                if model is None:
                    model = SemanticModelEntity(domain_id=domain_id, display_name=self.MODEL_NAME, description="由 KBot 管理的 KM Asset 固定问数模型", active_version=None, created_by=actor_id, updated_by=actor_id)
                    await uow.semantic_models.add(model)
                version_no = await uow.semantic_model_versions.next_version_no(semantic_model_id=model.semantic_model_id)
                if active is not None:
                    active.status = "RETIRED"
                version = SemanticModelVersionEntity(semantic_model_id=model.semantic_model_id, version_no=version_no, data_source_id=source.data_source_id, schema_snapshot_id=snapshot.schema_snapshot_id, status="ACTIVE", definition_json=definition, definition_hash=catalog_hash, published_by=actor_id, published_at=datetime.now(timezone.utc))
                await uow.semantic_model_versions.add(version)
                model.active_version = version_no
                model.updated_by = actor_id
                active = version
            policies = await uow.policy_bindings.list_by_domain(domain_id=domain_id, after_id=None, limit=10_000)
            policy = next((row for row in policies if row.policy_json.get("managed_consumer_app_id") == "km_asset" and str(model.semantic_model_id) in row.semantic_model_ids_json), None)
            if policy is None:
                subjects = {"actor_ids": [], "roles": ["__managed__:km_asset"]}
                policy_json = {"managed_consumer_app_id": "km_asset", "budget": {"max_rows": 1000, "max_result_bytes": 1048576, "statement_timeout_seconds": 30, "max_concurrent_runs": 4}}
                policy = PolicyBindingEntity(domain_id=domain_id, subject_selector_json=subjects, semantic_model_ids_json=[str(model.semantic_model_id)], policy_json=policy_json, policy_hash=_hash({"subjects": subjects, "models": [str(model.semantic_model_id)], "policy": policy_json}), status="ACTIVE", created_by=actor_id, updated_by=actor_id)
                await uow.policy_bindings.add(policy)
            await uow.commit()
            return {"consumer_app_id": "km_asset", "catalog_hash": catalog_hash, "data_source_id": str(source.data_source_id), "schema_snapshot_id": str(active.schema_snapshot_id), "semantic_model_id": str(model.semantic_model_id), "semantic_model_version_id": str(active.semantic_model_version_id), "semantic_model_version": int(active.version_no), "policy_binding_id": str(policy.policy_binding_id), "status": "READY"}
