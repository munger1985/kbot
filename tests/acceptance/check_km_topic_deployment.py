"""检查 KM 主题问数代码是否从当前 Python 环境正确加载。"""

from __future__ import annotations

import inspect

import platform_core
from agent_runtime.specialists.data_query import SemanticDataQueryExecutor
from agent_runtime.specialists.data_query.contracts import KMTopicExpansion
from agent_runtime.specialists.response_composer import ResponseComposerSkill
from agent_runtime.specialists.root import (
    KMAnswerBasis,
    RootAgentPlanner,
    RouteDecision,
    RouteType,
)
from data_query.application.managed_datasets import km_asset_definition
from platform_core.contracts.data_query import PlanFilter
from platform_core.prompts import load_prompt_catalog
from platform_core.prompts.catalog import DEFAULT_PROMPT_CATALOG


EXPECTED_PROMPTS = {
    "agent_runtime.km_asset_intent_route": "1.0.0",
    "agent_runtime.km_asset_context_route": "1.0.0",
    "agent_runtime.data_query_plan": "1.0.0",
    "agent_runtime.km_asset_enumeration_compose": "1.0.0",
    "agent_runtime.km_topic_english_expand": "1.0.0",
}


def main() -> int:
    """打印实际加载位置，并校验 Prompt 与托管模型代码。"""
    catalog = load_prompt_catalog()
    active_versions = {
        item.prompt_key: item.version
        for item in catalog.entries
        if item.active and item.prompt_key in EXPECTED_PROMPTS
    }
    definition = km_asset_definition(schema_name="CHECK_ONLY")
    dimension_names = {
        str(item.get("name"))
        for item in definition.get("dimensions", ())
        if isinstance(item, dict)
    }
    has_topic = "topic" in dimension_names
    has_enumeration_scope = {
        "asset_id", "title", "bundle_id", "bundle_revision_id"
    }.issubset(dimension_names)
    plan = RootAgentPlanner().build_plan(
        objective="列出关于 OAC 的 asset",
        decision=RouteDecision(
            route_type=RouteType.HYBRID_DATA_FIRST,
            confidence=1,
            reason="部署检查",
            answer_basis=KMAnswerBasis.SEMANTIC_RELEVANCE_ENUMERATION,
        ),
    )
    compose_task = next(
        item for item in plan.tasks if item.task_key == "response_compose"
    )
    has_scope_in_compose_plan = (
        "document_scope" in compose_task.depends_on
        and "task_output:document_scope" in compose_task.input_refs
    )
    has_multilingual_topic_search = all(
        hasattr(SemanticDataQueryExecutor, name)
        for name in (
            "_km_topic_terms",
            "_execute_km_asset_multilingual_enumeration",
            "_execute_km_asset_multilingual_count",
        )
    )
    english_topics_field = KMTopicExpansion.model_fields.get(
        "english_topics"
    )
    has_english_keyword_group = (
        english_topics_field is not None
        and any(
            getattr(item, "min_length", None) == 2
            for item in english_topics_field.metadata
        )
        and any(
            getattr(item, "max_length", None) == 3
            for item in english_topics_field.metadata
        )
    )
    try:
        has_contains_any = len(PlanFilter(
            field="topic",
            operator="CONTAINS",
            values=("finance", "financial"),
        ).values) == 2
    except ValueError:
        has_contains_any = False
    normalization_models = [{
        "semantic_model_id": "00000000-0000-0000-0000-000000000001",
        "semantic_model_version": 1,
        "datasets": [{"name": "assets"}],
        "dimensions": [
            {"name": name}
            for name in (
                "asset_id", "title", "bundle_id", "bundle_revision_id",
                "product", "solution", "asset_date", "topic",
            )
        ],
        "measures": [{
            "name": "asset_count",
            "aggregation": "COUNT",
        }],
        "max_rows": 1000,
    }]
    normalized_enumeration = (
        SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "semantic_model_id": (
                    "00000000-0000-0000-0000-000000000001"
                ),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [{
                    "name": "asset_count",
                    "aggregation": "COUNT",
                }],
                "dimensions": ["title"],
                "filters": [{
                    "field": "topic",
                    "operator": "CONTAINS",
                    "values": ["APEX"],
                }],
                "order_by": [{
                    "field": "asset_date",
                    "direction": "DESC",
                }],
                "limit": 3,
            },
            models=normalization_models,
            question="list 3 latest assets related to apex",
            consumer_app_id="km_asset",
            answer_basis="SEMANTIC_RELEVANCE_ENUMERATION",
        )
    )
    has_flexible_enumeration_limit = (
        normalized_enumeration.get("limit") == 3
        and normalized_enumeration.get("order_by") == [{
            "field": "asset_date",
            "direction": "DESC",
        }]
    )
    try:
        ResponseComposerSkill._validate_enumeration_body(
            "[{'asset_id': 'ASSET-1', 'title': 'APEX Asset'}]",
            assets=[{"asset_id": "ASSET-1", "title": "APEX Asset"}],
            allowed={},
            language="en-US",
        )
        has_serialized_row_guard = False
    except ValueError:
        has_serialized_row_guard = True

    print(f"platform_core = {platform_core.__file__}")
    print(f"prompt_catalog = {DEFAULT_PROMPT_CATALOG}")
    print(
        "managed_datasets = "
        f"{inspect.getsourcefile(km_asset_definition)}"
    )
    for prompt_key, expected_version in EXPECTED_PROMPTS.items():
        actual_version = active_versions.get(prompt_key, "MISSING")
        print(
            f"{prompt_key} = {actual_version} "
            f"(expected {expected_version})"
        )
    print(f"HAS_TOPIC_IN_CODE = {has_topic}")
    print(f"HAS_ENUMERATION_SCOPE_IN_CODE = {has_enumeration_scope}")
    print(f"HAS_SCOPE_IN_COMPOSE_PLAN = {has_scope_in_compose_plan}")
    print(
        "HAS_MULTILINGUAL_TOPIC_SEARCH = "
        f"{has_multilingual_topic_search}"
    )
    print(f"HAS_ENGLISH_KEYWORD_GROUP = {has_english_keyword_group}")
    print(f"HAS_CONTAINS_ANY = {has_contains_any}")
    print(
        "HAS_FLEXIBLE_ENUMERATION_LIMIT = "
        f"{has_flexible_enumeration_limit}"
    )
    print(f"HAS_SERIALIZED_ROW_GUARD = {has_serialized_row_guard}")

    prompts_ok = all(
        active_versions.get(prompt_key) == expected_version
        for prompt_key, expected_version in EXPECTED_PROMPTS.items()
    )
    if (
        prompts_ok
        and has_topic
        and has_enumeration_scope
        and has_scope_in_compose_plan
        and has_multilingual_topic_search
        and has_english_keyword_group
        and has_contains_any
        and has_flexible_enumeration_limit
        and has_serialized_row_guard
    ):
        print("KM 主题问数代码加载检查通过")
        return 0
    print("KM 主题问数代码加载检查失败：当前环境仍在加载旧代码")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
