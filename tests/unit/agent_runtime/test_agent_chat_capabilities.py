"""4.0 通用对话、问数、图表与 Dify Adapter 的聚焦测试。"""

import asyncio
import json
import unittest
from types import SimpleNamespace

from agent_runtime.application.commands import LeasedArtifact
from agent_runtime.runtime import ExecutionContext, SkillProgress, SkillResult
from agent_runtime.specialists.conversation_response import (
    ConversationResponseSkill,
)
from agent_runtime.specialists.data_query import (
    DataQuerySkill,
    MCPDataQueryExecutor,
    QueryResult,
)
from agent_runtime.specialists.hybrid import DocumentScopeExtractSkill
from agent_runtime.specialists.km_asset import (
    KmAssetAnswerBasis,
    KmAssetDocumentScopeExtractSkill,
    KmAssetRoutePlanner,
    KmAssetResponseComposerSkill as ResponseComposerSkill,
    KmAssetSemanticDataQueryExecutor as SemanticDataQueryExecutor,
)
from agent_runtime.specialists.visualization import EChartsSkill
from agent_runtime.specialists.root import (
    RootAgentPlanner,
    RouteType,
)
from main_api.api.dify import _records
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.contracts.data_query import DataQueryPlanV1
from platform_core.identity import uuid7


KMAnswerBasis = KmAssetAnswerBasis


def _root_planner(*, model_client=None, prompt_resolver=None):
    """测试 Root 时显式注册 KM Asset 专属路由器。"""
    return RootAgentPlanner(
        model_client=model_client,
        prompt_resolver=prompt_resolver,
        app_route_planners={
            "km_asset": KmAssetRoutePlanner(
                model_client=model_client,
                prompt_resolver=prompt_resolver,
                timeout_seconds=30,
            ),
        },
    )


class _PromptResolver:
    async def resolve(self, key):
        return SimpleNamespace(
            content=f"prompt:{key}",
            version="1.0.0",
            ref=lambda: {"prompt_key": key, "version": "1.0.0"},
        )


class _ModelClient:
    def __init__(self, *, response=None, responses=(), chunks=()):
        self.response = response or {}
        self.responses = list(responses)
        self.chunks = chunks
        self.last_json_request = None
        self.json_requests = []

    async def get_llm_json(self, **kwargs):
        self.last_json_request = kwargs
        self.json_requests.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        return self.response

    async def stream_llm_chunks(self, **kwargs):
        for content in self.chunks:
            yield SimpleNamespace(content=content, reasoning_content=None)


class _DataClient:
    def __init__(self):
        self.request = None

    async def query(self, **kwargs):
        self.request = kwargs
        return {
            "rows": [{"month": "2026-07", "sales": 10}],
            "upstream_row_count": 1,
            "truncated": False,
        }


class _UnavailableSemanticExecutor:
    async def execute(self, **kwargs):
        raise AssertionError("MCP 模式不应调用语义 Provider")


def _artifact(artifact_type, payload):
    return LeasedArtifact(
        artifact_id=uuid7(),
        task_id=uuid7(),
        artifact_type=artifact_type,
        schema_version=f"{artifact_type}.v1",
        producer="test",
        producer_version="1.0.0",
        payload=payload,
        content_hash="a" * 64,
        security_level=0,
    )


def _context(
    *,
    artifacts=(),
    agent=None,
    route=None,
    original_input="你好",
    policy_snapshot=None,
    language=None,
):
    config_snapshot = {"agent": agent or {}, "route": route}
    if language is not None:
        config_snapshot["language"] = language
    return ExecutionContext(
        domain_id=20,
        agent_id=uuid7(),
        run_id=uuid7(),
        task_id=uuid7(),
        task_key="test",
        actor_id="user-1",
        request_id="request-1",
        trace_id="trace-1",
        original_input=original_input,
        policy_snapshot=policy_snapshot or {},
        config_snapshot=config_snapshot,
        input_artifacts=tuple(artifacts),
    )


def _asset_plan_response(
    *, operation="LIST", semantic=False, ambiguities=(), author=None
):
    criteria = []
    expression = None
    if semantic:
        criteria = [{
            "criterion_id": "c1",
            "kind": "SEMANTIC_CONCEPT",
            "field_scope": ["CONTENT"],
            "operator": "RELATED_TO",
            "values": ["OAC"],
            "evidence_requirement": "CONTENT",
        }]
        expression = {"node_type": "REF", "criterion_id": "c1"}
    elif author:
        criteria = [{
            "criterion_id": "c1",
            "kind": "METADATA",
            "field_scope": ["author"],
            "operator": "EQ",
            "values": [author],
            "evidence_requirement": "QUERY_RESULT",
        }]
        expression = {"node_type": "REF", "criterion_id": "c1"}
    is_list = operation == "LIST"
    return {
        "operation": operation,
        "target": "ASSET",
        "criteria": criteria,
        "eligibility_expression": expression,
        "measures": (
            [] if is_list else [{"name": "asset_count", "aggregation": "COUNT"}]
        ),
        "display_limit": 10 if is_list else None,
        "result_assets": {
            "mode": "PRIMARY" if is_list else "SUPPORTING",
            "target_count": 10 if is_list else 5,
            "selection": "REQUESTED_ORDER" if is_list else "RECENT_WITHIN_RESULT",
        },
        "ambiguities": list(ambiguities),
    }


class AgentChatCapabilitiesTest(unittest.IsolatedAsyncioTestCase):
    def test_semantic_plan_removes_only_known_input_echo_fields(self):
        model_id = uuid7()
        models = [{
            "semantic_model_id": str(model_id),
            "semantic_model_version": 1,
            "datasets": [{"name": "assets"}],
            "dimensions": [{"name": "author"}],
            "measures": [{
                "name": "asset_count",
                "aggregation": "COUNT",
            }],
            "max_rows": 1000,
        }]
        normalized = SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "contract_version": "DataQueryPlan.v1",
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [{"name": "asset_count"}],
                "dimensions": ["title"],
                "filters": [{
                    "field": "author",
                    "operator": "EQ",
                    "values": ["madhumitha.k@oracle.com"],
                }],
                "limit": 100,
                "question": "any assets of madhumitha.k@oracle.com；",
                "models": models,
                "document_constraints": None,
            },
            models=models,
            question="any assets of madhumitha.k@oracle.com；",
            consumer_app_id="km_asset",
        )

        plan = DataQueryPlanV1.model_validate(normalized)
        self.assertEqual(
            ("madhumitha.k@oracle.com",),
            plan.filters[0].values,
        )
        for field in ("question", "models", "document_constraints"):
            self.assertNotIn(field, normalized)

    def test_semantic_plan_keeps_unknown_fields_for_strict_validation(self):
        model_id = uuid7()
        models = [{
            "semantic_model_id": str(model_id),
            "semantic_model_version": 1,
            "datasets": [{"name": "assets"}],
            "measures": [{
                "name": "asset_count",
                "aggregation": "COUNT",
            }],
            "max_rows": 1000,
        }]
        normalized = SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [{"name": "asset_count"}],
                "unexpected_field": "must still fail",
            },
            models=models,
            question="any assets",
            consumer_app_id="km_asset",
        )

        with self.assertRaises(ValueError):
            DataQueryPlanV1.model_validate(normalized)

    def test_semantic_plan_normalizes_catalog_aggregation_and_limit(self):
        model_id = uuid7()
        normalized = SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "contract_version": "DataQueryPlan.v1",
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [{"name": "asset_count"}],
                "limit": "100",
            },
            models=[{
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "datasets": [{"name": "assets"}],
                "measures": [{
                    "name": "asset_count",
                    "aggregation": "COUNT",
                }],
                "max_rows": 1000,
            }],
            question="有多少个 Asset",
            consumer_app_id="km_asset",
        )

        self.assertEqual(
            [{"name": "asset_count", "aggregation": "COUNT"}],
            normalized["measures"],
        )
        self.assertEqual(100, normalized["limit"])

    def test_semantic_plan_projects_order_dimension_from_catalog(self):
        model_id = uuid7()
        normalized = SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "contract_version": "DataQueryPlan.v1",
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [{"name": "asset_count"}],
                "dimensions": ["asset_id", "title"],
                "order_by": [{
                    "field": "asset_date",
                    "direction": "DESC",
                }],
                "limit": 10,
            },
            models=[{
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "datasets": [{"name": "assets"}],
                "dimensions": [
                    {"name": "asset_id"},
                    {"name": "title"},
                    {"name": "asset_date"},
                ],
                "measures": [{
                    "name": "asset_count",
                    "aggregation": "COUNT",
                }],
                "max_rows": 1000,
            }],
            question="list all assets",
            consumer_app_id="km_asset",
        )

        plan = DataQueryPlanV1.model_validate(normalized)
        self.assertEqual(
            ("asset_id", "title", "asset_date"), plan.dimensions,
        )
        self.assertEqual("asset_date", plan.order_by[0].field)

    def test_exact_metadata_enumeration_preserves_asset_rows_and_order(self):
        model_id = uuid7()
        dimensions = [
            {"name": name}
            for name in (
                "asset_id", "title", "bundle_id", "bundle_revision_id",
                "product", "solution", "ingestion_status", "asset_date",
            )
        ]
        normalized = SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [{"name": "asset_count"}],
                "dimensions": ["product"],
                "order_by": [{"field": "product", "direction": "ASC"}],
                "limit": 100,
            },
            models=[{
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "datasets": [{"name": "assets"}],
                "dimensions": dimensions,
                "measures": [{
                    "name": "asset_count",
                    "aggregation": "COUNT",
                }],
                "max_rows": 1000,
            }],
            question="Show assets sorted by domain",
            consumer_app_id="km_asset",
            answer_basis="EXACT_METADATA_ENUMERATION",
        )

        plan = DataQueryPlanV1.model_validate(normalized)
        self.assertEqual(
            (
                "asset_id", "title", "bundle_id", "bundle_revision_id",
                "product", "solution", "ingestion_status", "asset_date",
            ),
            plan.dimensions,
        )
        self.assertEqual("product", plan.order_by[0].field)
        self.assertEqual(10, plan.limit)

    def test_km_semantic_plan_restores_empty_measure_from_managed_catalog(self):
        model_id = uuid7()
        normalized = SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "contract_version": "DataQueryPlan.v1",
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [],
                "limit": None,
            },
            models=[{
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "datasets": [{"name": "assets"}],
                "measures": [
                    {"name": "asset_count", "aggregation": "COUNT"},
                    {
                        "name": "author_count",
                        "aggregation": "COUNT_DISTINCT",
                    },
                ],
                "max_rows": 50,
            }],
            question="AI 相关的 Asset 有哪些",
            consumer_app_id="km_asset",
        )

        self.assertEqual(
            [{"name": "asset_count", "aggregation": "COUNT"}],
            normalized["measures"],
        )
        self.assertEqual(50, normalized["limit"])

    def test_km_semantic_plan_normalizes_catalog_filter_shape(self):
        model_id = uuid7()
        normalized = SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "contract_version": "DataQueryPlan.v1",
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [{"name": "asset_count"}],
                "dimensions": ["title"],
                "filters": [{
                    "dimension": "solution",
                    "operator": "contains",
                    "value": "chatbi",
                }],
                "limit": 100,
            },
            models=[{
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "datasets": [{"name": "assets"}],
                "dimensions": [
                    {"name": "title"},
                    {"name": "solution"},
                ],
                "measures": [{
                    "name": "asset_count",
                    "aggregation": "COUNT",
                }],
                "max_rows": 1000,
            }],
            question="查一下关于 chatbi 的 asset",
            consumer_app_id="km_asset",
        )

        plan = DataQueryPlanV1.model_validate(normalized)
        self.assertEqual("solution", plan.filters[0].field)
        self.assertEqual("CONTAINS", plan.filters[0].operator)
        self.assertEqual(("chatbi",), plan.filters[0].values)

    def test_km_enumeration_keeps_primary_topic_and_defers_preference(self):
        model_id = uuid7()
        dimensions = [
            {"name": name}
            for name in (
                "asset_id", "title", "bundle_id", "bundle_revision_id",
                "product", "solution", "asset_date", "topic",
            )
        ]
        dimensions[-1].update({
            "groupable": False,
            "allowed_filter_operators": ["CONTAINS"],
        })
        normalized = SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [{"name": "asset_count"}],
                "dimensions": ["title"],
                "filters": [
                    {
                        "field": "topic",
                        "operator": "CONTAINS",
                        "values": ["OAC"],
                    },
                    {
                        "field": "topic",
                        "operator": "CONTAINS",
                        "values": ["金融欺诈"],
                    },
                ],
                "order_by": [],
                "limit": 10,
            },
            models=[{
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "datasets": [{"name": "assets"}],
                "dimensions": dimensions,
                "measures": [{
                    "name": "asset_count",
                    "aggregation": "COUNT",
                }],
                "max_rows": 1000,
            }],
            question="有哪些asset是关于oac的，最好涉及金融欺诈领域",
            consumer_app_id="km_asset",
            answer_basis="SEMANTIC_RELEVANCE_ENUMERATION",
        )

        plan = DataQueryPlanV1.model_validate(normalized)
        topic_filters = [
            item for item in plan.filters if item.field == "topic"
        ]
        self.assertEqual(1, len(topic_filters))
        self.assertEqual(("OAC",), topic_filters[0].values)

    def test_km_plan_forces_ready_assets_even_when_model_selects_failed(self):
        model_id = uuid7()
        normalized = SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [{"name": "asset_count"}],
                "dimensions": ["title", "ingestion_status"],
                "filters": [{
                    "field": "ingestion_status",
                    "operator": "EQ",
                    "values": ["FAILED"],
                }],
                "limit": 10,
            },
            models=[{
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "datasets": [{"name": "assets"}],
                "dimensions": [
                    {"name": "title"},
                    {
                        "name": "ingestion_status",
                        "allowed_filter_operators": ["EQ"],
                    },
                ],
                "measures": [{
                    "name": "asset_count",
                    "aggregation": "COUNT",
                }],
                "max_rows": 1000,
            }],
            question="列出金融欺诈相关的 Asset",
            consumer_app_id="km_asset",
        )

        plan = DataQueryPlanV1.model_validate(normalized)
        status_filters = [
            item for item in plan.filters
            if item.field == "ingestion_status"
        ]
        self.assertEqual(1, len(status_filters))
        self.assertEqual("EQ", status_filters[0].operator)
        self.assertEqual(("READY",), status_filters[0].values)

    def test_km_semantic_plan_uses_catalog_author_operator(self):
        model_id = uuid7()
        normalized = SemanticDataQueryExecutor._normalize_plan_response(
            response={
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "dataset": "assets",
                "measures": [{"name": "asset_count"}],
                "dimensions": ["title", "author"],
                "filters": [{
                    "field": "author",
                    "operator": "CONTAINS",
                    "values": ["lavkesh.singh"],
                }],
                "limit": 100,
            },
            models=[{
                "semantic_model_id": str(model_id),
                "semantic_model_version": 1,
                "datasets": [{"name": "assets"}],
                "dimensions": [
                    {"name": "title"},
                    {
                        "name": "author",
                        "allowed_filter_operators": ["EQ", "IN"],
                    },
                ],
                "measures": [{
                    "name": "asset_count",
                    "aggregation": "COUNT",
                }],
                "max_rows": 1000,
            }],
            question="list all assets of lavkesh.singh",
            consumer_app_id="km_asset",
        )

        plan = DataQueryPlanV1.model_validate(normalized)
        self.assertEqual("EQ", plan.filters[0].operator)

    async def test_km_topic_aggregate_plan_repairs_single_field_filter(self):
        model_id = uuid7()
        base_response = {
            "semantic_model_id": str(model_id),
            "semantic_model_version": 1,
            "dataset": "assets",
            "dimensions": [],
            "measures": [{"name": "asset_count", "aggregation": "COUNT"}],
            "order_by": [],
            "limit": 100,
        }
        model_client = _ModelClient(responses=(
            {
                **base_response,
                "filters": [{
                    "field": "solution",
                    "operator": "CONTAINS",
                    "values": ["OAC"],
                }],
            },
            {
                **base_response,
                "filters": [{
                    "field": "topic",
                    "operator": "CONTAINS",
                    "values": ["OAC"],
                }],
            },
        ))
        executor = SemanticDataQueryExecutor(
            client=None,
            model_client=model_client,
            prompt_resolver=_PromptResolver(),
        )
        context = _context(
            original_input="how many asset about OAC",
            agent={
                "owner_app_id": "km_asset",
                "models": {
                    "data_planner_llm": {
                        "served_model_name": "planner-model"
                    }
                },
            },
            route={
                "route_type": "DATA_QUERY",
                "answer_basis": "SEMANTIC_RELEVANCE_AGGREGATE",
            },
        )
        models = [{
            "semantic_model_id": str(model_id),
            "semantic_model_version": 1,
            "datasets": [{"name": "assets"}],
            "dimensions": [
                {
                    "name": "topic",
                    "groupable": False,
                    "allowed_filter_operators": ["CONTAINS"],
                },
                {"name": "solution"},
            ],
            "measures": [
                {"name": "asset_count", "aggregation": "COUNT"}
            ],
            "max_rows": 1000,
        }]

        plan = await executor._create_plan(
            context=context,
            question="how many asset about OAC",
            models=models,
        )

        self.assertEqual("topic", plan.filters[0].field)
        self.assertEqual("CONTAINS", plan.filters[0].operator)
        self.assertEqual(2, len(model_client.json_requests))

    async def test_km_topic_enumeration_preserves_requested_limit_and_order(self):
        model_id = uuid7()
        model_client = _ModelClient(response={
            "semantic_model_id": str(model_id),
            "semantic_model_version": 2,
            "dataset": "assets",
            "measures": [{
                "name": "asset_count", "aggregation": "COUNT"
            }],
            "dimensions": ["title"],
            "filters": [{
                "field": "topic",
                "operator": "CONTAINS",
                "values": ["OAC"],
            }],
            "order_by": [{
                "field": "asset_date",
                "direction": "DESC",
            }],
            "limit": 3,
        })
        executor = SemanticDataQueryExecutor(
            client=None,
            model_client=model_client,
            prompt_resolver=_PromptResolver(),
        )
        context = _context(
            original_input="列出最新 3 个关于 OAC 的 asset",
            agent={
                "owner_app_id": "km_asset",
                "models": {
                    "data_planner_llm": {
                        "served_model_name": "planner-model"
                    }
                },
            },
            route={
                "route_type": "HYBRID_DATA_FIRST",
                "answer_basis": "SEMANTIC_RELEVANCE_ENUMERATION",
            },
        )
        dimensions = [
            {"name": name}
            for name in (
                "asset_id", "title", "bundle_id", "bundle_revision_id",
                "product", "solution", "asset_date", "topic",
            )
        ]
        dimensions[-1].update({
            "groupable": False,
            "allowed_filter_operators": ["CONTAINS"],
        })
        models = [{
            "semantic_model_id": str(model_id),
            "semantic_model_version": 2,
            "datasets": [{"name": "assets"}],
            "dimensions": dimensions,
            "measures": [{
                "name": "asset_count", "aggregation": "COUNT"
            }],
            "max_rows": 1000,
        }]

        plan = await executor._create_plan(
            context=context,
            question="列出最新 3 个关于 OAC 的 asset",
            models=models,
        )

        self.assertEqual(
            (
                "asset_id", "title", "bundle_id", "bundle_revision_id",
                "product", "solution", "asset_date",
            ),
            plan.dimensions,
        )
        self.assertEqual(3, plan.limit)
        self.assertEqual("topic", plan.filters[0].field)
        self.assertEqual("asset_date", plan.order_by[0].field)
        self.assertEqual("DESC", plan.order_by[0].direction)

    async def test_km_topic_english_input_keeps_single_search_branch(self):
        model_client = _ModelClient(response={})
        executor = SemanticDataQueryExecutor(
            client=None,
            model_client=model_client,
            prompt_resolver=_PromptResolver(),
        )
        plan = DataQueryPlanV1(
            semantic_model_id=uuid7(),
            semantic_model_version=1,
            dataset="assets",
            measures=({"name": "asset_count", "aggregation": "COUNT"},),
            filters=({
                "field": "topic",
                "operator": "CONTAINS",
                "values": ("financial",),
            },),
        )

        for question in (
            "list assets about financial",
            "列出关于financial的asset",
        ):
            with self.subTest(question=question):
                terms, warnings = await executor._km_topic_terms(
                    context=_context(agent={}),
                    question=question,
                    plan=plan,
                )

                self.assertEqual(("financial",), terms)
                self.assertEqual((), warnings)
                self.assertEqual([], model_client.json_requests)

    async def test_km_topic_cjk_inputs_add_english_keyword_search_branch(self):
        cases = (
            (
                "zh-CN", "列出关于金融的asset", "金融",
                ("finance", "financial"),
            ),
            (
                "ja-JP", "金融詐欺に関するasset", "金融詐欺",
                ("finance fraud", "financial fraud"),
            ),
            (
                "ko-KR", "금융 관련 asset", "금융",
                ("finance", "financial"),
            ),
        )
        for language, question, original, english_topics in cases:
            with self.subTest(language=language):
                model_client = _ModelClient(response={
                    "source_language": language,
                    "original_topic": original,
                    "english_topics": english_topics,
                })
                executor = SemanticDataQueryExecutor(
                    client=None,
                    model_client=model_client,
                    prompt_resolver=_PromptResolver(),
                )
                plan = DataQueryPlanV1(
                    semantic_model_id=uuid7(),
                    semantic_model_version=1,
                    dataset="assets",
                    measures=({
                        "name": "asset_count",
                        "aggregation": "COUNT",
                    },),
                    filters=({
                        "field": "topic",
                        "operator": "CONTAINS",
                        "values": (original,),
                    },),
                )
                context = _context(agent={
                    "models": {
                        "data_planner_llm": {
                            "served_model_name": "planner-model"
                        }
                    }
                })

                terms, warnings = await executor._km_topic_terms(
                    context=context,
                    question=question,
                    plan=plan,
                )

                self.assertEqual((original, *english_topics), terms)
                self.assertEqual((), warnings)
                self.assertEqual(1, len(model_client.json_requests))

    async def test_km_multilingual_enumeration_runs_in_parallel_and_deduplicates(self):
        executor = SemanticDataQueryExecutor(
            client=None,
            model_client=_ModelClient(),
            prompt_resolver=_PromptResolver(),
        )
        model_id = uuid7()
        list_plan = DataQueryPlanV1(
            semantic_model_id=model_id,
            semantic_model_version=1,
            dataset="assets",
            measures=({"name": "asset_count", "aggregation": "COUNT"},),
            dimensions=(
                "asset_id", "title", "bundle_id", "bundle_revision_id",
                "asset_date",
            ),
            filters=({
                "field": "topic",
                "operator": "CONTAINS",
                "values": ("金融",),
            },),
            order_by=({"field": "asset_date", "direction": "DESC"},),
            limit=10,
        )
        rows = {
            "topic-original-list": [
                {"asset_id": "A", "title": "A", "asset_date": "2026-01-01"},
                {"asset_id": "B", "title": "B", "asset_date": "2026-03-01"},
            ],
            "topic-english-list": [
                {"asset_id": "B", "title": "B", "asset_date": "2026-03-01"},
                {"asset_id": "C", "title": "C", "asset_date": "2026-02-01"},
            ],
        }
        started: set[str] = set()
        submitted_plans: dict[str, DataQueryPlanV1] = {}
        all_started = asyncio.Event()

        async def run_plan(**kwargs):
            suffix = kwargs["idempotency_suffix"]
            started.add(suffix)
            submitted_plans[suffix] = kwargs["plan"]
            if len(started) == 2:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), timeout=1)
            return uuid7(), {
                "columns": [{"name": "asset_id"}],
                "preview_rows": rows[suffix],
                "truncated": False,
                "provenance": {},
            }

        executor._run_plan = run_plan
        auth = AuthContext(
            principal_kind=PrincipalKind.SERVICE,
            client_id="test",
            calling_service="test",
            request_id="request-1",
            trace_id="trace-1",
            domain_id="20",
            asserted_user_id="user-1",
        )

        result = await executor._execute_km_asset_enumeration(
            context=_context(),
            question="列出关于金融的asset",
            consumer_app_id="km_asset",
            agent_version_id=uuid7(),
            auth_context=auth,
            list_plan=list_plan,
            topic_terms=("金融", "finance", "financial"),
            expansion_warnings=(),
        )

        self.assertEqual(2, len(started))
        self.assertEqual(3, result.row_count)
        self.assertEqual(["B", "C", "A"], [row["asset_id"] for row in result.rows])
        self.assertFalse(result.truncated)
        self.assertEqual(
            "ORIGINAL_AND_ENGLISH_PARALLEL",
            result.provenance["topic_search_mode"],
        )
        self.assertEqual(
            ["金融", "finance", "financial"],
            result.provenance["topic_terms"],
        )
        self.assertEqual(
            ("finance", "financial"),
            next(
                item.values
                for item in submitted_plans["topic-english-list"].filters
                if item.field == "topic"
            ),
        )
        self.assertEqual(11, submitted_plans["topic-original-list"].limit)
        self.assertEqual(11, submitted_plans["topic-english-list"].limit)

    async def test_km_asset_enumeration_returns_ten_and_marks_remainder_truncated(self):
        executor = SemanticDataQueryExecutor(
            client=None,
            model_client=_ModelClient(),
            prompt_resolver=_PromptResolver(),
        )
        list_plan = DataQueryPlanV1(
            semantic_model_id=uuid7(),
            semantic_model_version=1,
            dataset="assets",
            measures=({"name": "asset_count", "aggregation": "COUNT"},),
            dimensions=(
                "asset_id", "title", "bundle_id", "bundle_revision_id",
                "asset_date",
            ),
            order_by=({"field": "asset_date", "direction": "DESC"},),
            limit=10,
        )
        submitted_plans: list[DataQueryPlanV1] = []

        async def run_plan(**kwargs):
            submitted_plans.append(kwargs["plan"])
            return uuid7(), {
                "columns": [{"name": "asset_id"}],
                "preview_rows": [
                    {"asset_id": f"A{index}", "title": f"Asset {index}"}
                    for index in range(11)
                ],
                "truncated": False,
                "provenance": {},
            }

        executor._run_plan = run_plan
        auth = AuthContext(
            principal_kind=PrincipalKind.SERVICE,
            client_id="test",
            calling_service="test",
            request_id="request-1",
            trace_id="trace-1",
            domain_id="20",
            asserted_user_id="user-1",
        )

        result = await executor._execute_km_asset_enumeration(
            context=_context(),
            question="list all assets",
            consumer_app_id="km_asset",
            agent_version_id=uuid7(),
            auth_context=auth,
            list_plan=list_plan,
            topic_terms=(),
            expansion_warnings=(),
        )

        self.assertEqual(1, len(submitted_plans))
        self.assertEqual(11, submitted_plans[0].limit)
        self.assertEqual(10, len(result.rows))
        self.assertEqual(11, result.row_count)
        self.assertTrue(result.truncated)
        self.assertFalse(result.provenance["count_exact"])
        self.assertIn("相关 Asset 超过 10 个，清单已截断", result.warnings)

    async def test_km_multilingual_count_uses_exact_set_union(self):
        executor = SemanticDataQueryExecutor(
            client=None,
            model_client=_ModelClient(),
            prompt_resolver=_PromptResolver(),
        )
        plan = DataQueryPlanV1(
            semantic_model_id=uuid7(),
            semantic_model_version=1,
            dataset="assets",
            measures=({"name": "asset_count", "aggregation": "COUNT"},),
            filters=({
                "field": "topic",
                "operator": "CONTAINS",
                "values": ("金融",),
            },),
            limit=1,
        )
        counts = {
            "topic-original-count": 2,
            "topic-english-count": 4,
            "topic-overlap-count": 1,
        }
        started: set[str] = set()
        all_started = asyncio.Event()

        async def run_plan(**kwargs):
            suffix = kwargs["idempotency_suffix"]
            started.add(suffix)
            if len(started) == 3:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), timeout=1)
            return uuid7(), {
                "columns": [{"name": "asset_count"}],
                "preview_rows": [{"asset_count": counts[suffix]}],
                "truncated": False,
                "provenance": {},
            }

        executor._run_plan = run_plan
        auth = AuthContext(
            principal_kind=PrincipalKind.SERVICE,
            client_id="test",
            calling_service="test",
            request_id="request-1",
            trace_id="trace-1",
            domain_id="20",
            asserted_user_id="user-1",
        )

        result = await executor._execute_km_asset_multilingual_count(
            context=_context(),
            question="有多少关于金融的asset",
            consumer_app_id="km_asset",
            agent_version_id=uuid7(),
            auth_context=auth,
            plan=plan,
            topic_terms=("金融", "finance", "financial"),
            expansion_warnings=(),
        )

        self.assertEqual(3, len(started))
        self.assertEqual(1, result.row_count)
        self.assertEqual(5, result.rows[0]["asset_count"])
        self.assertTrue(result.provenance["count_exact"])

    async def test_router_only_exposes_selected_knowledge_capabilities(self):
        model = _ModelClient(
            response={
                "route_type": "HYBRID_PARALLEL",
                "confidence": 0.9,
                "reason": "需要同时查询制度和业务数据",
                "clarification_question": None,
                "requires_chart": False,
                "context_required": False,
            }
        )
        planner = _root_planner(
            model_client=model,
            prompt_resolver=_PromptResolver(),
        )

        decision = await planner.decide_for_input(
            agent_snapshot={
                "enabled_capabilities": ["document", "data_query"],
                "models": {
                    "router_llm": {
                        "served_model_name": "router-model"
                    }
                },
            },
            objective="根据员工套餐制度统计本月使用人数",
        )

        request = json.loads(
            model.last_json_request["prompt"][2]["content"]
        )
        self.assertNotIn("CONVERSATION", request["enabled_routes"])
        self.assertIn("DOCUMENT", request["enabled_routes"])
        self.assertIn("DATA_QUERY", request["enabled_routes"])
        self.assertEqual(decision.route_type, RouteType.HYBRID_PARALLEL)

    async def test_km_topic_enumeration_uses_data_first_hybrid(self):
        model = _ModelClient(
            response=_asset_plan_response(semantic=True)
        )
        planner = _root_planner(
            model_client=model,
            prompt_resolver=_PromptResolver(),
        )

        decision = await planner.decide_for_input(
            agent_snapshot={
                "owner_app_id": "km_asset",
                "enabled_capabilities": ["document", "data_query"],
                "models": {
                    "router_llm": {"served_model_name": "router-model"}
                },
            },
            objective="查一下关于 ChatBI 的 asset",
        )

        self.assertEqual(RouteType.HYBRID_DATA_FIRST, decision.route_type)
        self.assertEqual("BALANCED", decision.coverage_mode)
        self.assertEqual(
            "asset-search-plan-v1:1.0.0", decision.classifier_version
        )
        self.assertIsNotNone(model.last_json_request)

    async def test_km_aggregate_question_uses_data_query(self):
        planner = _root_planner(
            model_client=_ModelClient(response=_asset_plan_response(
                operation="COUNT", author="THASNEEM.FATHIMA"
            )),
            prompt_resolver=_PromptResolver(),
        )

        decision = await planner.decide_for_input(
            agent_snapshot={
                "owner_app_id": "km_asset",
                "enabled_capabilities": ["document", "data_query"],
                "models": {
                    "router_llm": {"served_model_name": "router-model"}
                },
            },
            objective="THASNEEM.FATHIMA 发布了多少个 asset",
        )

        self.assertEqual(RouteType.DATA_QUERY, decision.route_type)

    async def test_km_exact_metadata_list_uses_enumeration_basis(self):
        planner = _root_planner(
            model_client=_ModelClient(response=_asset_plan_response()),
            prompt_resolver=_PromptResolver(),
        )

        decision = await planner.decide_for_input(
            agent_snapshot={
                "owner_app_id": "km_asset",
                "enabled_capabilities": ["document", "data_query"],
                "models": {
                    "router_llm": {"served_model_name": "router-model"}
                },
            },
            objective="Show assets sorted by domain",
        )

        self.assertEqual(RouteType.DATA_QUERY, decision.route_type)
        self.assertEqual(
            KMAnswerBasis.EXACT_METADATA_ENUMERATION,
            decision.answer_basis,
        )

    async def test_km_colloquial_count_questions_use_data_query(self):
        model = _ModelClient(response=_asset_plan_response(operation="COUNT"))
        planner = _root_planner(
            model_client=model,
            prompt_resolver=_PromptResolver(),
        )
        agent = {
            "owner_app_id": "km_asset",
            "enabled_capabilities": ["document", "data_query"],
            "models": {
                "router_llm": {"served_model_name": "router-model"}
            },
        }

        for objective in (
            "现在有几个asset",
            "总共有几条 Asset",
            "知识库中的 Asset 总数",
        ):
            with self.subTest(objective=objective):
                decision = await planner.decide_for_input(
                    agent_snapshot=agent,
                    objective=objective,
                )
                self.assertEqual(RouteType.DATA_QUERY, decision.route_type)

    async def test_km_multilingual_intents_use_semantic_router(self):
        cases = (
            (
                "How many assets are available?",
                RouteType.DATA_QUERY,
                "en-US",
                "BALANCED",
                KMAnswerBasis.UNSCOPED_AGGREGATE,
            ),
            (
                "현재 자산은 몇 개입니까?",
                RouteType.DATA_QUERY,
                "ko-KR",
                "BALANCED",
                KMAnswerBasis.UNSCOPED_AGGREGATE,
            ),
            (
                "現在のアセット数はいくつですか？",
                RouteType.DATA_QUERY,
                "ja-JP",
                "BALANCED",
                KMAnswerBasis.UNSCOPED_AGGREGATE,
            ),
            (
                "how many asset about OAC",
                RouteType.HYBRID_DATA_FIRST,
                "en-US",
                "BALANCED",
                KMAnswerBasis.SEMANTIC_RELEVANCE_ENUMERATION,
            ),
            (
                "有多少关于 OAC 的 asset",
                RouteType.HYBRID_DATA_FIRST,
                "zh-CN",
                "BALANCED",
                KMAnswerBasis.SEMANTIC_RELEVANCE_ENUMERATION,
            ),
            (
                "Find assets related to ChatBI",
                RouteType.HYBRID_DATA_FIRST,
                "en-US",
                "BALANCED",
                KMAnswerBasis.SEMANTIC_RELEVANCE_ENUMERATION,
            ),
            (
                "ChatBI 관련 자료를 찾아주세요",
                RouteType.HYBRID_DATA_FIRST,
                "ko-KR",
                "BALANCED",
                KMAnswerBasis.SEMANTIC_RELEVANCE_ENUMERATION,
            ),
            (
                "ChatBIに関連する資料を探してください",
                RouteType.HYBRID_DATA_FIRST,
                "ja-JP",
                "BALANCED",
                KMAnswerBasis.SEMANTIC_RELEVANCE_ENUMERATION,
            ),
        )
        agent = {
            "owner_app_id": "km_asset",
            "enabled_capabilities": ["document", "data_query"],
            "models": {
                "router_llm": {"served_model_name": "router-model"}
            },
        }
        for (
            objective,
            expected,
            expected_language,
            coverage_mode,
            answer_basis,
        ) in cases:
            with self.subTest(objective=objective):
                semantic = answer_basis == KMAnswerBasis.SEMANTIC_RELEVANCE_ENUMERATION
                operation = (
                    "COUNT"
                    if answer_basis == KMAnswerBasis.UNSCOPED_AGGREGATE
                    or "how many" in objective.casefold()
                    or "多少" in objective
                    else "LIST"
                )
                model = _ModelClient(response=_asset_plan_response(
                    operation=operation, semantic=semantic
                ))
                planner = _root_planner(
                    model_client=model,
                    prompt_resolver=_PromptResolver(),
                )
                decision = await planner.decide_for_input(
                    agent_snapshot=agent,
                    objective=objective,
                )
                self.assertEqual(expected, decision.route_type)
                self.assertEqual(coverage_mode, decision.coverage_mode)
                self.assertEqual(answer_basis, decision.answer_basis)
                messages = model.last_json_request["prompt"]
                request = json.loads(messages[-1]["content"])
                self.assertEqual(expected_language, request["language"])

    async def test_km_follow_up_resolves_previous_count_scope_without_clarify(self):
        model = _ModelClient(response=_asset_plan_response(
            operation="COUNT", semantic=True
        ))
        planner = _root_planner(
            model_client=model,
            prompt_resolver=_PromptResolver(),
        )
        context = {
            "summary": {},
            "recent_items": [
                {
                    "role": "USER",
                    "content": {"text": "现在有几个asset"},
                    "item_sequence": 1,
                },
                {
                    "role": "ASSISTANT",
                    "content": {
                        "text": (
                            "您指的是知识库中所有 asset 的总数，"
                            "还是特指与 chatbi 相关的 asset 数量？"
                        )
                    },
                    "item_sequence": 2,
                },
            ],
        }

        decision = await planner.decide_for_input(
            agent_snapshot={
                "owner_app_id": "km_asset",
                "enabled_capabilities": ["document", "data_query"],
                "models": {
                    "router_llm": {"served_model_name": "router-model"}
                },
            },
            objective="chatbi相关的",
            conversation_context=context,
        )

        self.assertEqual(RouteType.HYBRID_DATA_FIRST, decision.route_type)
        self.assertIsNone(decision.clarification_question)
        request = json.loads(model.last_json_request["prompt"][-1]["content"])
        self.assertEqual("chatbi相关的", request["current_input"])
        self.assertEqual(context["recent_items"], request["recent_items"])
        self.assertEqual(1, len(model.json_requests))

    async def test_km_topic_count_rejects_document_route_and_repairs(self):
        model = _ModelClient(response=_asset_plan_response(
            operation="COUNT", semantic=True
        ))
        planner = _root_planner(
            model_client=model,
            prompt_resolver=_PromptResolver(),
        )

        decision = await planner.decide_for_input(
            agent_snapshot={
                "owner_app_id": "km_asset",
                "enabled_capabilities": ["document", "data_query"],
                "models": {
                    "router_llm": {"served_model_name": "router-model"}
                },
            },
            objective="how many asset about OAC",
        )

        self.assertEqual(RouteType.HYBRID_DATA_FIRST, decision.route_type)
        self.assertEqual(
            KMAnswerBasis.SEMANTIC_RELEVANCE_ENUMERATION,
            decision.answer_basis,
        )
        self.assertEqual(1, len(model.json_requests))
        self.assertEqual(
            ("SEMANTIC_TOTAL_COUNT",),
            decision.asset_search_plan.unsupported_requests,
        )

    async def test_km_genuine_ambiguity_can_request_clarification(self):
        model = _ModelClient(response=_asset_plan_response(ambiguities=({
            "code": "MISSING_SCOPE",
            "question": "请说明您要查询哪个 Asset，以及需要内容还是统计数据。",
        },)))
        planner = _root_planner(
            model_client=model,
            prompt_resolver=_PromptResolver(),
        )

        decision = await planner.decide_for_input(
            agent_snapshot={
                "owner_app_id": "km_asset",
                "enabled_capabilities": ["document", "data_query"],
                "models": {
                    "router_llm": {"served_model_name": "router-model"}
                },
            },
            objective="那个呢",
        )

        self.assertEqual(RouteType.CLARIFY, decision.route_type)
        self.assertTrue(decision.clarification_question)

    async def test_multi_capability_router_selects_data_chart(self):
        planner = _root_planner(
            model_client=_ModelClient(
                response={
                    "route_type": "DATA_QUERY",
                    "confidence": 0.97,
                    "reason": "需要查询销售统计并绘图",
                    "clarification_question": None,
                    "requires_chart": True,
                }
            ),
            prompt_resolver=_PromptResolver(),
        )
        decision = await planner.decide_for_input(
            agent_snapshot={
                "enabled_capabilities": ["document", "data_query"],
                "models": {
                    "router_llm": {
                        "served_model_name": "router-model"
                    }
                },
            },
            objective="画出本月销售趋势图",
        )
        plan = planner.build_plan(
            objective="画出本月销售趋势图", decision=decision
        )

        self.assertEqual(decision.route_type, RouteType.DATA_QUERY)
        self.assertTrue(decision.requires_chart)
        self.assertEqual(
            [task.task_key for task in plan.tasks],
            [
                "context_rewrite",
                "data_query",
                "echarts",
                "response_compose",
            ],
        )

    async def test_query_image_routes_to_document(self):
        planner = _root_planner()
        decision = await planner.decide_for_input(
            agent_snapshot={
                "enabled_capabilities": ["conversation", "document"],
                "models": {
                    "router_llm": {
                        "served_model_name": "router-model"
                    }
                },
            },
            objective="找一下相似的案例",
            client_metadata={"query_images": [{"storage_uri": "/tmp/x"}]},
        )

        self.assertEqual(decision.route_type, RouteType.DOCUMENT)

    async def test_managed_resources_falls_back_to_document(self):
        planner = _root_planner(
            model_client=_ModelClient(
                response={
                    "route_type": "CLARIFY",
                    "confidence": 0.35,
                    "reason": "套餐含义不明确",
                    "clarification_question": (
                        "请说明这是通用问题、文档查询还是业务数据查询。"
                    ),
                    "requires_chart": False,
                }
            ),
            prompt_resolver=_PromptResolver(),
        )

        decision = await planner.decide_for_input(
            agent_snapshot={
                "enabled_capabilities": [
                    "conversation",
                    "document",
                    "data_query",
                ],
                "models": {
                    "router_llm": {
                        "served_model_name": "router-model"
                    }
                },
                "config": {"resource_mode": "managed_resources"},
            },
            objective="员工有哪些套餐",
        )
        plan = planner.build_plan(
            objective="员工有哪些套餐",
            decision=decision,
        )

        self.assertEqual(RouteType.DOCUMENT, decision.route_type)
        self.assertIsNone(decision.clarification_question)
        self.assertIn(
            "knowledge_retrieval",
            [task.task_key for task in plan.tasks],
        )

    async def test_mcp_data_skill_preserves_profile_and_rows(self):
        client = _DataClient()
        result = await DataQuerySkill(
            mcp_executor=MCPDataQueryExecutor(client=client),
            semantic_executor=_UnavailableSemanticExecutor(),
        ).execute(
            _context(
                original_input="查询销售额",
                agent={
                    "config": {
                        "data_query_mode": "MCP",
                        "data_profile_name": "SALES_PROFILE",
                    },
                },
                artifacts=(
                    _artifact(
                        "CONTEXT_REWRITE",
                        {"standalone_query": "查询 2026 年 7 月销售额"},
                    ),
                ),
            )
        )

        self.assertEqual(result.artifact.artifact_type, "QUERY_RESULT")
        self.assertEqual(result.artifact.schema_version, "QUERY_RESULT.v1")
        self.assertEqual(result.artifact.payload["schema"], "QUERY_RESULT.v1")
        self.assertEqual(result.artifact.payload["provider"], "MCP")
        self.assertEqual(
            result.artifact.payload["rows"][0]["sales"], 10
        )
        self.assertEqual(client.request["profile"], "SALES_PROFILE")
        self.assertEqual(
            client.request["question"], "查询 2026 年 7 月销售额"
        )

    async def test_data_answer_prompt_uses_frozen_response_language(self):
        query = QueryResult.model_validate({
            "schema": "QUERY_RESULT.v1",
            "query_result_id": str(uuid7()),
            "provider": "MCP",
            "columns": [{"name": "asset_count"}],
            "rows": [{"asset_count": 3}],
            "row_count": 1,
            "truncated": False,
            "warnings": [],
            "provenance": {"profile": "KM_ASSET"},
        })
        skill = ResponseComposerSkill(
            model_client=_ModelClient(),
            prompt_resolver=_PromptResolver(),
        )

        _, messages = await skill._query_prompt(
            _context(
                original_input="현재 자산은 몇 개입니까?",
                language="ko-KR",
                agent={
                    "models": {
                        "composer_llm": {
                            "served_model_name": "composer-model"
                        }
                    }
                },
            ),
            query,
        )

        self.assertIn("language=ko-KR", messages[0]["content"])
        self.assertIn("language=ko-KR", messages[1]["content"])

    async def test_truncated_query_cannot_claim_all_results_are_shown(self):
        query = QueryResult.model_validate({
            "query_result_id": str(uuid7()),
            "provider": "SEMANTIC",
            "columns": [{"name": "asset_id"}],
            "rows": [
                {"asset_id": f"A{index}"} for index in range(10)
            ],
            "row_count": 11,
            "truncated": True,
            "warnings": [],
            "provenance": {
                "count_exact": False,
                "display_limit": 10,
            },
        })
        skill = ResponseComposerSkill(
            model_client=_ModelClient(),
            prompt_resolver=_PromptResolver(),
        )
        context = _context(
            original_input="Show assets sorted by domain",
            language="en-US",
            agent={
                "models": {
                    "composer_llm": {
                        "served_model_name": "composer-model"
                    }
                }
            },
        )

        _, messages = await skill._query_prompt(context, query)
        self.assertIn("不得声称这是全部结果", messages[0]["content"])

        result = skill._query_result_artifact(
            context,
            query,
            "| Asset |\n|---|\n| A0 |",
        )
        answer = str(result.artifact.payload["answer"])
        self.assertIn("Showing the first 10 results", answer)
        self.assertIn("no full count was run", answer)
        self.assertIn(
            "问数结果已按服务端上限截断",
            result.artifact.payload["warnings"],
        )

    async def test_semantic_and_mcp_share_query_result_contract(self):
        semantic_model_id = uuid7()

        class _SemanticClient:
            async def get_planning_context(self, **kwargs):
                return {"models": [{
                    "semantic_model_id": str(semantic_model_id),
                    "semantic_model_version": 1,
                    "display_name": "销售",
                    "datasets": [{"name": "sales"}],
                    "dimensions": [],
                    "measures": [{"name": "amount"}],
                    "max_rows": 100,
                }]}

            async def create_run(self, **kwargs):
                return {"data_query_run_id": str(uuid7())}

            async def get_run(self, **kwargs):
                return {"status": "COMPLETED"}

            async def get_result(self, **kwargs):
                return {
                    "columns": [{"name": "amount"}],
                    "preview_rows": [{"amount": 10}],
                    "row_count": 1,
                    "observed_row_count": 1,
                    "truncated": False,
                    "provenance": {
                        "data_source_id": str(uuid7()),
                        "semantic_model_id": str(semantic_model_id),
                        "semantic_model_version": "1",
                        "query_plan_hash": "a" * 64,
                    },
                }

        model = _ModelClient(response={
            "contract_version": "DataQueryPlan.v1",
            "semantic_model_id": str(semantic_model_id),
            "semantic_model_version": 1,
            "dataset": "sales",
            "measures": [{"name": "amount", "aggregation": "SUM"}],
            "dimensions": [],
            "filters": [],
            "order_by": [],
            "limit": 100,
            "time_zone": "Asia/Shanghai",
        })
        auth = AuthContext(
            principal_kind=PrincipalKind.SERVICE,
            client_id="test",
            calling_service="test",
            request_id="request-1",
            trace_id="trace-1",
            domain_id="20",
            asserted_user_id="user-1",
        )
        semantic = SemanticDataQueryExecutor(
            client=_SemanticClient(),
            model_client=model,
            prompt_resolver=_PromptResolver(),
        )
        result = await DataQuerySkill(
            mcp_executor=MCPDataQueryExecutor(client=None),
            semantic_executor=semantic,
        ).execute(_context(
            original_input="查询销售额",
            agent={
                "config": {"data_query_mode": "SEMANTIC"},
                "agent_version_id": str(uuid7()),
                "models": {"data_planner_llm": {"served_model_name": "planner"}},
            },
            policy_snapshot={"auth_context": auth.model_dump(mode="json")},
        ))

        self.assertEqual(result.artifact.schema_version, "QUERY_RESULT.v1")
        self.assertEqual(result.artifact.payload["schema"], "QUERY_RESULT.v1")
        self.assertEqual(result.artifact.payload["provider"], "SEMANTIC")
        self.assertEqual(result.artifact.payload["rows"], [{"amount": 10}])

    async def test_data_query_stream_only_returns_final_artifact(self):
        skill = DataQuerySkill(
            mcp_executor=MCPDataQueryExecutor(client=_DataClient()),
            semantic_executor=_UnavailableSemanticExecutor(),
        )

        items = [
            item async for item in skill.execute_stream(_context(
                original_input="查询销售额",
                agent={
                    "config": {
                        "data_query_mode": "MCP",
                        "data_profile_name": "SALES_PROFILE",
                    },
                },
            ))
        ]

        self.assertEqual(1, len(items))
        self.assertIsInstance(items[0], SkillResult)

    def test_km_enumeration_threshold_message_is_exact(self):
        complete = ResponseComposerSkill._enumeration_prefix(
            language="zh-CN",
            total_count=5,
            shown_count=5,
            truncated=False,
            source_truncated=False,
        )
        clipped = ResponseComposerSkill._enumeration_prefix(
            language="zh-CN",
            total_count=12,
            shown_count=10,
            truncated=True,
            source_truncated=False,
        )

        self.assertIn("共命中 5 个", complete)
        self.assertIn("全部列出", complete)
        self.assertIn("共命中 12 个", clipped)
        self.assertIn("前 10 个", clipped)
        self.assertIn("已截断", clipped)

    def test_km_enumeration_rejects_serialized_rows_and_asset_ids(self):
        assets = [{
            "asset_id": "ASSET-1",
            "title": "APEX Asset",
            "product": "Application Express",
            "solution": "Development",
        }]

        with self.assertRaisesRegex(ValueError, "序列化容器"):
            ResponseComposerSkill._validate_enumeration_body(
                "[{'asset_id': 'ASSET-1', 'title': 'APEX Asset'}]",
                assets=assets,
                allowed={},
                language="en-US",
            )
        with self.assertRaisesRegex(ValueError, "asset_id"):
            ResponseComposerSkill._validate_enumeration_body(
                "1. **APEX Asset** — asset_id: ASSET-1 [Q1]",
                assets=assets,
                allowed={},
                language="en-US",
            )

    def test_km_enumeration_fallback_keeps_bundle_citations(self):
        first_bundle_id = uuid7()
        second_bundle_id = uuid7()
        allowed = {
            "C1": SimpleNamespace(bundle_id=first_bundle_id),
            "C2": SimpleNamespace(bundle_id=second_bundle_id),
        }
        assets = [
            {
                "asset_id": "ASSET-1",
                "title": "First APEX Asset",
                "bundle_id": str(first_bundle_id),
            },
            {
                "asset_id": "ASSET-2",
                "title": "Second APEX Asset",
                "bundle_id": str(second_bundle_id),
            },
        ]

        answer = ResponseComposerSkill._enumeration_fallback(
            assets, language="en-US", allowed=allowed
        )

        self.assertIn("First APEX Asset** [Q1] [C1]", answer)
        self.assertIn("Second APEX Asset** [Q1] [C2]", answer)
        ResponseComposerSkill._validate_enumeration_body(
            answer,
            assets=assets,
            allowed=allowed,
            language="en-US",
        )

    def test_km_metadata_assets_each_keep_query_citation_without_documents(self):
        assets = [
            {"title": "First Asset", "bundle_id": str(uuid7())},
            {"title": "Second Asset", "bundle_id": str(uuid7())},
        ]

        selected = ResponseComposerSkill._select_result_assets(
            assets, (), semantic=False, result_limit=10
        )
        answer = ResponseComposerSkill._enumeration_fallback(
            selected, language="en-US"
        )

        self.assertEqual(assets, selected)
        self.assertIn("First Asset** [Q1]", answer)
        self.assertIn("Second Asset** [Q1]", answer)
        self.assertEqual(2, answer.count("[Q1]"))

    def test_km_semantic_assets_require_same_bundle_document_citation(self):
        cited_bundle = uuid7()
        uncited_bundle = uuid7()
        assets = [
            {"title": "Cited Asset", "bundle_id": str(cited_bundle)},
            {"title": "Uncited Asset", "bundle_id": str(uncited_bundle)},
        ]
        citations = (SimpleNamespace(bundle_id=cited_bundle),)

        selected = ResponseComposerSkill._select_result_assets(
            assets, citations, semantic=True, result_limit=10
        )

        self.assertEqual([assets[0]], selected)

    def test_km_enumeration_rejects_cross_bundle_citation(self):
        first_bundle_id = uuid7()
        second_bundle_id = uuid7()
        assets = [
            {
                "title": "First APEX Asset",
                "bundle_id": str(first_bundle_id),
            },
            {
                "title": "Second APEX Asset",
                "bundle_id": str(second_bundle_id),
            },
        ]
        allowed = {
            "C1": SimpleNamespace(bundle_id=first_bundle_id),
            "C2": SimpleNamespace(bundle_id=second_bundle_id),
        }

        with self.assertRaisesRegex(ValueError, "Bundle"):
            ResponseComposerSkill._validate_enumeration_body(
                "1. **First APEX Asset** [Q1] [C2]\n"
                "2. **Second APEX Asset** [Q1] [C1]",
                assets=assets,
                allowed=allowed,
                language="en-US",
            )
    def test_km_enumeration_can_restore_assets_from_query_rows(self):
        query = QueryResult(
            query_result_id=uuid7(),
            provider="SEMANTIC",
            columns=(),
            rows=({
                "ASSET_ID": "asset-1",
                "TITLE": "OAC Asset",
                "PRODUCT": "OAC",
                "SOLUTION": "Analytics",
                "BUNDLE_ID": str(uuid7()),
                "BUNDLE_REVISION_ID": str(uuid7()),
            },),
            row_count=1,
            provenance={},
        )

        assets = ResponseComposerSkill._enumeration_assets_from_query(query)

        self.assertEqual(1, len(assets))
        self.assertEqual("asset-1", assets[0]["asset_id"])
        self.assertEqual("OAC Asset", assets[0]["title"])

    def test_km_document_scope_decodes_oracle_raw_bundle_ids(self):
        query = QueryResult(
            query_result_id=uuid7(),
            provider="SEMANTIC",
            columns=(),
            rows=({
                "asset_id": "ASSET-1",
                "title": "APEX Asset",
                "bundle_id": {
                    "encoding": "base64",
                    "value": "AaATx4pDcfuZFydoc6rFpg==",
                },
                "bundle_revision_id": {
                    "encoding": "base64",
                    "value": "AaATx4vfeai2yCm4reAl7g==",
                },
            },),
            row_count=1,
            provenance={},
        )
        context = _context(
            artifacts=(_artifact(
                "QUERY_RESULT", query.model_dump(mode="json")
            ),),
            route={
                "answer_basis": "SEMANTIC_RELEVANCE_ENUMERATION"
            },
            original_input="list assets related to apex",
        )

        result = KmAssetDocumentScopeExtractSkill._enumeration_scope(
            context
        )

        self.assertEqual([{
            "bundle_id": "01a013c7-8a43-71fb-9917-276873aac5a6",
            "bundle_revision_id": (
                "01a013c7-8bdf-79a8-b6c8-29b8ade025ee"
            ),
            "title": "APEX Asset",
            "asset_id": "ASSET-1",
        }], result.artifact.payload["bundle_targets"])
        self.assertEqual(
            "01a013c7-8a43-71fb-9917-276873aac5a6",
            result.artifact.payload["assets"][0]["bundle_id"],
        )

    async def test_conversation_response_streams_and_returns_answer(self):
        skill = ConversationResponseSkill(
            model_client=_ModelClient(chunks=("你", "好")),
            prompt_resolver=_PromptResolver(),
        )
        items = [
            item
            async for item in skill.execute_stream(
                _context(
                    agent={
                        "models": {
                            "composer_llm": {
                                "served_model_name": "chat-model"
                            }
                        }
                    }
                )
            )
        ]

        self.assertTrue(any(isinstance(item, SkillProgress) for item in items))
        result = next(
            item for item in items if isinstance(item, SkillResult)
        )
        self.assertEqual(result.artifact.payload["answer"], "你好")
        self.assertEqual(result.artifact.payload["references"], [])

    async def test_route_clarification_is_returned_as_chat_answer(self):
        skill = ConversationResponseSkill(
            model_client=_ModelClient(),
            prompt_resolver=_PromptResolver(),
        )
        items = [
            item
            async for item in skill.execute_stream(
                _context(
                    route={
                        "route_type": "CLARIFY",
                        "reason": "置信度不足",
                        "clarification_question": "请问要查文档还是业务数据？",
                    }
                )
            )
        ]
        result = next(
            item for item in items if isinstance(item, SkillResult)
        )

        self.assertEqual(
            result.artifact.payload["status"],
            "CLARIFICATION_REQUIRED",
        )
        self.assertIn("文档还是业务数据", result.artifact.payload["answer"])

    async def test_echarts_rejects_executable_formatter(self):
        skill = EChartsSkill(
            model_client=_ModelClient(
                response={
                    "chart_type": "bar",
                    "title": "销售",
                    "option": {
                        "tooltip": {
                            "formatter": "function(x){return x.value}"
                        }
                    },
                }
            ),
            prompt_resolver=_PromptResolver(),
        )
        query = {
            "schema": "QUERY_RESULT.v1",
            "query_result_id": str(uuid7()),
            "provider": "MCP",
            "columns": [{"name": "sales"}],
            "rows": [{"sales": 10}],
            "row_count": 1,
            "truncated": False,
            "warnings": [],
            "provenance": {"profile": "SALES_PROFILE"},
        }
        with self.assertRaisesRegex(ValueError, "可执行脚本"):
            await skill.execute(
                _context(
                    agent={
                        "models": {
                            "composer_llm": {
                                "served_model_name": "chart-model"
                            }
                        }
                    },
                    artifacts=(_artifact("QUERY_RESULT", query),),
                )
            )

    def test_dify_records_use_document_evidence(self):
        bundle_id = uuid7()
        records = _records(
            candidates=[
                {
                    "collection_id": str(uuid7()),
                    "bundle_id": str(bundle_id),
                    "bundle_revision_id": str(uuid7()),
                    "display_title": "案例文档",
                    "rrf_score": 0.02,
                }
            ],
            citations=[
                {
                    "bundle_id": str(bundle_id),
                    "items": [
                        {
                            "final_role": "PRIMARY",
                            "evidence": {
                                "evidence_id": str(uuid7()),
                                "document_id": str(uuid7()),
                                "document_version_id": str(uuid7()),
                                "document_name": "case.pdf",
                                "content_text": "这是可引用的案例正文。",
                                "locator": {"page": 3},
                            },
                        }
                    ],
                }
            ],
            limit=10,
            threshold=0.5,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["title"], "case.pdf")
        self.assertEqual(records[0]["score"], 1.0)
        self.assertIn("案例正文", records[0]["content"])


if __name__ == "__main__":
    unittest.main()
