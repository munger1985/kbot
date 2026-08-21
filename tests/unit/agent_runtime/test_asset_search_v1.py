"""KM Asset 统一搜索计划、编译和结果展示的聚焦测试。"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agent_runtime.application.commands import LeasedArtifact
from agent_runtime.runtime import ExecutionContext
from agent_runtime.specialists.km_asset.search import (
    AssetSearchDataQueryCompiler,
    AssetSearchPlanner,
)
from agent_runtime.specialists.data_query import QueryResult
from agent_runtime.specialists.document import (
    Citation,
    CitationPack,
    DocumentRetrievalResult,
    RetrievalCoverage,
)
from agent_runtime.specialists.km_asset import (
    KmAssetKnowledgeRetrievalSkill as KnowledgeRetrievalSkill,
    KmAssetResponseComposerSkill as ResponseComposerSkill,
)
from data_query.application.managed_datasets import km_asset_definition
from data_query.connectors.dialect_compiler import compile_dialect_query
from data_query.contracts import SemanticModelDefinition
from platform_core.contracts import AssetSearchPlanV1
from platform_core.identity import uuid7


MODEL_ID = "01900000-0000-7000-8000-000000000003"


def _catalog() -> list[dict]:
    definition = km_asset_definition(schema_name="KBOTUI_DEV")
    return [{
        "semantic_model_id": MODEL_ID,
        "semantic_model_version": 1,
        "max_rows": 1000,
        **definition,
    }]


def _base_plan(**updates) -> AssetSearchPlanV1:
    payload = {
        "query_text": "列出 Asset",
        "language": "zh-CN",
        "operation": "LIST",
        "target": "ASSET",
        "criteria": [],
        "eligibility_expression": None,
        "projection": ["asset_id", "title"],
        "order_by": [{"field": "asset_date", "direction": "DESC"}],
        "display_limit": 10,
        "result_assets": {
            "mode": "PRIMARY",
            "target_count": 10,
            "selection": "REQUESTED_ORDER",
        },
    }
    payload.update(updates)
    return AssetSearchPlanV1.model_validate(payload)


def _artifact(artifact_type: str, payload: dict) -> LeasedArtifact:
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


class AssetSearchV1Test(unittest.IsolatedAsyncioTestCase):
    async def test_completed_asset_list_has_one_c_reference_per_asset(self):
        first_bundle = uuid7()
        second_bundle = uuid7()
        first_revision = uuid7()
        second_revision = uuid7()
        assets = [
            {
                "asset_id": "ASSET-1",
                "title": "First Asset",
                "bundle_id": str(first_bundle),
                "bundle_revision_id": str(first_revision),
            },
            {
                "asset_id": "ASSET-2",
                "title": "Second Asset",
                "bundle_id": str(second_bundle),
                "bundle_revision_id": str(second_revision),
            },
        ]

        def citation(label, bundle_id, revision_id, title):
            return Citation(
                citation_label=label,
                collection_id=uuid7(),
                bundle_id=bundle_id,
                bundle_revision_id=revision_id,
                document_id=uuid7(),
                document_version_id=uuid7(),
                evidence_ids=(uuid7(),),
                title=title,
                bundle_title=title,
                document_role="MANIFEST",
                excerpt=f"Asset Title: {title}",
                locator={},
                locator_schema_version="document/v1",
                relevance_reason="Asset manifest",
            )

        citations = (
            citation("C1", first_bundle, first_revision, "First Asset"),
            citation("C2", second_bundle, second_revision, "Second Asset"),
        )
        retrieval = DocumentRetrievalResult(
            status="READY",
            citation_pack=CitationPack(
                question="list assets",
                query_plan={},
                bundle_candidates=(),
                citations=citations,
                coverage=RetrievalCoverage(
                    candidate_bundle_count=2,
                    selected_document_count=2,
                    selected_evidence_count=2,
                ),
            ),
            retrieval_report={},
        )
        query = QueryResult(
            query_result_id=uuid7(),
            provider="SEMANTIC",
            columns=(),
            rows=tuple(assets),
            row_count=2,
            provenance={"count_exact": True},
        )
        context = ExecutionContext(
            domain_id=20,
            agent_id=uuid7(),
            run_id=uuid7(),
            task_id=uuid7(),
            task_key="test",
            actor_id="user",
            request_id="request",
            trace_id="trace",
            original_input="list assets",
            policy_snapshot={},
            config_snapshot={"agent": {}},
            input_artifacts=(_artifact("DOCUMENT_SCOPE", {
                "assets": assets,
                "total_count": 2,
                "truncated": False,
            }),),
        )
        skill = ResponseComposerSkill(
            model_client=None,
            prompt_resolver=None,
        )

        result = await skill._compose_km_asset_enumeration(
            context,
            query,
            retrieval,
            search_plan=_base_plan(result_assets={
                "mode": "PRIMARY",
                "target_count": 2,
                "selection": "REQUESTED_ORDER",
            }),
        )

        payload = result.artifact.payload
        self.assertEqual("READY", payload["status"])
        self.assertIn("First Asset** [C1]", payload["answer"])
        self.assertIn("Second Asset** [C2]", payload["answer"])
        self.assertNotIn("[Q", payload["answer"])
        self.assertEqual(["C1", "C2"], payload["used_citation_labels"])
        self.assertEqual(2, len(payload["references"]))
        self.assertEqual(
            ["C1", "C2"],
            [
                item["citation_label"]
                for item in payload["references"]
                if item["reference_type"] == "DOCUMENT"
            ],
        )
        self.assertEqual(
            ["First Asset", "Second Asset"],
            [
                item["title"]
                for item in payload["references"]
                if item["reference_type"] == "DOCUMENT"
            ],
        )

    async def test_asset_list_never_falls_back_to_query_only_references(self):
        query = QueryResult(
            query_result_id=uuid7(),
            provider="SEMANTIC",
            columns=(),
            rows=({
                "asset_id": "ASSET-1",
                "title": "Metadata-only Asset",
                "bundle_id": str(uuid7()),
                "bundle_revision_id": str(uuid7()),
            },),
            row_count=1,
            provenance={"count_exact": True},
        )
        context = ExecutionContext(
            domain_id=20,
            agent_id=uuid7(),
            run_id=uuid7(),
            task_id=uuid7(),
            task_key="test",
            actor_id="user",
            request_id="request",
            trace_id="trace",
            original_input="list assets",
            policy_snapshot={},
            config_snapshot={"agent": {}},
            input_artifacts=(),
        )
        skill = ResponseComposerSkill(
            model_client=None,
            prompt_resolver=None,
        )

        result = await skill._compose_asset_query_result(
            context, query, _base_plan()
        )

        self.assertEqual(
            "INSUFFICIENT_EVIDENCE", result.artifact.payload["status"]
        )
        self.assertEqual([], result.artifact.payload["references"])
        self.assertNotIn("Metadata-only Asset", result.artifact.payload["answer"])

    async def test_planner_timeout_falls_back_to_contractual_semantic_list(self):
        async def slow_model_call(**kwargs):
            await asyncio.sleep(1)
            return {}

        model_client = SimpleNamespace(get_llm_json=slow_model_call)
        prompt_resolver = SimpleNamespace(
            resolve=AsyncMock(
                return_value=SimpleNamespace(
                    content="测试规划提示词",
                    version="1.0.0",
                )
            )
        )
        planner = AssetSearchPlanner(
            model_client=model_client,
            prompt_resolver=prompt_resolver,
            timeout_seconds=0.01,
        )

        plan, version = await planner.plan(
            model_name="slow-router",
            question="找个financial fraud的asset",
            language="zh-CN",
            conversation_context=None,
        )

        self.assertEqual("LIST", plan.operation)
        self.assertEqual(5, plan.display_limit)
        self.assertEqual("SEMANTIC_CONCEPT", plan.criteria[0].kind)
        self.assertEqual(
            ("找个financial fraud的asset",),
            plan.criteria[0].values,
        )
        self.assertIn("TITLE", plan.criteria[0].field_scope)
        self.assertIn("CONTENT", plan.criteria[0].field_scope)
        self.assertEqual(
            "1.0.0-semantic-fallback",
            version,
        )

    def test_complex_metadata_expression_compiles_to_parameterized_sql(self):
        plan = _base_plan(
            criteria=[
                {
                    "criterion_id": "c1", "kind": "METADATA",
                    "field_scope": ["author"], "operator": "EQ",
                    "values": ["alice"], "evidence_requirement": "QUERY_RESULT",
                },
                {
                    "criterion_id": "c2", "kind": "METADATA",
                    "field_scope": ["author"], "operator": "EQ",
                    "values": ["bob"], "evidence_requirement": "QUERY_RESULT",
                },
                {
                    "criterion_id": "c3", "kind": "METADATA",
                    "field_scope": ["asset_date"], "operator": "GTE",
                    "values": ["2026-07-01"], "evidence_requirement": "QUERY_RESULT",
                },
            ],
            eligibility_expression={
                "node_type": "ALL",
                "children": [
                    {
                        "node_type": "ANY",
                        "children": [
                            {"node_type": "REF", "criterion_id": "c1"},
                            {"node_type": "REF", "criterion_id": "c2"},
                        ],
                    },
                    {"node_type": "REF", "criterion_id": "c3"},
                ],
            },
        )
        query_plan = AssetSearchDataQueryCompiler.compile(
            search_plan=plan, models=_catalog()
        )
        self.assertEqual("ALL", query_plan.filter_expression.node_type)
        self.assertEqual("ANY", query_plan.filter_expression.children[0].node_type)

        model = SemanticModelDefinition.model_validate(
            km_asset_definition(schema_name="KBOTUI_DEV")
        )
        compiled = compile_dialect_query(
            dialect="ORACLE", plan=query_plan, model=model,
            policy_max_limit=1000, scope_value=41,
        )
        self.assertIn(" OR ", compiled.sql)
        self.assertIn(" AND ", compiled.sql)
        self.assertNotIn("alice", compiled.sql)
        self.assertIn("alice", compiled.parameters)
        self.assertIn("bob", compiled.parameters)

    def test_semantic_branch_is_not_miscompiled_as_metadata_filter(self):
        plan = _base_plan(
            criteria=[
                {
                    "criterion_id": "c1", "kind": "METADATA",
                    "field_scope": ["product"], "operator": "EQ",
                    "values": ["OAC"], "evidence_requirement": "QUERY_RESULT",
                },
                {
                    "criterion_id": "c2", "kind": "SEMANTIC_CONCEPT",
                    "field_scope": ["CONTENT"], "operator": "RELATED_TO",
                    "values": ["金融欺诈"], "evidence_requirement": "CONTENT",
                },
            ],
            eligibility_expression={
                "node_type": "ANY",
                "children": [
                    {"node_type": "REF", "criterion_id": "c1"},
                    {"node_type": "REF", "criterion_id": "c2"},
                ],
            },
        )
        query_plan = AssetSearchDataQueryCompiler.compile(
            search_plan=plan, models=_catalog()
        )
        self.assertEqual((), query_plan.filters)
        self.assertIsNone(query_plan.filter_expression)
        self.assertEqual(1000, query_plan.limit)

    def test_semantic_count_is_normalized_to_reference_assets(self):
        normalized = AssetSearchPlanner.normalize_response(
            question="关于 OAC 的 Asset 总共有多少个",
            language="zh-CN",
            response={
                "operation": "COUNT",
                "target": "ASSET",
                "criteria": [{
                    "criterion_id": "c1", "kind": "SEMANTIC_CONCEPT",
                    "field_scope": ["CONTENT"], "operator": "RELATED_TO",
                    "values": ["OAC"], "evidence_requirement": "CONTENT",
                }],
                "eligibility_expression": {
                    "node_type": "REF", "criterion_id": "c1"
                },
                "measures": [{"name": "asset_count", "aggregation": "COUNT"}],
                "result_assets": {
                    "mode": "SUPPORTING", "target_count": 5,
                    "selection": "RECENT_WITHIN_RESULT",
                },
            },
        )
        plan = AssetSearchPlanV1.model_validate(normalized)
        self.assertEqual("LIST", plan.operation)
        self.assertEqual(("SEMANTIC_TOTAL_COUNT",), plan.unsupported_requests)
        self.assertFalse(plan.include_total_count)
        self.assertEqual(5, plan.display_limit)

    def test_asset_answer_semantic_scope_includes_searchable_metadata(self):
        normalized = AssetSearchPlanner.normalize_response(
            question="找个 financial fraud 的 Asset",
            language="zh-CN",
            response={
                "operation": "ANSWER",
                "target": "ASSET",
                "criteria": [{
                    "criterion_id": "c1",
                    "kind": "SEMANTIC_CONCEPT",
                    "field_scope": ["CONTENT"],
                    "operator": "RELATED_TO",
                    "values": ["financial fraud"],
                    "evidence_requirement": "CONTENT",
                }],
                "eligibility_expression": {
                    "node_type": "REF", "criterion_id": "c1"
                },
                "result_assets": {
                    "mode": "SUPPORTING",
                    "target_count": 5,
                    "selection": "EVIDENCE_COVERAGE_THEN_RECENT",
                },
            },
        )

        plan = AssetSearchPlanV1.model_validate(normalized)
        criterion = plan.criteria[0]
        self.assertEqual(
            ("CONTENT", "TITLE", "PRODUCT", "SOLUTION"),
            criterion.field_scope,
        )
        self.assertEqual(
            "METADATA_OR_CONTENT",
            criterion.evidence_requirement,
        )
        self.assertTrue(KnowledgeRetrievalSkill._criterion_matches(
            criterion,
            asset={"title": "Agentic Financial Fraud Detection"},
            bundle_id="b1",
            hit_sets={"c1": set()},
            semantic_terms=("financial fraud",),
        ))

    def test_list_normalizes_string_result_assets_without_crashing(self):
        normalized = AssetSearchPlanner.normalize_response(
            question="找几个关于 OAC 的 Asset，最好关于金融欺诈的案例",
            language="zh-CN",
            response={
                "operation": "LIST",
                "target": "ASSET",
                "criteria": [],
                "result_assets": "优先返回相关案例",
                "projection": "title",
            },
        )

        plan = AssetSearchPlanV1.model_validate(normalized)
        self.assertEqual("PRIMARY", plan.result_assets.mode)
        self.assertEqual("REQUESTED_ORDER", plan.result_assets.selection)
        self.assertEqual(10, plan.result_assets.target_count)
        self.assertEqual(
            (
                "asset_id", "title", "bundle_id", "bundle_revision_id",
                "product", "solution", "asset_date",
            ),
            plan.projection,
        )

    def test_list_normalizes_observed_planner_contract_dialect(self):
        normalized = AssetSearchPlanner.normalize_response(
            question="找几个关于OAC的asset，最好关于金融欺诈的案例",
            language="zh-CN",
            response={
                "operation": "LIST",
                "target": "ASSET",
                "criteria": [
                    {
                        "criterion_id": 0,
                        "kind": "SEMANTIC_CONCEPT",
                        "field_scope": ["CONTENT"],
                        "operator": "RELATED_TO",
                        "value": "OAC",
                        "evidence_requirement": "SUGGESTED",
                    },
                    {
                        "criterion_id": 1,
                        "kind": "SEMANTIC_CONCEPT",
                        "field_scope": ["CONTENT"],
                        "operator": "RELATED_TO",
                        "value": "金融欺诈案例",
                        "evidence_requirement": "SUGGESTED",
                    },
                ],
                "eligibility_expression": {
                    "node_type": "ALL",
                    "conditions": [
                        {"node_type": "REF", "ref": 0},
                        {"node_type": "REF", "ref": 1},
                    ],
                },
                "preferences": [{
                    "preference_id": 1,
                    "priority": 1,
                    "description": "优先展示关于金融欺诈案例的资产",
                    "criterion": {
                        "criterion_id": 1,
                        "kind": "SEMANTIC_CONCEPT",
                        "field_scope": ["CONTENT"],
                        "operator": "RELATED_TO",
                        "value": "金融欺诈案例",
                        "evidence_requirement": "SUGGESTED",
                    },
                }],
                "group_by": None,
                "evidence_policy": {
                    "citations": "REQUIRED",
                    "confidence": "SHOW",
                },
            },
        )

        plan = AssetSearchPlanV1.model_validate(normalized)
        self.assertEqual(1, len(plan.criteria))
        self.assertEqual(("OAC",), plan.criteria[0].values)
        self.assertEqual("c1", plan.criteria[0].criterion_id)
        self.assertEqual(
            ("CONTENT", "TITLE", "PRODUCT", "SOLUTION"),
            plan.criteria[0].field_scope,
        )
        self.assertEqual("c1", plan.eligibility_expression.criterion_id)
        self.assertEqual(1, len(plan.preferences))
        self.assertEqual(("金融欺诈案例",), plan.preferences[0].criterion.values)
        self.assertEqual("p1", plan.preferences[0].preference_id)
        self.assertEqual(
            "METADATA_OR_CONTENT",
            plan.preferences[0].evidence_requirement,
        )

    def test_condition_matrix_enforces_all_any_and_not(self):
        expression = AssetSearchPlanV1.model_validate({
            **_base_plan().model_dump(mode="json"),
            "criteria": [
                {
                    "criterion_id": "c1", "kind": "METADATA",
                    "field_scope": ["product"], "operator": "EQ",
                    "values": ["OAC"], "evidence_requirement": "QUERY_RESULT",
                },
                {
                    "criterion_id": "c2", "kind": "SEMANTIC_CONCEPT",
                    "field_scope": ["CONTENT"], "operator": "RELATED_TO",
                    "values": ["fraud"], "evidence_requirement": "CONTENT",
                },
                {
                    "criterion_id": "c3", "kind": "SEMANTIC_CONCEPT",
                    "field_scope": ["CONTENT"], "operator": "RELATED_TO",
                    "values": ["demo"], "occurrence": "MUST_NOT",
                    "evidence_requirement": "CONTENT",
                },
            ],
            "eligibility_expression": {
                "node_type": "ALL",
                "children": [
                    {
                        "node_type": "ANY",
                        "children": [
                            {"node_type": "REF", "criterion_id": "c1"},
                            {"node_type": "REF", "criterion_id": "c2"},
                        ],
                    },
                    {
                        "node_type": "NOT",
                        "child": {"node_type": "REF", "criterion_id": "c3"},
                    },
                ],
            },
        }).eligibility_expression
        self.assertTrue(KnowledgeRetrievalSkill._expression_matches(
            expression, {"c1": False, "c2": True, "c3": False}
        ))
        self.assertFalse(KnowledgeRetrievalSkill._expression_matches(
            expression, {"c1": True, "c2": False, "c3": True}
        ))

    def test_semantic_asset_scope_accepts_exact_product_metadata_support(self):
        criterion = _base_plan(
            criteria=[{
                "criterion_id": "c1",
                "kind": "SEMANTIC_CONCEPT",
                "field_scope": ["TITLE", "PRODUCT", "SOLUTION", "CONTENT"],
                "operator": "RELATED_TO",
                "values": ["OAC"],
                "evidence_requirement": "METADATA_OR_CONTENT",
            }],
            eligibility_expression={
                "node_type": "REF", "criterion_id": "c1"
            },
        ).criteria[0]

        self.assertTrue(KnowledgeRetrievalSkill._criterion_matches(
            criterion,
            asset={"product": "Business Analytics -> OAC (Oracle Analytics Cloud)"},
            bundle_id="b1",
            hit_sets={"c1": set()},
        ))

    def test_content_only_semantic_scope_rejects_metadata_only_support(self):
        criterion = _base_plan(
            criteria=[{
                "criterion_id": "c1",
                "kind": "SEMANTIC_CONCEPT",
                "field_scope": ["CONTENT"],
                "operator": "RELATED_TO",
                "values": ["OAC"],
                "evidence_requirement": "CONTENT",
            }],
            eligibility_expression={
                "node_type": "REF", "criterion_id": "c1"
            },
        ).criteria[0]

        self.assertFalse(KnowledgeRetrievalSkill._criterion_matches(
            criterion,
            asset={"product": "Business Analytics -> OAC (Oracle Analytics Cloud)"},
            bundle_id="b1",
            hit_sets={"c1": set()},
        ))

    async def test_cjk_semantic_criterion_adds_separate_english_queries(self):
        plan = _base_plan(
            query_text="找几个关于金融欺诈的 Asset",
            criteria=[{
                "criterion_id": "c1",
                "kind": "SEMANTIC_CONCEPT",
                "field_scope": ["TITLE", "PRODUCT", "SOLUTION", "CONTENT"],
                "operator": "RELATED_TO",
                "values": ["金融欺诈"],
                "evidence_requirement": "METADATA_OR_CONTENT",
            }],
            eligibility_expression={
                "node_type": "REF", "criterion_id": "c1"
            },
        )
        model_client = SimpleNamespace(get_llm_json=AsyncMock(return_value={
            "source_language": "zh-CN",
            "original_topic": "金融欺诈",
            "english_topics": ["financial fraud", "fraud detection"],
        }))
        prompt_resolver = SimpleNamespace(resolve=AsyncMock(
            return_value=SimpleNamespace(content="translate")
        ))
        skill = KnowledgeRetrievalSkill(
            knowledge_core_client=None,
            service_name="agent_runtime",
            model_client=model_client,
            prompt_resolver=prompt_resolver,
        )
        context = ExecutionContext(
            domain_id=20, agent_id=uuid7(), run_id=uuid7(), task_id=uuid7(),
            task_key="test", actor_id="user", request_id="request",
            trace_id="trace", original_input=plan.query_text,
            policy_snapshot={},
            config_snapshot={
                "agent": {"models": {"data_planner_llm": {
                    "served_model_name": "planner",
                }}},
            },
            input_artifacts=(),
        )

        queries, warnings = await skill._criterion_queries(
            context=context,
            plan=plan,
            criterion=plan.criteria[0],
        )

        self.assertEqual(
            ("金融欺诈", "financial fraud", "fraud detection"), queries
        )
        self.assertEqual((), warnings)

    def test_multilingual_evidence_groups_merge_by_bundle(self):
        groups = KnowledgeRetrievalSkill._merge_groups_by_criterion([
            ("c1", {}, [{
                "bundle_id": "b1",
                "items": [{"evidence_id": "e1"}],
            }]),
            ("c1", {}, [{
                "bundle_id": "b1",
                "items": [
                    {"evidence_id": "e1"},
                    {"evidence_id": "e2"},
                ],
            }]),
        ])

        self.assertEqual(1, len(groups["c1"]))
        self.assertEqual(
            ["e1", "e2"],
            [item["evidence_id"] for item in groups["c1"][0]["items"]],
        )

    def test_kc_evidence_must_contain_original_or_expanded_topic(self):
        criterion = _base_plan(
            criteria=[{
                "criterion_id": "c1",
                "kind": "SEMANTIC_CONCEPT",
                "field_scope": ["TITLE", "PRODUCT", "SOLUTION", "CONTENT"],
                "operator": "RELATED_TO",
                "values": ["金融欺诈"],
                "evidence_requirement": "METADATA_OR_CONTENT",
            }],
            eligibility_expression={
                "node_type": "REF", "criterion_id": "c1"
            },
        ).criteria[0]
        hit_sets = KnowledgeRetrievalSkill._qualified_evidence_hit_sets(
            {"c1": [{
                "bundle_id": "fraud-bundle",
                "items": [{"evidence": {
                    "content_text": "Financial fraud detection and risk insights",
                }}],
            }, {
                "bundle_id": "k3s-bundle",
                "items": [{"evidence": {
                    "content_text": "A K3s HA environment operations guide",
                }}],
            }]},
            criteria_by_key={"c1": criterion},
            queries_by_key={
                "c1": ("金融欺诈", "financial fraud", "fraud detection")
            },
        )

        self.assertEqual({"fraud-bundle"}, hit_sets["c1"])

    async def test_semantic_qualification_covers_scope_and_rejects_noise(self):
        collection_id = uuid7()
        fraud_bundle = uuid7()
        noise_bundle = uuid7()
        fraud_revision = uuid7()
        noise_revision = uuid7()
        plan = _base_plan(
            query_text="find financial fraud assets",
            criteria=[{
                "criterion_id": "c1",
                "kind": "SEMANTIC_CONCEPT",
                "field_scope": ["TITLE", "PRODUCT", "SOLUTION", "CONTENT"],
                "operator": "RELATED_TO",
                "values": ["financial fraud"],
                "evidence_requirement": "METADATA_OR_CONTENT",
            }],
            eligibility_expression={
                "node_type": "REF", "criterion_id": "c1"
            },
            display_limit=2,
            result_assets={
                "mode": "PRIMARY",
                "target_count": 2,
                "selection": "RECENT_RELEVANT",
            },
        )
        client = SimpleNamespace(retrieve_evidence=AsyncMock(return_value={
            "citations": [{
                "bundle_id": str(fraud_bundle),
                "items": [{"evidence": {
                    "document_id": str(uuid7()),
                    "content_text": "Financial fraud detection and risk insights",
                }}],
            }, {
                "bundle_id": str(noise_bundle),
                "items": [{"evidence": {
                    "document_id": str(uuid7()),
                    "content_text": "A K3s HA environment operations guide",
                }}],
            }],
            "warnings": [],
        }))
        skill = KnowledgeRetrievalSkill(
            knowledge_core_client=client,
            service_name="agent_runtime",
            model_client=None,
            prompt_resolver=None,
        )
        candidates = [{
            "collection_id": str(collection_id),
            "bundle_id": str(fraud_bundle),
            "bundle_revision_id": str(fraud_revision),
            "document_version_ids": [],
            "display_title": "Risk Analytics",
        }, {
            "collection_id": str(collection_id),
            "bundle_id": str(noise_bundle),
            "bundle_revision_id": str(noise_revision),
            "document_version_ids": [],
            "display_title": "K3s Operations",
        }]
        context = ExecutionContext(
            domain_id=20, agent_id=uuid7(), run_id=uuid7(), task_id=uuid7(),
            task_key="test", actor_id="user", request_id="request",
            trace_id="trace", original_input=plan.query_text,
            policy_snapshot={},
            config_snapshot={"agent": {"config": {}}},
            input_artifacts=(_artifact("DOCUMENT_SCOPE", {
                "assets": [{
                    "title": "Risk Analytics",
                    "bundle_id": str(fraud_bundle),
                }, {
                    "title": "K3s Operations",
                    "bundle_id": str(noise_bundle),
                }],
            }),),
        )

        evidence, eligible = await skill._retrieve_asset_plan_evidence(
            context=context,
            plan=plan,
            candidates=candidates,
            retrieval_config={"max_citations": 1, "context_limit": 0},
            coverage_mode="BALANCED",
        )

        self.assertEqual(
            [str(fraud_bundle)],
            [item["bundle_id"] for item in eligible],
        )
        self.assertEqual(1, len(evidence["citations"]))
        self.assertEqual(
            2,
            client.retrieve_evidence.await_args.kwargs["max_evidence"],
        )

    async def test_asset_without_attachments_gets_manifest_citation(self):
        collection_id = uuid7()
        bundle_id = uuid7()
        revision_id = uuid7()
        manifest_document_id = uuid7()
        manifest_version_id = uuid7()
        manifest_evidence_id = uuid7()
        client = SimpleNamespace(retrieve_evidence=AsyncMock(return_value={
            "citations": [{
                "collection_id": str(collection_id),
                "bundle_id": str(bundle_id),
                "bundle_revision_id": str(revision_id),
                "document_version_id": str(manifest_version_id),
                "items": [{
                    "final_role": "PRIMARY",
                    "evidence": {
                        "evidence_id": str(manifest_evidence_id),
                        "document_id": str(manifest_document_id),
                        "document_version_id": str(manifest_version_id),
                        "document_role": "MANIFEST",
                        "content_text": "Asset Title: Metadata-only Asset",
                        "locator": {},
                        "locator_schema_version": "document/v1",
                    },
                }],
            }],
            "warnings": [],
        }))
        skill = KnowledgeRetrievalSkill(
            knowledge_core_client=client,
            service_name="agent_runtime",
            model_client=None,
            prompt_resolver=None,
        )
        candidate = {
            "collection_id": str(collection_id),
            "bundle_id": str(bundle_id),
            "bundle_revision_id": str(revision_id),
            "document_version_ids": [],
            "display_title": "Metadata-only Asset",
        }
        context = ExecutionContext(
            domain_id=20,
            agent_id=uuid7(),
            run_id=uuid7(),
            task_id=uuid7(),
            task_key="test",
            actor_id="user",
            request_id="request",
            trace_id="trace",
            original_input="list assets",
            policy_snapshot={},
            config_snapshot={"agent": {"config": {}}},
            input_artifacts=(_artifact("DOCUMENT_SCOPE", {
                "assets": [{
                    "title": "Metadata-only Asset",
                    "bundle_id": str(bundle_id),
                }],
            }),),
        )

        evidence, eligible = await skill._retrieve_asset_plan_evidence(
            context=context,
            plan=_base_plan(result_assets={
                "mode": "PRIMARY",
                "target_count": 1,
                "selection": "REQUESTED_ORDER",
            }),
            candidates=[candidate],
            retrieval_config={"max_citations": 12, "context_limit": 4},
            coverage_mode="ANSWER",
        )

        self.assertEqual(1, len(eligible))
        self.assertEqual(1, len(evidence["citations"]))
        self.assertEqual(
            "MANIFEST",
            evidence["citations"][0]["items"][0]["evidence"]["document_role"],
        )
        citations = skill._map_citations(
            evidence["citations"], candidates=[candidate]
        )
        self.assertEqual(1, len(citations))
        self.assertEqual("MANIFEST", citations[0].document_role)
        self.assertEqual(manifest_version_id, citations[0].document_version_id)
        request = client.retrieve_evidence.await_args.kwargs
        self.assertEqual("Metadata-only Asset", request["query"])
        self.assertEqual([], request["candidates"][0]["document_version_ids"])

    def test_translated_semantic_term_supports_english_asset_title(self):
        criterion = _base_plan(
            criteria=[{
                "criterion_id": "c1",
                "kind": "SEMANTIC_CONCEPT",
                "field_scope": ["TITLE", "PRODUCT", "SOLUTION", "CONTENT"],
                "operator": "RELATED_TO",
                "values": ["金融欺诈"],
                "evidence_requirement": "METADATA_OR_CONTENT",
            }],
            eligibility_expression={
                "node_type": "REF", "criterion_id": "c1"
            },
        ).criteria[0]

        self.assertTrue(KnowledgeRetrievalSkill._criterion_matches(
            criterion,
            asset={"title": "Agentic Financial Fraud Detection"},
            bundle_id="b1",
            hit_sets={"c1": set()},
            semantic_terms=("金融欺诈", "financial fraud", "fraud detection"),
        ))

    def test_exact_phrase_requires_literal_content_support(self):
        criterion = _base_plan(
            criteria=[{
                "criterion_id": "c1", "kind": "EXACT_PHRASE",
                "field_scope": ["CONTENT"], "operator": "CONTAINS",
                "values": ["fraud pattern"], "evidence_requirement": "CONTENT",
            }],
            eligibility_expression={
                "node_type": "REF", "criterion_id": "c1"
            },
        ).criteria[0]
        groups = [{
            "bundle_id": "b1",
            "items": [{"evidence": {"content_text": "Fraud pattern detection"}}],
        }, {
            "bundle_id": "b2",
            "items": [{"evidence": {"content_text": "Fraud analytics"}}],
        }]
        filtered = KnowledgeRetrievalSkill._exact_phrase_groups(
            groups, criterion
        )
        self.assertEqual(["b1"], [item["bundle_id"] for item in filtered])

    async def test_exact_list_rejects_query_only_asset_references(self):
        plan = _base_plan(display_limit=2, result_assets={
            "mode": "PRIMARY", "target_count": 2,
            "selection": "REQUESTED_ORDER",
        })
        query = QueryResult(
            query_result_id=uuid7(), provider="SEMANTIC",
            columns=(),
            rows=(
                {"asset_id": "a1", "title": "Asset One", "product": "OAC"},
                {"asset_id": "a2", "title": "Asset Two", "product": "ADW"},
            ),
            row_count=2,
            provenance={"count_exact": True},
        )
        context = ExecutionContext(
            domain_id=20, agent_id=uuid7(), run_id=uuid7(), task_id=uuid7(),
            task_key="test", actor_id="user", request_id="request",
            trace_id="trace", original_input="列出 Asset",
            policy_snapshot={},
            config_snapshot={
                "agent": {},
                "route": {"asset_search_plan": plan.model_payload()},
            },
            input_artifacts=(_artifact("QUERY_RESULT", query.model_dump(mode="json")),),
        )
        composer = ResponseComposerSkill(
            model_client=SimpleNamespace(), prompt_resolver=SimpleNamespace()
        )
        result = await composer.execute(context)
        answer = result.artifact.payload["answer"]
        self.assertNotIn("Asset One", answer)
        self.assertNotIn("Asset Two", answer)
        self.assertEqual(
            "INSUFFICIENT_EVIDENCE",
            result.artifact.payload["status"],
        )
        self.assertEqual([], result.artifact.payload["used_citation_labels"])
        self.assertEqual([], result.artifact.payload["references"])

    async def test_exact_count_hides_samples_without_asset_citations(self):
        plan = _base_plan(
            operation="COUNT", display_limit=None,
            measures=[{"name": "asset_count", "aggregation": "COUNT"}],
            result_assets={
                "mode": "SUPPORTING", "target_count": 3,
                "selection": "RECENT_WITHIN_RESULT",
            },
        )
        supporting_id = uuid7()
        query = QueryResult(
            query_result_id=uuid7(), provider="SEMANTIC",
            columns=({"name": "asset_count"},),
            rows=({"asset_count": 41},), row_count=1,
            supporting_query_result_id=supporting_id,
            supporting_columns=({"name": "title"},),
            supporting_rows=(
                {"asset_id": "a1", "title": "Asset One"},
                {"asset_id": "a2", "title": "Asset Two"},
            ),
            provenance={"count_exact": True},
        )

        class Model:
            async def get_llm_json(self, **_):
                return {"answer": "共有 41 个可用 Asset。"}

        class Prompts:
            async def resolve(self, _):
                return SimpleNamespace(content="解释问数结果")

        context = ExecutionContext(
            domain_id=20, agent_id=uuid7(), run_id=uuid7(), task_id=uuid7(),
            task_key="test", actor_id="user", request_id="request",
            trace_id="trace", original_input="有多少个可用 Asset",
            policy_snapshot={},
            config_snapshot={
                "agent": {"models": {"composer_llm": {
                    "served_model_name": "composer"
                }}},
                "route": {"asset_search_plan": plan.model_payload()},
            },
            input_artifacts=(_artifact("QUERY_RESULT", query.model_dump(mode="json")),),
        )
        result = await ResponseComposerSkill(
            model_client=Model(), prompt_resolver=Prompts()
        ).execute(context)
        payload = result.artifact.payload
        self.assertIn("41", payload["answer"])
        self.assertNotIn("Asset One", payload["answer"])
        self.assertNotIn("[Q", payload["answer"])
        self.assertEqual([], payload["used_citation_labels"])
        self.assertEqual([], payload["references"])
        self.assertEqual(1, len(payload["query_results"]))
        self.assertIn("缺少必需", payload["warnings"][-1])


if __name__ == "__main__":
    unittest.main()
