"""KM Asset 统一搜索计划、编译和结果展示的聚焦测试。"""

import unittest
from types import SimpleNamespace

from agent_runtime.application.commands import LeasedArtifact
from agent_runtime.runtime import ExecutionContext
from agent_runtime.specialists.asset_search import (
    AssetSearchDataQueryCompiler,
    AssetSearchPlanner,
)
from agent_runtime.specialists.data_query import QueryResult
from agent_runtime.specialists.document import KnowledgeRetrievalSkill
from agent_runtime.specialists.response_composer import ResponseComposerSkill
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

    async def test_exact_list_renders_every_asset_with_query_reference(self):
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
        self.assertIn("Asset One", answer)
        self.assertIn("Asset Two", answer)
        self.assertEqual(3, answer.count("[Q1]"))
        self.assertEqual(["Q1"], result.artifact.payload["used_citation_labels"])

    async def test_exact_count_uses_separate_query_reference_for_samples(self):
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
        self.assertIn("Asset One", payload["answer"])
        self.assertIn("[Q2]", payload["answer"])
        self.assertEqual(["Q1", "Q2"], payload["used_citation_labels"])
        self.assertEqual(2, len(payload["query_results"]))
        self.assertEqual(str(supporting_id), payload["query_results"][1]["query_result_id"])


if __name__ == "__main__":
    unittest.main()
