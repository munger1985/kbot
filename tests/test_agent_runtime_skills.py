"""Agent Runtime 内置 Skill 的契约测试。"""

from datetime import datetime, timezone
import unittest

from agent_runtime.application import LeasedArtifact
from agent_runtime.runtime import ExecutionContext
from agent_runtime.specialists.document import KnowledgeRetrievalSkill
from agent_runtime.specialists.response_composer import ResponseComposerSkill
from agent_runtime.specialists.root import RouteType, RootAgentPlanner
from platform_core.identity import uuid7


class _KnowledgeCoreClient:
    def __init__(self):
        self.collection_id = uuid7()
        self.bundle_id = uuid7()
        self.revision_id = uuid7()
        self.document_id = uuid7()
        self.document_version_id = uuid7()
        self.parse_view_id = uuid7()
        self.evidence_id = uuid7()

    async def list_agent_bindings(self, **kwargs):
        return {
            "bindings": [
                {
                    "collection_id": str(self.collection_id),
                    "status": "ACTIVE",
                }
            ]
        }

    async def discover(self, **kwargs):
        return {
            "candidates": [
                {
                    "collection_id": str(self.collection_id),
                    "collection_key": "cases",
                    "bundle_id": str(self.bundle_id),
                    "bundle_revision_id": str(self.revision_id),
                    "display_title": "数据库优化案例",
                    "member_count": 1,
                    "matched_members": [],
                    "match_signals": ["TEXT", "VECTOR"],
                    "local_rank": 1,
                    "rrf_score": 0.03,
                    "candidate_scope": "SINGLE_MEMBER",
                }
            ]
        }

    async def retrieve_evidence(self, **kwargs):
        evidence = {
            "evidence_id": str(self.evidence_id),
            "document_id": str(self.document_id),
            "content_text": "该案例通过调整索引降低了查询延迟。",
            "locator": {"page": 3},
            "heading_path": ["优化方案"],
            "provenance": {"source_hash": "source-hash"},
        }
        return {
            "citations": [
                {
                    "citation_label": "C1",
                    "collection_id": str(self.collection_id),
                    "bundle_id": str(self.bundle_id),
                    "bundle_revision_id": str(self.revision_id),
                    "document_version_id": str(self.document_version_id),
                    "parse_view_id": str(self.parse_view_id),
                    "primary_evidence_ids": [str(self.evidence_id)],
                    "structural_context_ids": [],
                    "neighbor_evidence_ids": [],
                    "items": [
                        {
                            "item_label": "G1-A1",
                            "evidence": evidence,
                            "input_role": "ANCHOR",
                            "final_role": "PRIMARY",
                            "promoted_from_context": False,
                        }
                    ],
                }
            ]
        }


class _ModelClient:
    async def get_llm_json(self, **kwargs):
        return {
            "answer": "该案例通过调整索引降低了查询延迟。[C1]",
            "used_citation_labels": ["C1"],
        }


def _context(*, input_artifacts=()):
    run_id = uuid7()
    return ExecutionContext(
        app_id=1,
        domain_id=10,
        agent_id=uuid7(),
        run_id=run_id,
        task_id=uuid7(),
        task_key="knowledge_retrieval",
        actor_id="user-1",
        request_id="request-1",
        trace_id="trace-1",
        original_input="这个案例如何降低查询延迟？",
        config_snapshot={
            "agent": {
                "composer_model_name": "composer-model",
                "instruction": "准确回答。",
                "config": {},
            },
            "retrieval": {
                "collection_ids": [],
                "security_level": 2,
            },
        },
        deadline_at=datetime.now(timezone.utc),
        input_artifacts=input_artifacts,
    )


class AgentRuntimeSkillTest(unittest.IsolatedAsyncioTestCase):
    def test_root_planner_builds_fixed_document_dag(self):
        planner = RootAgentPlanner()
        decision = planner.decide(
            agent_snapshot={
                "enabled_capabilities": ["document"],
                "config": {},
            }
        )
        plan = planner.build_plan(
            objective="总结文档", decision=decision
        )

        self.assertEqual(decision.route_type, RouteType.DOCUMENT)
        self.assertEqual(plan.final_task_key, "response_compose")
        self.assertEqual(
            plan.tasks[1].depends_on, ("knowledge_retrieval",)
        )

    def test_root_planner_builds_aiops_delegation_dag(self):
        planner = RootAgentPlanner()
        decision = planner.decide(
            agent_snapshot={
                "enabled_capabilities": ["aiops"],
                "config": {
                    "aiops_agent_id": str(uuid7()),
                    "aiops_target_id": str(uuid7()),
                },
            }
        )
        plan = planner.build_plan(
            objective="分析数据库性能下降原因",
            decision=decision,
        )

        self.assertEqual(decision.route_type, RouteType.AIOPS)
        self.assertEqual(plan.tasks[0].execution_kind, "DELEGATION")
        self.assertEqual(plan.tasks[0].delegate_service, "aiops_agent")
        self.assertEqual(
            plan.tasks[1].input_refs,
            ("task_output:aiops_diagnosis",),
        )

    def test_root_planner_rejects_aiops_without_frozen_target(self):
        decision = RootAgentPlanner().decide(
            agent_snapshot={
                "enabled_capabilities": ["aiops"],
                "config": {},
            }
        )

        self.assertEqual(decision.route_type, RouteType.CLARIFY)

    async def test_document_skill_builds_document_level_citation(self):
        client = _KnowledgeCoreClient()
        result = await KnowledgeRetrievalSkill(
            knowledge_core_client=client,
            service_name="agent-worker",
        ).execute(_context())

        payload = result.artifact.payload
        citation = payload["citation_pack"]["citations"][0]
        self.assertEqual(result.artifact.artifact_type, "CITATION_PACK")
        self.assertEqual(citation["title"], "数据库优化案例")
        self.assertEqual(citation["document_id"], str(client.document_id))
        self.assertEqual(citation["locator"], {"page": 3})

    async def test_composer_returns_only_actually_used_references(self):
        client = _KnowledgeCoreClient()
        retrieval = await KnowledgeRetrievalSkill(
            knowledge_core_client=client,
            service_name="agent-worker",
        ).execute(_context())
        artifact = LeasedArtifact(
            artifact_id=uuid7(),
            task_id=uuid7(),
            artifact_type=retrieval.artifact.artifact_type,
            schema_version=retrieval.artifact.schema_version,
            producer="knowledge-retrieval",
            producer_version="1.0.0",
            payload=retrieval.artifact.payload,
            content_hash="hash",
            provenance=retrieval.artifact.provenance,
            security_level=2,
        )
        result = await ResponseComposerSkill(
            model_client=_ModelClient()
        ).execute(_context(input_artifacts=(artifact,)))

        payload = result.artifact.payload
        self.assertEqual(payload["used_citation_labels"], ["C1"])
        self.assertEqual(len(payload["references"]), 1)
        self.assertEqual(
            payload["references"][0]["document_id"],
            str(client.document_id),
        )

    async def test_composer_projects_safe_aiops_result(self):
        delegation_id = uuid7()
        ops_run_id = uuid7()
        diagnosis_artifact_id = uuid7()
        artifact = LeasedArtifact(
            artifact_id=uuid7(),
            task_id=uuid7(),
            artifact_type="DELEGATED_AIOPS_RESULT",
            schema_version="DELEGATED_AIOPS_RESULT.v1",
            producer="aiops-agent",
            producer_version="1",
            payload={
                "delegation_id": str(delegation_id),
                "ops_run_id": str(ops_run_id),
                "status": "COMPLETED",
                "safe_summary": "发现慢 SQL 与缺失索引相关。",
                "diagnosis": {
                    "root_cause_grade": "CONFIRMED",
                    "artifact": {
                        "artifact_id": str(diagnosis_artifact_id),
                        "content_hash": "a" * 64,
                    },
                },
            },
            content_hash="result-hash",
            provenance={},
            security_level=2,
        )

        result = await ResponseComposerSkill(
            model_client=_ModelClient()
        ).execute(_context(input_artifacts=(artifact,)))

        payload = result.artifact.payload
        self.assertEqual(payload["used_citation_labels"], ["O1"])
        self.assertIn("[O1]", payload["answer"])
        self.assertEqual(
            payload["references"][0]["ops_run_id"], str(ops_run_id)
        )


if __name__ == "__main__":
    unittest.main()
