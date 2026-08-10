"""Conversation API 与长期记忆安全规则测试。"""

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from agent_runtime.api import conversation_router, memory_router
from agent_runtime.application.memory import (
    ConversationSnapshotOutput,
    MemoryCandidate,
    MemoryCandidateBatch,
    MemoryConsolidationWorker,
    MemoryJobLease,
)
from agent_runtime.application import ConversationService
from agent_runtime.application.conversations import MemoryRecallQuery
from platform_core.contracts import (
    AuthContext,
    ConversationTurnReceipt,
    ConversationView,
    PrincipalKind,
)
from platform_core.identity import uuid7
from platform_core.prompts import ResolvedPrompt


class AgentConversationApiTest(unittest.TestCase):
    def setUp(self):
        self.service = AsyncMock()
        self.agent_id = uuid7()
        self.conversation_id = uuid7()
        self.turn_id = uuid7()
        self.run_id = uuid7()
        now = datetime.now(timezone.utc)
        self.service.create.return_value = ConversationView(
            conversation_id=self.conversation_id,
            agent_id=self.agent_id,
            title="文档讨论",
            status="ACTIVE",
            row_version=1,
            last_turn_sequence=0,
            last_active_at=now,
            created_at=now,
            retention_policy="DEFAULT",
            purge_after=None,
        )
        self.service.create_turn.return_value = ConversationTurnReceipt(
            conversation_id=self.conversation_id,
            turn_id=self.turn_id,
            turn_sequence=1,
            turn_status="RUNNING",
            run_id=self.run_id,
            run_status="RUNNING",
            event_cursor=2,
            events_url=(
                f"/api/v1/apps/knowledge-retrieval/runs/"
                f"{self.run_id}/events"
            ),
        )
        app = FastAPI()
        app.state.conversation_service = self.service
        app.state.agent_runtime_budget = {"max_tasks": 16}

        @app.middleware("http")
        async def inject_identity(request: Request, call_next):
            request.state.auth_context = AuthContext(
                principal_kind=PrincipalKind.PORTAL,
                client_id="portal",
                api_key_id="portal-key",
                domain_id="20",
                asserted_user_id="user-1",
                request_id="request-1",
                trace_id="trace-1",
            )
            return await call_next(request)

        app.include_router(conversation_router)
        app.include_router(memory_router)
        self.client = TestClient(app)

    def _execution_spec(self):
        return {
            "schema_version": "1.0",
            "owner_app_id": "knowledge_retrieval",
            "domain_id": 20,
            "consumer_agent_id": str(self.agent_id),
            "consumer_agent_version_id": str(uuid7()),
            "agent_kind": "KNOWLEDGE_RETRIEVAL",
            "display_name": "文档助手",
            "enabled_capabilities": ["document"],
            "models": {"composer_llm": str(uuid7())},
        }

    def test_create_conversation_uses_trusted_identity(self):
        response = self.client.post(
            "/internal/v1/conversations",
            json={
                "agent_id": str(self.agent_id),
                "execution_spec": self._execution_spec(),
                "title": "文档讨论",
            },
        )

        self.assertEqual(response.status_code, 201)
        kwargs = self.service.create.await_args.kwargs
        self.assertEqual(kwargs["domain_id"], 20)
        self.assertEqual(kwargs["actor_id"], "user-1")

    def test_create_turn_forwards_idempotency_and_context_scope(self):
        response = self.client.post(
            f"/internal/v1/conversations/{self.conversation_id}/turns",
            headers={"Idempotency-Key": "turn-1"},
            json={
                "input": "它有什么优势？",
                "expected_conversation_version": 1,
            },
        )

        self.assertEqual(response.status_code, 202)
        kwargs = self.service.create_turn.await_args.kwargs
        self.assertEqual(kwargs["idempotency_key"], "turn-1")
        self.assertEqual(kwargs["actor_id"], "user-1")
        self.assertEqual(kwargs["domain_id"], 20)


class MemorySafetyTest(unittest.TestCase):
    def test_snapshot_accepts_named_entities(self):
        snapshot = ConversationSnapshotOutput.model_validate(
            {
                "active_topic": "AIOps Agent 配置",
                "user_goal": "完成 Agent 与 Target 绑定",
                "entities": [
                    "aiops_agent_id",
                    "aiops_target_id",
                    {"name": "Target", "kind": "RESOURCE"},
                ],
                "corrections": [],
                "unresolved_questions": [],
            }
        )

        self.assertEqual(snapshot.entities[0], "aiops_agent_id")

    def test_sensitive_candidate_is_rejected_deterministically(self):
        batch = MemoryCandidateBatch.model_validate(
            {
                "candidates": [
                    {
                        "memory_type": "USER_PREFERENCE",
                        "canonical_key": "response.language",
                        "value": {"language": "zh-CN"},
                        "search_text": "偏好中文回答",
                        "confidence": 0.9,
                        "salience": 0.8,
                    },
                    {
                        "memory_type": "USER_FACT",
                        "canonical_key": "database.password",
                        "value": {"password": "do-not-store"},
                        "search_text": "数据库密码",
                        "confidence": 1,
                        "salience": 1,
                    },
                ]
            }
        )

        accepted = MemoryConsolidationWorker._safe_candidates(batch)

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].canonical_key, "response.language")

    def test_forget_keys_reject_wildcard(self):
        with self.assertRaises(ValueError):
            MemoryConsolidationWorker._safe_forget_keys(
                ("response.*",)
            )

    def test_retention_policy_has_deterministic_deadline(self):
        now = datetime.now(timezone.utc)
        deadline = ConversationService._retention_deadline(
            "DAYS_30", now=now
        )

        self.assertEqual((deadline - now).days, 30)
        self.assertIsNone(
            ConversationService._retention_deadline(
                "KEEP_FOREVER", now=now
            )
        )

    def test_hybrid_recall_prefers_matching_vector(self):
        profile_id = uuid7()
        lexical = SimpleNamespace(
            canonical_key="database.oracle",
            search_text="oracle",
            value_json={},
            salience=0.9,
            updated_at=None,
            index_profile_id=None,
            embedding=None,
        )
        semantic = SimpleNamespace(
            canonical_key="user.preference",
            search_text="用户偏好",
            value_json={},
            salience=0.5,
            updated_at=None,
            index_profile_id=profile_id,
            embedding=[1.0, 0.0],
        )

        selected = ConversationService._select_memories(
            "oracle performance",
            [lexical, semantic],
            limit=1,
            recall_query=MemoryRecallQuery(
                index_profile_id=profile_id,
                embedding=(1.0, 0.0),
            ),
        )

        self.assertIs(selected[0], semantic)


class _ConflictModel:
    def __init__(self):
        self.calls = 0

    async def get_llm_json(self, **kwargs):
        self.calls += 1
        return {
            "action": "SUPERSEDE",
            "reason": "用户明确表达了新的语言偏好",
        }


class _ConflictPromptResolver:
    async def resolve(self, prompt_key):
        return ResolvedPrompt(
            prompt_key=prompt_key,
            version="1.1.0",
            sha256="a" * 64,
            content="candidate=${candidate}\nexisting=${existing_memory}",
            input_variables=("candidate", "existing_memory"),
            output_schema="MemoryConflictDecision.v1",
            source="TEST",
        )


class _SnapshotRepairModel:
    def __init__(self):
        self.calls = 0

    async def get_llm_json(self, **kwargs):
        self.calls += 1
        return {
            "active_topic": "基础算术",
            "user_goal": "求解5+5的值",
            "entities": [],
            "corrections": [],
            "unresolved_questions": [],
        }


class MemoryConflictTest(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_shape_error_gets_one_model_repair(self):
        model = _SnapshotRepairModel()
        worker = MemoryConsolidationWorker(
            uow_factory=None,
            model_client=model,
            prompt_resolver=None,
            worker_id="memory-test",
            poll_interval_seconds=1,
        )

        snapshot = await worker._validate_model_output(
            model_type=ConversationSnapshotOutput,
            response={
                "active_topic": "基础算术",
                "user_goal": ["求解5+5的值"],
                "entities": [],
                "corrections": [],
                "unresolved_questions": [],
            },
            model_name="test-model",
            prompt_version="1.0.1",
            rendered_prompt="snapshot prompt",
            output_name="会话摘要",
            correction_instruction=(
                "active_topic 和 user_goal 必须是字符串或 null。"
            ),
        )

        self.assertEqual(model.calls, 1)
        self.assertEqual(snapshot.user_goal, "求解5+5的值")

    async def test_only_same_key_different_value_calls_conflict_model(self):
        model = _ConflictModel()
        worker = MemoryConsolidationWorker(
            uow_factory=None,
            model_client=model,
            prompt_resolver=_ConflictPromptResolver(),
            worker_id="memory-test",
            poll_interval_seconds=1,
        )
        existing_id = uuid7()
        lease = MemoryJobLease(
            job_id=uuid7(),
            lease_token=uuid7(),
            attempt_count=1,
            max_attempts=3,
            conversation_id=uuid7(),
            turn_id=uuid7(),
            turn_sequence=1,
            domain_id=2,
            actor_id="user-1",
            agent_id=uuid7(),
            user_item_id=uuid7(),
            user_message="以后请用英文回答",
            assistant_message={"text": "好的"},
            previous_summary={},
            existing_memories=(
                {
                    "memory_id": str(existing_id),
                    "memory_type": "USER_PREFERENCE",
                    "canonical_key": "response.language",
                    "value": {"language": "zh-CN"},
                },
            ),
        )
        decisions = await worker._decide_candidates(
            lease,
            candidates=(
                MemoryCandidate(
                    memory_type="USER_PREFERENCE",
                    canonical_key="response.language",
                    value={"language": "en-US"},
                    search_text="偏好英文回答",
                    confidence=1,
                    salience=0.9,
                ),
                MemoryCandidate(
                    memory_type="USER_FACT",
                    canonical_key="database.engine",
                    value={"engine": "Oracle"},
                    search_text="使用 Oracle",
                    confidence=0.9,
                    salience=0.7,
                ),
            ),
            model_name="test-model",
        )

        self.assertEqual(model.calls, 1)
        self.assertEqual(
            [decision.action for decision in decisions],
            ["SUPERSEDE", "ADD"],
        )
        self.assertEqual(decisions[0].existing_memory_id, existing_id)


if __name__ == "__main__":
    unittest.main()
