"""AI DBA 输入理解与调查规划测试。"""

from __future__ import annotations

import unittest

from aiops_agent.investigation import InvestigationReasoner
from aiops_agent.contracts.diagnosis import ModelInvocationReceipt
from aiops_agent.ports.model import StructuredModelResult
from platform_core.contracts.aiops import InvestigationPlanningOutput


def _output(*, tool_id: str | None = None) -> dict:
    actions = []
    if tool_id is not None:
        actions.append(
            {
                "action_id": "a1",
                "question": "数据库当前是否仍然可连接？",
                "tool_id": tool_id,
                "input": {},
                "expected_evidence_kind": "DATABASE_STATUS",
                "measurement_semantics": "CURRENT_ACTIVITY",
            }
        )
    return {
        "input_envelope": {
            "materials": [
                {
                    "item_no": 1,
                    "material_kind": "ORACLE_ALERT_LOG",
                    "summary": "用户提供了实例异常退出前后的 Alert Log",
                    "key_facts": ["出现 ORA-27157 和 semop status 43"],
                    "confidence": 0.99,
                    "contains_user_evidence": True,
                }
            ],
            "inferred_question": "分析实例退出原因",
            "supplied_evidence_summary": ["Oracle Alert Log"],
        },
        "task_frame": {
            "objective": "DIAGNOSE",
            "problem_statement": "判断 Oracle 实例退出的直接原因和后续核查方向",
            "known_facts": ["操作系统信号量对象被移除"],
            "unknowns": ["谁删除了 IPC 对象"],
            "constraints": ["Target 当前离线"],
            "success_criteria": ["解释错误链并给出可验证的下一步"],
            "requires_change": False,
        },
        "plan": {
            "revision_no": 1,
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "statement": "宿主机 IPC 对象被外部清理",
                    "rationale": "多个后台进程同时报告 Identifier removed",
                    "confidence": 0.9,
                }
            ],
            "actions": actions,
            "answer_if_no_more_evidence": True,
        },
        "suggested_playbook_ids": [],
    }


class _Model:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def generate_structured(self, **kwargs):
        output = kwargs["output_model"].model_validate(self.payload)
        return StructuredModelResult(
            output=output,
            receipt=ModelInvocationReceipt(
                purpose="test",
                schema_id="test.v1",
                model_technical_name="test",
                model_revision="1",
                prompt_id="test",
                prompt_version="1",
                prompt_sha256="a" * 64,
                input_sha256="b" * 64,
                output_sha256="c" * 64,
                duration_ms=1,
            ),
        )


class InvestigationReasonerTest(unittest.IsolatedAsyncioTestCase):
    async def test_user_alert_log_can_be_answered_without_external_tool(self) -> None:
        reasoner = InvestigationReasoner(_Model(_output()))

        result = await reasoner.plan(
            content=({"content_type": "LOG", "text": "ORA-27157"},),
            conversation_context=(),
            available_tools=(),
            available_playbooks=(),
            model_snapshot={},
            deadline=None,
            idempotency_key="turn-1",
        )

        self.assertIsInstance(result.output, InvestigationPlanningOutput)
        self.assertTrue(
            result.output.input_envelope.materials[0].contains_user_evidence
        )
        self.assertEqual((), result.output.plan.actions)

    async def test_plan_rejects_tool_outside_current_catalog(self) -> None:
        reasoner = InvestigationReasoner(_Model(_output(tool_id="shell.exec")))

        with self.assertRaisesRegex(ValueError, "未注册工具"):
            await reasoner.plan(
                content=({"content_type": "TEXT", "text": "检查数据库"},),
                conversation_context=(),
                available_tools=(),
                available_playbooks=(),
                model_snapshot={},
                deadline=None,
                idempotency_key="turn-2",
            )

    def test_plan_rejects_cyclic_tool_dependencies(self) -> None:
        payload = _output(tool_id="db.instance.identity")
        payload["plan"]["actions"] = [
            {
                "action_id": "a1",
                "question": "先检查实例",
                "tool_id": "db.instance.identity",
                "expected_evidence_kind": "DATABASE_STATUS",
                "measurement_semantics": "CURRENT_ACTIVITY",
                "depends_on": ["a2"],
            },
            {
                "action_id": "a2",
                "question": "再确认服务",
                "tool_id": "db.instance.identity",
                "expected_evidence_kind": "DATABASE_STATUS",
                "measurement_semantics": "CURRENT_ACTIVITY",
                "depends_on": ["a1"],
            },
        ]

        with self.assertRaisesRegex(ValueError, "不能包含环"):
            InvestigationPlanningOutput.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
