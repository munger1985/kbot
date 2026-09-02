"""AI DBA 输入理解与调查规划测试。"""

from __future__ import annotations

import unittest

from aiops_agent.application.investigation import InvestigationReasoner
from aiops_agent.contracts.diagnosis import ModelInvocationReceipt
from aiops_agent.ports.model import StructuredModelResult
from platform_core.contracts.aiops import (
    CompactPlanningOutput,
    InvestigationPlanningOutput,
)


TARGET_CONTEXT = {
    "target_id": "target-1",
    "display_name": "订单生产库",
    "db_type": "ORACLE",
    "configured_version": "19c",
    "environment": "PROD",
    "db_role": "PRIMARY",
    "status": "ENABLED",
    "connectivity_status": "CONNECTED",
    "selection_status": "BOUND",
}
PROMPT_SNAPSHOT = {"frozen": {"prompt_version_id": "prompt-version-1"}}


class _Prompt:
    content = "不得再把“目标是哪一个”列为未知项。"

    @staticmethod
    def ref() -> dict[str, str]:
        return {
            "prompt_id": "aiops_agent.test",
            "prompt_version": "1.0.0",
            "prompt_sha256": "d" * 64,
            "prompt_version_id": "prompt-version-1",
            "prompt_source": "DATABASE",
        }


class _Prompts:
    async def resolve(self, *_args, **_kwargs):
        return _Prompt()

    async def snapshot(self, *_args, **_kwargs):
        return PROMPT_SNAPSHOT


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
            "objectives": ["DIAGNOSE"],
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
        self.calls = []

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
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
    async def test_compact_planner_uses_small_routing_contract(self) -> None:
        payload = {
            "planning_mode": "READ_ONLY_LOOKUP",
            "problem_statement": "列出 TCC Schema 下的表",
            "success_criteria": ["返回当前表清单"],
            "selected_tool_ids": ["db.oracle.readonly_query"],
            "selected_playbook_ids": [],
            "actions": [
                {
                    "action_id": "a1",
                    "question": "TCC Schema 当前有哪些表？",
                    "tool_id": "db.oracle.readonly_query",
                    "input": {
                        "sql": "SELECT table_name FROM all_tables WHERE owner = :owner",
                        "parameters": {"owner": "TCC"},
                    },
                    "expected_evidence_kind": "DATABASE_OBJECTS",
                    "measurement_semantics": "CURRENT_ACTIVITY",
                }
            ],
            "public_reasoning_summary": "问题范围明确，将直接执行只读目录查询",
        }
        model = _Model(payload)
        reasoner = InvestigationReasoner(model, _Prompts())

        result = await reasoner.plan_compact(
            question="数据库用户 TCC 下有哪些表？",
            target_context=TARGET_CONTEXT,
            prompt_snapshot=PROMPT_SNAPSHOT,
            tool_cards=(
                {
                    "tool_id": "db.oracle.readonly_query",
                    "description": "执行只读 Oracle 查询",
                },
            ),
            available_playbooks=(),
            model_snapshot={"technical_name": "planner"},
            deadline=None,
            idempotency_key="turn-compact-1",
        )

        self.assertIsInstance(result.output, CompactPlanningOutput)
        self.assertEqual("aiops.compact-planning", model.calls[0]["purpose"])
        self.assertEqual(1, len(result.output.actions))

    async def test_compact_planner_rejects_unknown_selected_tool(self) -> None:
        payload = {
            "planning_mode": "FULL_INVESTIGATION",
            "problem_statement": "分析数据库性能问题",
            "success_criteria": ["形成证据结论"],
            "selected_tool_ids": ["shell.exec"],
            "actions": [],
            "public_reasoning_summary": "需要进行完整调查",
        }
        reasoner = InvestigationReasoner(_Model(payload), _Prompts())

        with self.assertRaisesRegex(ValueError, "未注册工具"):
            await reasoner.plan_compact(
                question="分析数据库性能问题",
                target_context=TARGET_CONTEXT,
                prompt_snapshot=PROMPT_SNAPSHOT,
                tool_cards=(),
                available_playbooks=(),
                model_snapshot={},
                deadline=None,
                idempotency_key="turn-compact-2",
            )

    async def test_compact_controlled_action_requires_readonly_precheck(self):
        with self.assertRaisesRegex(ValueError, "只读核验动作"):
            CompactPlanningOutput.model_validate(
                {
                    "planning_mode": "CONTROLLED_ACTION",
                    "problem_statement": "收集表统计信息",
                    "success_criteria": ["生成审批提案"],
                    "selected_tool_ids": ["db.table.statistics"],
                    "actions": [],
                    "public_reasoning_summary": "先核验后审批",
                }
            )

    async def test_user_alert_log_can_be_answered_without_external_tool(self) -> None:
        model = _Model(_output())
        reasoner = InvestigationReasoner(model, _Prompts())

        result = await reasoner.plan(
            content=({"content_type": "LOG", "text": "ORA-27157"},),
            conversation_context=(),
            target_context=TARGET_CONTEXT,
            prompt_snapshot=PROMPT_SNAPSHOT,
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
        request = model.calls[0]
        self.assertEqual(
            TARGET_CONTEXT,
            request["input_payload"]["target_context"],
        )
        self.assertIn(
            "不得再把“目标是哪一个”",
            request["prompt_ref"]["content"],
        )

    async def test_plan_rejects_tool_outside_current_catalog(self) -> None:
        reasoner = InvestigationReasoner(
            _Model(_output(tool_id="shell.exec")), _Prompts()
        )

        with self.assertRaisesRegex(ValueError, "未注册工具"):
            await reasoner.plan(
                content=({"content_type": "TEXT", "text": "检查数据库"},),
                conversation_context=(),
                target_context=TARGET_CONTEXT,
                prompt_snapshot=PROMPT_SNAPSHOT,
                available_tools=(),
                available_playbooks=(),
                model_snapshot={},
                deadline=None,
                idempotency_key="turn-2",
            )

    async def test_replan_rejects_identical_tool_call_without_progress(self) -> None:
        payload = _output(tool_id="db.instance.identity")
        payload["plan"]["revision_no"] = 2
        reasoner = InvestigationReasoner(_Model(payload), _Prompts())

        with self.assertRaisesRegex(ValueError, "不得原样重复"):
            await reasoner.replan(
                content=({"content_type": "TEXT", "text": "检查数据库"},),
                conversation_context=(),
                target_context=TARGET_CONTEXT,
                prompt_snapshot=PROMPT_SNAPSHOT,
                source_run_evidence=None,
                task_frame=payload["task_frame"],
                prior_plan={
                    "actions": [
                        {"tool_id": "db.instance.identity", "input": {}}
                    ]
                },
                assessment={"status": "NEEDS_EVIDENCE"},
                available_tools=(
                    {"tool_id": "db.instance.identity", "version": "1.0.0"},
                ),
                available_playbooks=(),
                model_snapshot={},
                deadline=None,
                idempotency_key="turn-2-replan",
                revision_no=2,
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

    def test_plan_rejects_unknown_measurement_semantics(self) -> None:
        payload = _output(tool_id="db.instance.identity")
        payload["plan"]["actions"][0][
            "measurement_semantics"
        ] = "point-in-time"

        with self.assertRaisesRegex(ValueError, "measurement_semantics"):
            InvestigationPlanningOutput.model_validate(payload)

    async def test_policy_repair_receives_allowlist_and_rejection_detail(
        self,
    ) -> None:
        rejected_payload = _output(tool_id="db.oracle.readonly_query")
        rejected_payload["plan"]["actions"][0]["input"] = {
            "sql": "SELECT custom_function(sid) AS result FROM v$session",
            "parameters": {},
        }
        rejected = InvestigationPlanningOutput.model_validate(
            rejected_payload
        )
        repaired_payload = _output(tool_id="db.instance.identity")
        model = _Model(repaired_payload)
        reasoner = InvestigationReasoner(model, _Prompts())
        tools = (
            {
                "tool_id": "db.instance.identity",
                "version": "1.0.0",
            },
            {
                "tool_id": "db.oracle.readonly_query",
                "version": "1.0.0",
                "policy": {"allowed_functions": ["COUNT", "ROUND"]},
            },
        )

        result = await reasoner.repair_policy_invalid_plan(
            content=({"content_type": "TEXT", "text": "检查数据库"},),
            conversation_context=(),
            target_context=TARGET_CONTEXT,
            prompt_snapshot=PROMPT_SNAPSHOT,
            source_run_evidence=None,
            invalid_output=rejected,
            validation_error=(
                "动态查询未通过策略：DYNAMIC_SQL_FUNCTION_FORBIDDEN；"
                "动态 SQL 函数不在允许清单：CUSTOM_FUNCTION"
            ),
            available_tools=tools,
            available_playbooks=(),
            model_snapshot={},
            deadline=None,
            idempotency_key="turn-3-policy-repair",
        )

        self.assertEqual(
            "db.instance.identity", result.output.plan.actions[0].tool_id
        )
        request = model.calls[0]
        self.assertEqual(
            "aiops.investigation-policy-repair", request["purpose"]
        )
        self.assertIn(
            "CUSTOM_FUNCTION",
            request["input_payload"]["validation_error"],
        )
        self.assertEqual(
            ["COUNT", "ROUND"],
            request["input_payload"]["available_tools"][1]["policy"][
                "allowed_functions"
            ],
        )
        self.assertEqual(
            TARGET_CONTEXT,
            request["input_payload"]["target_context"],
        )

    async def test_policy_repair_cannot_rewrite_task_frame(self) -> None:
        rejected_payload = _output(tool_id="db.oracle.readonly_query")
        rejected_payload["plan"]["actions"][0]["input"] = {
            "sql": "SELECT custom_function(sid) AS result FROM v$session",
            "parameters": {},
        }
        rejected = InvestigationPlanningOutput.model_validate(
            rejected_payload
        )
        repaired_payload = _output(tool_id="db.instance.identity")
        repaired_payload["task_frame"]["problem_statement"] = "改写后的问题"
        reasoner = InvestigationReasoner(
            _Model(repaired_payload), _Prompts()
        )

        with self.assertRaisesRegex(ValueError, "不得改变任务框架"):
            await reasoner.repair_policy_invalid_plan(
                content=({"content_type": "TEXT", "text": "检查数据库"},),
                conversation_context=(),
                target_context=TARGET_CONTEXT,
                prompt_snapshot=PROMPT_SNAPSHOT,
                source_run_evidence=None,
                invalid_output=rejected,
                validation_error="DYNAMIC_SQL_FUNCTION_FORBIDDEN",
                available_tools=(
                    {"tool_id": "db.instance.identity", "version": "1.0.0"},
                ),
                available_playbooks=(),
                model_snapshot={},
                deadline=None,
                idempotency_key="turn-4-policy-repair",
            )


if __name__ == "__main__":
    unittest.main()
