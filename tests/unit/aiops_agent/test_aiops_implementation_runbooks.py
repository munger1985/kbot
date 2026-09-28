from pathlib import Path
import unittest

from aiops_agent.application.implementation import (
    compile_implementation_runbook,
)
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.contracts.implementation import (
    RunbookApplicability,
    RunbookStatus,
)
from aiops_agent.contracts.turn_answer import (
    DbaSufficiencyAssessment,
    TurnEvidenceFact,
)
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler
from platform_core.contracts.aiops import (
    ActionIntent,
    AnswerBlockType,
    CompactPlanningMode,
    CompactPlanningOutput,
    DiagnosticProfile,
    EvidenceSourceStrategy,
    ImplementationProfile,
    MeasurementSemantics,
    SufficiencyStatus,
    TaskObjective,
)
from platform_core.contracts.aiops.playbooks import PresentationPreference


ROOT = Path(__file__).resolve().parents[3]


def _compact() -> CompactPlanningOutput:
    return CompactPlanningOutput(
        planning_mode=CompactPlanningMode.IMPLEMENTATION_RUNBOOK,
        objectives=(TaskObjective.PLAN,),
        action_intent=ActionIntent.NONE,
        diagnostic_profile=DiagnosticProfile.GENERAL,
        implementation_profile=ImplementationProfile.ORACLE_ADG_BUILD,
        evidence_source_strategy=EvidenceSourceStrategy.DATABASE_FIRST,
        subject_ref={},
        problem_statement="结合当前参数生成完整 ADG 建设方案",
        success_criteria=("生成完整实施 Runbook",),
        public_reasoning_summary="读取当前参数后生成 ADG 实施 Runbook",
    )


def _precheck_fact(
    *,
    log_mode: str = "NOARCHIVELOG",
    force_logging: str = "NO",
    shortage: int = 2,
) -> TurnEvidenceFact:
    names = (
        "database_name",
        "db_unique_name",
        "instance_name",
        "host_name",
        "service_names",
        "container_name",
        "container_id",
        "database_role",
        "log_mode",
        "force_logging",
        "flashback_on",
        "protection_mode",
        "db_recovery_file_dest",
        "db_recovery_file_dest_size",
        "db_create_file_dest",
        "standby_file_management",
        "standby_redo_shortage",
        "online_redo_max_size_mb",
        "redo_thread_plan",
    )
    return TurnEvidenceFact(
        evidence_ref="evidence:adg-precheck",
        artifact_id="artifact-adg-precheck",
        source_id="oracle-source",
        step_id="a2",
        tool_id="db.ha.adg_precheck",
        measurement_semantics=MeasurementSemantics.CURRENT_ACTIVITY,
        presentation_kind=PresentationPreference.TABLE,
        captured_at="2026-09-28T04:00:00Z",
        columns=tuple({"name": name} for name in names),
        rows=(
            (
                "TESTDB",
                "testdb",
                "testdb1",
                "db-primary.example.com",
                "testdb.example.com",
                "CDB$ROOT",
                1,
                "PRIMARY",
                log_mode,
                force_logging,
                "YES",
                "MAXIMUM PERFORMANCE",
                "+FRA",
                "107374182400",
                "+DATA",
                "AUTO",
                shortage,
                200,
                f"1:2:3:{shortage}:200",
            ),
        ),
        row_count=1,
    )


class ImplementationRunbookPlanningTests(unittest.TestCase):
    def test_contract_repairs_omitted_adg_profile_and_fixed_semantics(
        self,
    ) -> None:
        compact = CompactPlanningOutput.model_validate(
            {
                "planning_mode": "IMPLEMENTATION_RUNBOOK",
                "objectives": ["PLAN", "ASSESS"],
                "action_intent": "ADVISORY",
                "diagnostic_profile": "SINGLE_SQL_PERFORMANCE",
                "subject_ref": {},
                "problem_statement": "根据当前参数生成 ADG 实施文档",
                "success_criteria": ["生成完整实施 Runbook"],
                "public_reasoning_summary": "生成 ADG 实施方案",
            }
        )

        self.assertEqual(
            ImplementationProfile.ORACLE_ADG_BUILD,
            compact.implementation_profile,
        )
        self.assertEqual((TaskObjective.PLAN,), compact.objectives)
        self.assertEqual(ActionIntent.NONE, compact.action_intent)
        self.assertEqual(
            DiagnosticProfile.GENERAL,
            compact.diagnostic_profile,
        )

    def test_schema_requires_explicit_implementation_profile(self) -> None:
        required = CompactPlanningOutput.model_json_schema()["required"]

        self.assertIn("implementation_profile", required)

    def test_adg_profile_repairs_full_investigation_mode(self) -> None:
        compact = CompactPlanningOutput.model_validate(
            {
                "planning_mode": "FULL_INVESTIGATION",
                "objectives": ["ASSESS"],
                "action_intent": "NONE",
                "diagnostic_profile": "GENERAL",
                "implementation_profile": "ORACLE_ADG_BUILD",
                "subject_ref": {},
                "problem_statement": "根据当前参数生成 ADG 实施文档",
                "success_criteria": ["生成完整实施 Runbook"],
                "public_reasoning_summary": "需要形成完整实施方案",
            }
        )

        self.assertEqual(
            CompactPlanningMode.IMPLEMENTATION_RUNBOOK,
            compact.planning_mode,
        )

    def test_adg_profile_expands_to_fixed_readonly_plan(self) -> None:
        compact = _compact()

        output = TurnPlanningService._implementation_runbook_output(
            question="结合当前参数生成完整 ADG 实施方案",
            compact=compact,
            target_context={"display_name": "TestDB"},
        )

        self.assertEqual((TaskObjective.PLAN,), output.task_frame.objectives)
        self.assertEqual(ActionIntent.NONE, output.task_frame.action_intent)
        self.assertFalse(output.task_frame.requires_change)
        self.assertEqual(
            ImplementationProfile.ORACLE_ADG_BUILD,
            output.task_frame.implementation_profile,
        )
        self.assertEqual(
            ("db.instance.identity", "db.ha.adg_precheck"),
            tuple(action.tool_id for action in output.plan.actions),
        )
        self.assertEqual(
            ("db.instance.identity", "db.ha.adg_precheck"),
            TurnPlanningService._profile_tool_ids(compact),
        )


class OracleAdgRunbookTests(unittest.TestCase):
    def test_missing_prerequisites_become_remediation_steps(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_ADG_BUILD,
            evidence=(_precheck_fact(),),
        )
        steps = {
            step.step_id: step
            for phase in runbook.phases
            for step in phase.steps
        }

        self.assertEqual(
            RunbookStatus.BLOCKED_BY_REQUIRED_INPUTS,
            runbook.status,
        )
        self.assertEqual("AIOPS_IMPLEMENTATION_RUNBOOK.v2", runbook.schema_version)
        self.assertEqual(
            RunbookApplicability.REQUIRED,
            steps["primary.archivelog"].applicability,
        )
        self.assertEqual(
            RunbookApplicability.REQUIRED,
            steps["primary.force_logging"].applicability,
        )
        self.assertEqual(
            RunbookApplicability.REQUIRED,
            steps["primary.srl"].applicability,
        )
        command_text = "\n".join(
            command.content
            for phase in runbook.phases
            for step in phase.steps
            for command in step.commands
        )
        self.assertIn("ALTER DATABASE ARCHIVELOG", command_text)
        self.assertIn("ALTER DATABASE FORCE LOGGING", command_text)
        self.assertIn(
            "DG_CONFIG=(testdb,testdb_stby)",
            command_text,
        )
        self.assertIn(
            "SERVICE=testdb_stby ASYNC NOAFFIRM",
            command_text,
        )
        self.assertIn("DB_UNIQUE_NAME=testdb_stby", command_text)
        self.assertNotIn("${", command_text)
        self.assertNotIn(
            "STANDBY_DB_UNIQUE_NAME",
            {item.key for item in runbook.required_inputs},
        )
        self.assertEqual(
            RunbookApplicability.BLOCKED,
            steps["standby.duplicate"].applicability,
        )

    def test_confirmed_external_facts_render_complete_commands(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_ADG_BUILD,
            evidence=(_precheck_fact(),),
            context={
                "implementation_parameters": {
                    "STANDBY_HOST": "db-standby.example.com",
                    "ORACLE_HOME": "/u01/app/oracle/product/26ai/dbhome_1",
                    "STANDBY_STORAGE": (
                        "*.db_create_file_dest='+DATA'\n"
                        "*.db_recovery_file_dest='+FRA'\n"
                        "*.db_recovery_file_dest_size=107374182400"
                    ),
                }
            },
        )
        command_text = "\n".join(
            command.content
            for phase in runbook.phases
            for step in phase.steps
            for command in step.commands
        )

        self.assertEqual(RunbookStatus.READY, runbook.status)
        self.assertFalse(runbook.required_inputs)
        self.assertIn("DUPLICATE TARGET DATABASE FOR STANDBY", command_text)
        self.assertIn("HOST=db-standby.example.com", command_text)
        self.assertIn("db_unique_name='testdb_stby'", command_text)
        self.assertNotIn("${", command_text)

    def test_satisfied_prerequisites_are_marked_without_removing_steps(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_ADG_BUILD,
            evidence=(
                _precheck_fact(
                    log_mode="ARCHIVELOG",
                    force_logging="YES",
                    shortage=0,
                ),
            ),
        )
        steps = {
            step.step_id: step
            for phase in runbook.phases
            for step in phase.steps
        }

        for step_id in (
            "primary.archivelog",
            "primary.force_logging",
            "primary.srl",
        ):
            self.assertEqual(
                RunbookApplicability.ALREADY_SATISFIED,
                steps[step_id].applicability,
            )

    def test_missing_precheck_still_returns_complete_partial_runbook(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_ADG_BUILD,
            evidence=(),
        )

        self.assertEqual(RunbookStatus.PARTIAL_EVIDENCE, runbook.status)
        self.assertGreaterEqual(len(runbook.phases), 8)
        self.assertTrue(runbook.required_inputs)


class ImplementationRunbookAnswerTests(unittest.TestCase):
    def test_answer_contains_runbook_and_never_proposal(self) -> None:
        handler = DbaAnswerComposeHandler(model_client=None, prompts=None)
        context = TaskExecutionContext(
            run_id="run",
            task_id="task",
            task_key="answer",
            target_id="target",
            agent_id="agent",
            trigger_type="API",
            trace_id="trace",
            attempt=1,
            deadline_at=None,
            plan_snapshot={
                "answer_context": {
                    "task_frame": {
                        "objectives": ["PLAN"],
                        "action_intent": "NONE",
                        "implementation_profile": "ORACLE_ADG_BUILD",
                    },
                    "workflow_kind": "",
                }
            },
            policy_snapshot={},
            input_artifacts=(),
        )
        assessment = DbaSufficiencyAssessment(
            status=SufficiencyStatus.ANSWERABLE,
            evidence=(_precheck_fact(),),
        )

        blocks = handler._assemble_blocks(
            context=context,
            assessment=assessment,
            markdown="已依据当前参数生成完整实施方案。",
        )

        self.assertIn(
            AnswerBlockType.IMPLEMENTATION_RUNBOOK,
            tuple(block.block_type for block in blocks),
        )
        self.assertNotIn(
            AnswerBlockType.PROPOSAL_SUMMARY,
            tuple(block.block_type for block in blocks),
        )
        self.assertFalse(handler._include_proposal(context))

    def test_ui_has_runbook_renderer_and_copy_buttons(self) -> None:
        source = (ROOT / "ui/aiops/js/aiops-workspaces.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('block.block_type === "IMPLEMENTATION_RUNBOOK"', source)
        self.assertIn("payload.resolved_parameters", source)
        self.assertIn('BLOCKED: "等待外部输入"', source)
        self.assertIn("data-copy-code", source)

    def test_partial_runbook_does_not_degrade_to_evidence_request_only(self) -> None:
        handler = DbaAnswerComposeHandler(model_client=None, prompts=None)
        context = TaskExecutionContext(
            run_id="run",
            task_id="task",
            task_key="answer",
            target_id="target",
            agent_id="agent",
            trigger_type="API",
            trace_id="trace",
            attempt=1,
            deadline_at=None,
            plan_snapshot={
                "answer_context": {
                    "task_frame": {
                        "objectives": ["PLAN"],
                        "action_intent": "NONE",
                        "implementation_profile": "ORACLE_ADG_BUILD",
                    },
                    "workflow_kind": "",
                }
            },
            policy_snapshot={},
            input_artifacts=(),
        )
        assessment = DbaSufficiencyAssessment(
            status=SufficiencyStatus.CAPABILITY_UNAVAILABLE,
            reasons=("当前未取得数据库前置参数",),
        )

        blocks = handler._assemble_blocks(
            context=context,
            assessment=assessment,
            markdown="当前前置证据不完整，已生成受输入阻断的方案。",
        )

        block_types = tuple(block.block_type for block in blocks)
        self.assertIn(AnswerBlockType.IMPLEMENTATION_RUNBOOK, block_types)
        self.assertNotIn(AnswerBlockType.EVIDENCE_REQUEST, block_types)

    def test_oracle_schema_accepts_implementation_runbook_block(self) -> None:
        canonical = (
            ROOT
            / "database/oracle/aiops_agent/008_ops_conversations_reports.sql"
        ).read_text(encoding="utf-8")
        upgrade = (
            ROOT / "database/oracle/operations/apply_aiops_schema_27.sql"
        ).read_text(encoding="utf-8")
        readiness = (
            ROOT
            / "services/aiops_agent/src/aiops_agent/bootstrap/common.py"
        ).read_text(encoding="utf-8")

        for source in (canonical, upgrade, readiness):
            self.assertIn("IMPLEMENTATION_RUNBOOK", source)


if __name__ == "__main__":
    unittest.main()
