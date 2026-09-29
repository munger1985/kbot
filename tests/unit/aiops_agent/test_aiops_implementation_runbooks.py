import asyncio
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
    remote_login_passwordfile: str = "NONE",
    shortage: int = 2,
    configured_dataguard: bool = False,
    container_name: str = "CDB$ROOT",
    container_id: int = 1,
) -> TurnEvidenceFact:
    names = (
        "database_name",
        "db_unique_name",
        "instance_name",
        "host_name",
        "service_names",
        "db_domain",
        "compatible",
        "audit_file_dest",
        "diagnostic_dest",
        "control_files",
        "spfile",
        "processes",
        "sga_target",
        "platform_name",
        "cdb",
        "container_name",
        "container_id",
        "database_role",
        "open_mode",
        "log_mode",
        "force_logging",
        "flashback_on",
        "protection_mode",
        "switchover_status",
        "remote_login_passwordfile",
        "log_archive_config",
        "log_archive_dest_1",
        "log_archive_dest_2",
        "log_archive_dest_state_2",
        "fal_server",
        "fal_client",
        "standby_file_management",
        "db_file_name_convert",
        "log_file_name_convert",
        "db_create_file_dest",
        "db_recovery_file_dest",
        "db_recovery_file_dest_size",
        "dg_broker_start",
        "character_set",
        "national_character_set",
        "sample_datafile_path",
        "sample_tempfile_path",
        "sample_redo_member_path",
        "password_file_path",
        "datafile_bytes",
        "max_redo_group_number",
        "fra_space_limit_bytes",
        "fra_space_used_bytes",
        "online_redo_threads",
        "online_redo_groups",
        "online_redo_min_size_mb",
        "online_redo_max_size_mb",
        "standby_redo_groups",
        "standby_redo_min_size_mb",
        "standby_redo_max_size_mb",
        "required_standby_redo_groups",
        "standby_redo_shortage",
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
                "example.com",
                "26.0.0",
                "/u01/app/oracle/admin/TESTDB/adump",
                "/u01/app/oracle",
                "/u02/oradata/TESTDB/control01.ctl",
                "/u01/app/oracle/product/26ai/dbhome_1/dbs/spfiletestdb1.ora",
                1000,
                4294967296,
                "Linux x86 64-bit",
                "YES",
                container_name,
                container_id,
                "PRIMARY",
                "READ WRITE",
                log_mode,
                force_logging,
                "YES",
                "MAXIMUM PERFORMANCE",
                "TO STANDBY",
                remote_login_passwordfile,
                "DG_CONFIG=(testdb,testdb_stby)" if configured_dataguard else "",
                "LOCATION=USE_DB_RECOVERY_FILE_DEST VALID_FOR=(ALL_LOGFILES,ALL_ROLES) DB_UNIQUE_NAME=testdb",
                (
                    "SERVICE=testdb_stby ASYNC NOAFFIRM "
                    "VALID_FOR=(ONLINE_LOGFILES,PRIMARY_ROLE) "
                    "DB_UNIQUE_NAME=testdb_stby"
                    if configured_dataguard else ""
                ),
                "ENABLE",
                "testdb_stby" if configured_dataguard else "",
                "",
                "AUTO",
                "",
                "",
                "/u02/oradata/TESTDB",
                "/u03/fra",
                "107374182400",
                "TRUE" if configured_dataguard else "FALSE",
                "AL32UTF8",
                "AL16UTF16",
                "/u02/oradata/TESTDB/PDB01/system01.dbf",
                "/u02/oradata/TESTDB/PDB01/temp01.dbf",
                "/u02/oradata/TESTDB/redo01.log",
                "/u01/app/oracle/product/26ai/dbhome_1/dbs/orapwtestdb1",
                53687091200,
                3,
                107374182400,
                10737418240,
                1,
                2,
                200,
                200,
                0,
                None,
                None,
                3,
                shortage,
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

        self.assertEqual(RunbookStatus.READY, runbook.status)
        self.assertEqual("AIOPS_IMPLEMENTATION_RUNBOOK.v3", runbook.schema_version)
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
            steps["primary.passwordfile_mode"].applicability,
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
        self.assertIn("remote_login_passwordfile='EXCLUSIVE'", command_text)
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
            RunbookApplicability.REQUIRED,
            steps["standby.duplicate"].applicability,
        )
        self.assertFalse(runbook.required_inputs)

    def test_target_infrastructure_defaults_render_complete_commands(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_ADG_BUILD,
            evidence=(_precheck_fact(),),
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
        self.assertIn("HOST=db-primary-stby.example.com", command_text)
        self.assertIn("/u01/app/oracle/product/26ai/dbhome_1", command_text)
        self.assertIn("/u02/oradata/TESTDB", command_text)
        self.assertIn("oracle-database-preinstall-26ai", command_text)
        self.assertIn(
            "scp oracle@db-primary.example.com:"
            "/u01/app/oracle/product/26ai/dbhome_1/dbs/orapwtestdb1 "
            "/u01/app/oracle/product/26ai/dbhome_1/dbs/orapwTESTDBSTBY",
            command_text,
        )
        self.assertIn("db_unique_name='testdb_stby'", command_text)
        self.assertNotIn("${", command_text)
        self.assertNotRegex(command_text, r"<[A-Z][A-Z0-9_]*>")

    def test_pdb_target_generates_zero_input_dgpdb_runbook(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_ADG_BUILD,
            evidence=(
                _precheck_fact(
                    container_name="PDB01",
                    container_id=3,
                ),
            ),
        )
        command_text = "\n".join(
            command.content
            for phase in runbook.phases
            for step in phase.steps
            for command in step.commands
        )

        self.assertEqual(RunbookStatus.READY, runbook.status)
        self.assertFalse(runbook.required_inputs)
        self.assertIn("PDB 级 Data Guard（DGPDB）", runbook.title)
        self.assertIn(
            "dbca -silent -createDatabase \\\n  -templateName",
            command_text,
        )
        self.assertIn(
            "$ORACLE_HOME/runInstaller -silent -waitforcompletion \\\n"
            "  oracle.install.option=INSTALL_DB_SWONLY",
            command_text,
        )
        self.assertIn("ADD PLUGGABLE DATABASE 'PDB01'", command_text)
        self.assertIn(
            "ADD CONFIGURATION 'testdb_dgpdb_cfg' CONNECT IDENTIFIER IS "
            "'testdb_dgpdb';",
            command_text,
        )
        self.assertNotIn("DUPLICATE TARGET DATABASE", command_text)
        self.assertNotIn("CREATE CONFIGURATION GROUP", command_text)
        self.assertNotIn("ENABLE PLUGGABLE DATABASE", command_text)
        self.assertNotRegex(
            command_text,
            r"(?:SHOW|VALIDATE) PLUGGABLE DATABASE[^;]+VERBOSE",
        )
        self.assertIn("db-primary-stby.example.com", command_text)
        self.assertIn(
            "/u01/app/oracle/product/26ai/dbhome_1/dbs/orapwTESTDBS",
            command_text,
        )
        self.assertIn("/u02/oradata/TESTDB/PDB01", command_text)
        self.assertNotIn("${", command_text)
        self.assertNotRegex(command_text, r"<[A-Z][A-Z0-9_]*>")

    def test_satisfied_prerequisites_are_marked_without_removing_steps(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_ADG_BUILD,
            evidence=(
                _precheck_fact(
                    log_mode="ARCHIVELOG",
                    force_logging="YES",
                    remote_login_passwordfile="EXCLUSIVE",
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
            "primary.passwordfile_mode",
            "primary.srl",
        ):
            self.assertEqual(
                RunbookApplicability.ALREADY_SATISFIED,
                steps[step_id].applicability,
            )
            self.assertFalse(steps[step_id].commands)

    def test_existing_dataguard_parameters_only_emit_verification(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_ADG_BUILD,
            evidence=(
                _precheck_fact(
                    log_mode="ARCHIVELOG",
                    force_logging="YES",
                    remote_login_passwordfile="EXCLUSIVE",
                    shortage=0,
                    configured_dataguard=True,
                ),
            ),
        )
        steps = {
            step.step_id: step
            for phase in runbook.phases
            for step in phase.steps
        }

        self.assertEqual(
            RunbookApplicability.ALREADY_SATISFIED,
            steps["parameters.primary"].applicability,
        )
        self.assertFalse(steps["parameters.primary"].commands)
        self.assertTrue(steps["parameters.primary"].verification_commands)

    def test_runbook_ends_with_validation_and_daily_operations(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_ADG_BUILD,
            evidence=(_precheck_fact(),),
        )

        self.assertEqual("operations", runbook.phases[-1].phase_id)
        command_text = "\n".join(
            command.content
            for step in runbook.phases[-1].steps
            for command in step.commands
        )
        self.assertIn("SHOW CONFIGURATION VERBOSE", command_text)
        self.assertIn("v$dataguard_stats", command_text)
        self.assertIn("log_archive_dest_state_2=DEFER", command_text)

    def test_missing_precheck_still_returns_complete_partial_runbook(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_ADG_BUILD,
            evidence=(),
        )

        self.assertEqual(RunbookStatus.PARTIAL_EVIDENCE, runbook.status)
        self.assertGreaterEqual(len(runbook.phases), 8)
        self.assertFalse(runbook.required_inputs)


class ImplementationRunbookAnswerTests(unittest.TestCase):
    @staticmethod
    def _execution_context(
        assessment: DbaSufficiencyAssessment,
    ) -> TaskExecutionContext:
        return TaskExecutionContext(
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
            input_artifacts=(
                {
                    "artifact_id": "assessment",
                    "schema_version": "DBA_SUFFICIENCY.v1",
                    "payload": assessment.model_dump(mode="json"),
                },
            ),
        )

    def test_execute_bypasses_model_and_returns_runbook_directly(self) -> None:
        assessment = DbaSufficiencyAssessment(
            status=SufficiencyStatus.ANSWERABLE,
            evidence=(_precheck_fact(),),
        )
        handler = DbaAnswerComposeHandler(model_client=None, prompts=None)

        result = asyncio.run(handler.execute(self._execution_context(assessment)))

        self.assertEqual("COMPLETED", result.status)
        self.assertIsNone(result.model_receipt)
        self.assertIn(
            AnswerBlockType.IMPLEMENTATION_RUNBOOK,
            tuple(block.block_type for block in result.blocks),
        )
        self.assertEqual(
            "已按当前主库参数生成 ADG 实施操作文档；本轮仅生成文档，不执行命令。",
            result.blocks[0].payload["markdown"],
        )

    def test_stream_execute_emits_single_summary_then_runbook(self) -> None:
        assessment = DbaSufficiencyAssessment(
            status=SufficiencyStatus.ANSWERABLE,
            evidence=(_precheck_fact(),),
        )
        handler = DbaAnswerComposeHandler(model_client=None, prompts=None)

        async def collect():
            return [
                item
                async for item in handler.execute_stream(
                    self._execution_context(assessment)
                )
            ]

        items = asyncio.run(collect())

        self.assertEqual("answer.delta", items[0].event_type)
        self.assertTrue(items[-1].answer_streamed)
        self.assertIn(
            AnswerBlockType.IMPLEMENTATION_RUNBOOK,
            tuple(block.block_type for block in items[-1].blocks),
        )

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

    def test_rac_runbook_receives_version_from_target_snapshot(self) -> None:
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
                        "implementation_profile": "ORACLE_RAC_BUILD",
                    },
                    "workflow_kind": "",
                },
                "investigation_execution": {
                    "database": {
                        "configured_version": "26ai",
                    }
                },
            },
            policy_snapshot={},
            input_artifacts=(),
        )
        assessment = DbaSufficiencyAssessment(
            status=SufficiencyStatus.ANSWERABLE,
            evidence=(),
        )

        block = handler._implementation_runbook_block(
            context=context,
            assessment=assessment,
        )

        self.assertIsNotNone(block)
        payload = block.payload
        self.assertIn(
            "oracle-database-preinstall-26ai",
            str(payload),
        )
        self.assertNotIn("oracle-database-preinstall-23ai", str(payload))
        resolved = {
            item["key"]: item["value"]
            for item in payload["resolved_parameters"]
        }
        self.assertEqual("26ai", resolved["VERSION"])
        self.assertEqual("26ai", resolved["ORACLE_RELEASE"])

    def test_ui_has_runbook_renderer_and_copy_buttons(self) -> None:
        source = (ROOT / "ui/aiops/js/aiops-workspaces.js").read_text(
            encoding="utf-8"
        )
        styles = (ROOT / "ui/aiops/css/workspaces.css").read_text(
            encoding="utf-8"
        )
        self.assertIn('block.block_type === "IMPLEMENTATION_RUNBOOK"', source)
        self.assertIn("payload.resolved_parameters", source)
        self.assertIn('BLOCKED: "等待必要事实"', source)
        self.assertIn("data-copy-code", source)
        self.assertIn('data-download-implementation-runbook="pdf"', source)
        self.assertIn('data-download-implementation-runbook="markdown"', source)
        self.assertNotIn('data-download-implementation-runbook="json"', source)
        self.assertIn("implementation-runbook.${format.extension}", source)
        self.assertIn('class="ops-runbook-toc"', source)
        self.assertNotIn('<details class="ops-runbook-phase"', source)
        self.assertNotIn('<details class="ops-runbook-appendix"', source)
        self.assertNotIn(".ops-runbook-body { max-height:", styles)
        self.assertIn("white-space: pre-wrap", styles)
        self.assertLess(
            source.index('${commandGroup("人工确认项", manualItems, true)}'),
            source.index('${commandGroup("实施命令", implementationCommands)}'),
        )

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

    def test_oracle_schema_does_not_enumerate_answer_block_types(self) -> None:
        canonical = (
            ROOT
            / "database/oracle/aiops_agent/008_ops_conversations_reports.sql"
        ).read_text(encoding="utf-8")
        upgrade = (
            ROOT / "database/oracle/operations/apply_aiops_schema_28.sql"
        ).read_text(encoding="utf-8")

        self.assertNotIn("CK_OPS_ANSWER_BLOCK_TYPE", canonical)
        self.assertIn("CONSTRAINT_TYPE = 'C'", upgrade)
        self.assertIn(AnswerBlockType.IMPLEMENTATION_RUNBOOK, AnswerBlockType)


if __name__ == "__main__":
    unittest.main()
