"""数据库实施文档中心 v3 的档案、脚本和导出测试。"""

from __future__ import annotations

import hashlib
import io
import unittest
import zipfile
from types import SimpleNamespace

from aiops_agent.application.implementation import compile_implementation_runbook
from aiops_agent.application.implementation.artifacts import (
    generated_artifact_payloads,
    render_runbook_artifact_zip,
)
from aiops_agent.application.implementation.registry import (
    registered_implementation_profiles,
)
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.contracts.implementation import RunbookStatus
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from platform_core.contracts.aiops import (
    ImplementationProfile,
    MeasurementSemantics,
)
from platform_core.contracts.aiops.playbooks import PresentationPreference


def _rman_fact(
    *,
    rman_configuration: str = "",
    recovery_dest: str = "",
    log_mode: str = "ARCHIVELOG",
) -> TurnEvidenceFact:
    names = (
        "instance_name",
        "database_name",
        "db_unique_name",
        "database_role",
        "log_mode",
        "db_recovery_file_dest",
        "rman_configuration",
    )
    return TurnEvidenceFact(
        evidence_ref="evidence:rman-precheck",
        artifact_id="artifact-rman-precheck",
        source_id="oracle-source",
        step_id="precheck",
        tool_id="db.backup.rman_configuration",
        measurement_semantics=MeasurementSemantics.CURRENT_ACTIVITY,
        presentation_kind=PresentationPreference.TABLE,
        captured_at="2026-09-28T08:00:00Z",
        columns=tuple({"name": name} for name in names),
        rows=((
            "TESTDB",
            "TESTDB",
            "testdb",
            "PRIMARY",
            log_mode,
            recovery_dest,
            rman_configuration,
        ),),
        row_count=1,
    )


class DatabaseImplementationLibraryTest(unittest.TestCase):
    def test_registry_covers_every_implementation_profile(self) -> None:
        self.assertEqual(
            set(ImplementationProfile) - {ImplementationProfile.NONE},
            set(registered_implementation_profiles()),
        )

    def test_every_profile_expands_to_identity_and_fixed_precheck(self) -> None:
        for profile, profile_config in (
            TurnPlanningService.IMPLEMENTATION_PROFILE_CATALOG.items()
        ):
            output = TurnPlanningService._implementation_runbook_output(
                question="生成实施文档",
                compact=SimpleNamespace(
                    implementation_profile=profile,
                    subject_ref={},
                ),
                target_context={
                    "target_id": "target-1",
                    "display_name": "TestDB",
                },
            )
            self.assertEqual(profile, output.task_frame.implementation_profile)
            self.assertEqual(
                ("db.instance.identity", profile_config[0]),
                tuple(item.tool_id for item in output.plan.actions),
            )
            self.assertEqual(
                (profile_config[1],),
                output.suggested_playbook_ids,
            )

    def test_missing_infrastructure_facts_block_steps_without_placeholders(self) -> None:
        for profile in registered_implementation_profiles():
            if profile == ImplementationProfile.ORACLE_ADG_BUILD:
                continue
            runbook = compile_implementation_runbook(
                profile=profile,
                evidence=(),
                context={},
            )
            self.assertEqual(
                RunbookStatus.BLOCKED_BY_REQUIRED_FACTS,
                runbook.status,
                profile,
            )
            self.assertTrue(runbook.missing_facts, profile)
            serialized = runbook.model_dump_json()
            for marker in ("${", "CHANGEME", "TODO", "<NODE", "<SCAN"):
                self.assertNotIn(marker, serialized)

    def test_rman_backup_artifacts_are_hashed_and_zip_is_deterministic(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_RMAN_BACKUP_BUILD,
            evidence=(),
            context={
                "implementation_parameters": {
                    "INSTANCE_NAME": "TESTDB",
                    "BACKUP_DEST": "/backup/testdb",
                }
            },
        )
        self.assertEqual(RunbookStatus.PARTIAL_EVIDENCE, runbook.status)
        self.assertGreaterEqual(len(runbook.artifacts), 10)
        generated = generated_artifact_payloads(runbook)
        contents = {
            item["artifact_id"]: item["content"] for item in generated
        }
        for descriptor in runbook.artifacts:
            self.assertEqual(
                descriptor.sha256,
                hashlib.sha256(
                    contents[descriptor.artifact_id].encode("utf-8")
                ).hexdigest(),
            )
        payload = runbook.model_dump(mode="json")
        first = render_runbook_artifact_zip(payload, contents)
        second = render_runbook_artifact_zip(payload, contents)
        self.assertEqual(first, second)
        with zipfile.ZipFile(io.BytesIO(first)) as archive:
            names = archive.namelist()
            self.assertEqual("00-manifest.json", names[0])
            self.assertIn(
                "oracle-rman-backup/rman/backup_level0.rman",
                names,
            )
            self.assertIn(
                "oracle-rman-backup/bin/run-rman-job.sh",
                names,
            )

    def test_rman_backup_uses_real_commands_for_every_ready_phase(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_RMAN_BACKUP_BUILD,
            evidence=(),
            context={
                "implementation_parameters": {
                    "INSTANCE_NAME": "TESTDB",
                    "BACKUP_DEST": "/backup/testdb",
                }
            },
        )

        commands = [
            command
            for phase in runbook.phases
            for step in phase.steps
            for command in step.commands
        ]
        self.assertTrue(commands)
        self.assertTrue(all(step.commands for phase in runbook.phases for step in phase.steps))
        self.assertNotIn("MANUAL", {item.command_type.value for item in commands})
        serialized = runbook.model_dump_json()
        self.assertNotIn("按本阶段检查表实施并留存证据", serialized)
        self.assertIn("REPORT NEED BACKUP RECOVERY WINDOW OF 7 DAYS", serialized)
        self.assertIn("RESTORE DATABASE VALIDATE CHECK LOGICAL", serialized)
        self.assertIn("promtool check rules", serialized)
        self.assertIn("df -P /backup/testdb", serialized)

    def test_blocked_steps_do_not_publish_manual_pseudo_commands(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_RMAN_BACKUP_BUILD,
            evidence=(),
            context={},
        )

        blocked_steps = [
            step
            for phase in runbook.phases
            for step in phase.steps
            if step.applicability.value == "BLOCKED"
        ]
        self.assertTrue(blocked_steps)
        self.assertTrue(all(not step.commands for step in blocked_steps))
        self.assertTrue(
            all(
                step.applicability.value == "BLOCKED" or step.commands
                for phase in runbook.phases
                for step in phase.steps
            )
        )
        self.assertNotIn(
            "登记并重新核验实施事实",
            runbook.model_dump_json(),
        )

    def test_rman_backup_adds_archivelog_conversion_for_noarchivelog(self) -> None:
        evidence = (_rman_fact(log_mode="NOARCHIVELOG"),)
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_RMAN_BACKUP_BUILD,
            evidence=evidence,
            context={},
        )

        self.assertIn("ALTER DATABASE ARCHIVELOG", runbook.model_dump_json())

    def test_rman_backup_uses_oraenv_and_existing_rman_destination(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_RMAN_BACKUP_BUILD,
            evidence=(_rman_fact(
                rman_configuration=(
                    "CHANNEL=DEVICE TYPE DISK FORMAT "
                    "'/backup/existing/%d_%T_%U.bkp'"
                ),
                recovery_dest="/u02/fast_recovery_area",
            ),),
            context={},
        )

        self.assertFalse(runbook.missing_facts)
        parameters = {item.key: item.value for item in runbook.resolved_parameters}
        self.assertEqual("/backup/existing", parameters["BACKUP_DEST"])
        self.assertEqual("RMAN_CONFIGURATION", parameters["BACKUP_DEST_SOURCE"])
        self.assertNotIn("ORACLE_HOME", parameters)
        generated = {
            item["artifact_id"]: item["content"]
            for item in generated_artifact_payloads(runbook)
        }
        self.assertIn("oraenv", generated["rman.oracle.env"])
        self.assertNotIn("$ORACLE_HOME", generated["rman.runner"])
        self.assertIn("rman target /", generated["rman.runner"])
        self.assertNotIn(
            "CONFIGURE CHANNEL DEVICE TYPE DISK FORMAT",
            generated["rman.configure"],
        )

    def test_rman_backup_derives_and_configures_default_destination(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_RMAN_BACKUP_BUILD,
            evidence=(_rman_fact(),),
            context={},
        )

        parameters = {item.key: item.value for item in runbook.resolved_parameters}
        self.assertEqual(
            "/u01/app/oracle/backup/testdb",
            parameters["BACKUP_DEST"],
        )
        self.assertEqual("YES", parameters["BACKUP_DEST_REVIEW_REQUIRED"])
        serialized = runbook.model_dump_json()
        self.assertIn("CONFIGURE CHANNEL DEVICE TYPE DISK FORMAT", serialized)
        self.assertIn("按实际独立备份挂载点调整路径", serialized)

    def test_rman_backup_reuses_filesystem_recovery_destination(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_RMAN_BACKUP_BUILD,
            evidence=(_rman_fact(
                recovery_dest="/u02/fast_recovery_area",
            ),),
            context={},
        )

        parameters = {item.key: item.value for item in runbook.resolved_parameters}
        self.assertEqual(
            "/u02/fast_recovery_area",
            parameters["BACKUP_DEST"],
        )
        self.assertEqual(
            "DB_RECOVERY_FILE_DEST",
            parameters["BACKUP_DEST_SOURCE"],
        )
        self.assertEqual("NO", parameters["BACKUP_DEST_REVIEW_REQUIRED"])

    def test_pitr_without_target_is_blocked_and_has_no_set_until(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_RMAN_RECOVERY,
            evidence=(),
            context={
                "implementation_parameters": {
                    "RECOVERY_SCENARIO": "PITR",
                    "BACKUP_CHAIN_STATUS": "AVAILABLE",
                    "ORACLE_HOME": "/u01/app/oracle/product/26ai/dbhome_1",
                    "INSTANCE_NAME": "TESTDB",
                }
            },
        )
        self.assertEqual(
            RunbookStatus.BLOCKED_BY_REQUIRED_FACTS,
            runbook.status,
        )
        self.assertIn(
            "RECOVERY_TARGET",
            {item.fact_key for item in runbook.missing_facts},
        )
        self.assertNotIn("SET UNTIL", runbook.model_dump_json())

    def test_external_fact_cannot_inject_shell_content(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_RMAN_BACKUP_BUILD,
            evidence=(),
            context={
                "implementation_parameters": {
                    "INSTANCE_NAME": "TESTDB",
                    "BACKUP_DEST": "/backup/testdb;touch /tmp/unsafe",
                }
            },
        )

        self.assertEqual(
            RunbookStatus.PARTIAL_EVIDENCE,
            runbook.status,
        )
        self.assertNotIn("BACKUP_DEST", {item.fact_key for item in runbook.missing_facts})
        self.assertIn("/u01/app/oracle/backup/testdb", runbook.model_dump_json())
        self.assertNotIn("touch /tmp/unsafe", runbook.model_dump_json())


if __name__ == "__main__":
    unittest.main()
