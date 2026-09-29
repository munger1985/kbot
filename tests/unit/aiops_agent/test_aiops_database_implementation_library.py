"""数据库实施文档中心 v3 的档案、脚本和导出测试。"""

from __future__ import annotations

import hashlib
import io
import json
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


def _database_fact(
    *,
    tool_id: str,
    version: str = "",
    compatible: str = "",
) -> TurnEvidenceFact:
    names = ["database_name", "version"]
    values = ["TESTDB", version]
    if tool_id == "db.ha.rac_precheck":
        names.append("compatible")
        values.append(compatible)
    return TurnEvidenceFact(
        evidence_ref=f"evidence:{tool_id}",
        artifact_id=f"artifact-{tool_id}",
        source_id="oracle-source",
        step_id="precheck",
        tool_id=tool_id,
        measurement_semantics=MeasurementSemantics.CURRENT_ACTIVITY,
        presentation_kind=PresentationPreference.TABLE,
        captured_at="2026-09-29T08:00:00Z",
        columns=tuple({"name": name} for name in names),
        rows=(tuple(values),),
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
            self.assertEqual("README.md", names[1])
            self.assertIn(
                "oracle-rman-backup/rman/backup_level0.rman",
                names,
            )
            self.assertIn(
                "oracle-rman-backup/bin/run-rman-job.sh",
                names,
            )
            readme = archive.read("README.md").decode("utf-8")
            self.assertIn("下载格式的定位", readme)
            self.assertIn("RMAN 备份包使用顺序", readme)
            self.assertIn(
                "/var/tmp/kbot-runbooks/oracle-rman-backup/"
                "bin/run-rman-job.sh level0",
                readme,
            )
            self.assertNotIn("JSON：", readme)
            manifest = json.loads(
                archive.read("00-manifest.json").decode("utf-8")
            )
            self.assertEqual(
                hashlib.sha256(readme.encode("utf-8")).hexdigest(),
                manifest["package_files"][0]["sha256"],
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
        manual_commands = [
            item.command_id
            for item in commands
            if item.command_type.value == "MANUAL"
        ]
        self.assertEqual(["artifact.package.transfer"], manual_commands)
        serialized = runbook.model_dump_json()
        self.assertNotIn("按本阶段检查表实施并留存证据", serialized)
        self.assertIn("REPORT NEED BACKUP RECOVERY WINDOW OF 7 DAYS", serialized)
        self.assertIn("RESTORE DATABASE VALIDATE CHECK LOGICAL", serialized)
        self.assertIn("promtool check rules", serialized)
        self.assertIn("df -P /backup/testdb", serialized)

    def test_rac_places_zip_deployment_before_artifact_commands(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_RAC_BUILD,
            evidence=(),
            context={
                "implementation_parameters": {
                    "DATABASE_NAME": "TESTDB",
                    "VERSION": "26ai",
                }
            },
        )

        self.assertEqual("artifact_package", runbook.phases[0].phase_id)
        deployment = runbook.phases[0].steps[0]
        command_by_id = {
            item.command_id: item
            for item in deployment.commands
        }
        self.assertIn(
            "/var/tmp/oracle-rac-build.zip",
            command_by_id["artifact.package.transfer"].content,
        )
        self.assertIn(
            "unzip -oq /var/tmp/oracle-rac-build.zip "
            "-d /var/tmp/kbot-runbooks",
            command_by_id["artifact.package.deploy"].content,
        )
        self.assertIn(
            "/var/tmp/kbot-runbooks/oracle-rac-build/"
            "database/prepare-source.sql",
            deployment.verification_commands[0].content,
        )
        phase_ids = [item.phase_id for item in runbook.phases]
        self.assertLess(
            phase_ids.index("artifact_package"),
            phase_ids.index("source_protection"),
        )
        steps = {
            step.step_id: step
            for phase in runbook.phases
            for step in phase.steps
        }
        for step_id in (
            "os_prepare.execute",
            "network.execute",
            "shared_storage.execute",
            "grid_install.execute",
            "asm.execute",
        ):
            self.assertNotEqual("BLOCKED", steps[step_id].applicability.value)
            self.assertTrue(steps[step_id].commands)
        serialized = runbook.model_dump_json()
        self.assertIn("gridSetup.sh", serialized)
        self.assertIn("runcluvfy.sh", serialized)
        self.assertIn("asmca", serialized)
        self.assertIn("oracle-database-preinstall-26ai", serialized)
        missing_keys = {item.fact_key for item in runbook.missing_facts}
        self.assertNotIn("GRID_HOME", missing_keys)
        self.assertNotIn("NODE1_VIP", missing_keys)
        self.assertNotIn("NODE2_VIP", missing_keys)
        self.assertNotIn("ASM_DISK_WWIDS", missing_keys)

    def test_rac_uses_target_release_for_preinstall_package(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_RAC_BUILD,
            evidence=(),
            context={
                "implementation_parameters": {
                    "DATABASE_NAME": "TESTDB",
                    "VERSION": "26ai",
                }
            },
        )

        serialized = runbook.model_dump_json()
        self.assertIn("oracle-database-preinstall-26ai", serialized)
        self.assertNotIn("oracle-database-preinstall-23ai", serialized)
        resolved = {
            item.key: item.value for item in runbook.resolved_parameters
        }
        self.assertEqual("26ai", resolved["ORACLE_RELEASE"])
        self.assertEqual(
            "TARGET_CONFIGURATION",
            resolved["ORACLE_RELEASE_SOURCE"],
        )

    def test_rac_maps_26ai_internal_version_without_using_base_version(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_RAC_BUILD,
            evidence=(
                _database_fact(
                    tool_id="db.instance.identity",
                    version="23.26.0.0.0",
                ),
                _database_fact(
                    tool_id="db.ha.rac_precheck",
                    version="23.0.0.0.0",
                    compatible="23.0.0",
                ),
            ),
            context={},
        )

        serialized = runbook.model_dump_json()
        self.assertIn("oracle-database-preinstall-26ai", serialized)
        self.assertNotIn("oracle-database-preinstall-23ai", serialized)
        resolved = {
            item.key: item.value for item in runbook.resolved_parameters
        }
        self.assertEqual(
            "V$INSTANCE.VERSION_FULL",
            resolved["ORACLE_RELEASE_SOURCE"],
        )
        self.assertEqual("23.26.0.0.0", resolved["ORACLE_OBSERVED_VERSION"])

    def test_rac_maps_19c_release_from_database_version(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_RAC_BUILD,
            evidence=(
                _database_fact(
                    tool_id="db.instance.identity",
                    version="19.27.0.0.0",
                ),
            ),
            context={},
        )

        serialized = runbook.model_dump_json()
        self.assertIn("oracle-database-preinstall-19c", serialized)
        self.assertNotIn("oracle-database-preinstall-23ai", serialized)

    def test_rac_blocks_version_specific_steps_when_release_is_unknown(self) -> None:
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_RAC_BUILD,
            evidence=(),
            context={
                "implementation_parameters": {
                    "DATABASE_NAME": "TESTDB",
                }
            },
        )

        steps = {
            step.step_id: step
            for phase in runbook.phases
            for step in phase.steps
        }
        for step_id in ("os_prepare", "grid_install", "asm"):
            self.assertEqual("BLOCKED", steps[step_id].applicability.value)
            self.assertFalse(steps[step_id].commands)
        serialized = runbook.model_dump_json()
        self.assertNotIn("oracle-database-preinstall-", serialized)
        self.assertNotIn("/u01/app/23ai/grid", serialized)
        self.assertNotIn("/u01/app/26ai/grid", serialized)
        self.assertNotIn(
            "rac.verify.cluster",
            {item.artifact_id for item in runbook.artifacts},
        )
        resolved_keys = {item.key for item in runbook.resolved_parameters}
        self.assertNotIn("ORACLE_RELEASE", resolved_keys)
        self.assertNotIn("GRID_HOME", resolved_keys)

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
