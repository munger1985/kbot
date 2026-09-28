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
from platform_core.contracts.aiops import ImplementationProfile


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
                    "ORACLE_HOME": "/u01/app/oracle/product/26ai/dbhome_1",
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
                    "ORACLE_HOME": "/u01/app/oracle/product/26ai/dbhome_1",
                    "BACKUP_DEST": "/backup/testdb;touch /tmp/unsafe",
                }
            },
        )

        self.assertEqual(
            RunbookStatus.BLOCKED_BY_REQUIRED_FACTS,
            runbook.status,
        )
        self.assertIn(
            "BACKUP_DEST",
            {item.fact_key for item in runbook.missing_facts},
        )
        self.assertNotIn("touch /tmp/unsafe", runbook.model_dump_json())


if __name__ == "__main__":
    unittest.main()
