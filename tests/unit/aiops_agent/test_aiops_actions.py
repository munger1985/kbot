"""步骤 9 Action Catalog、参数血缘和 Advisory 降级测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.actions import (
    ActionCompilerRegistry,
    ActionRegistry,
    ActionRenderer,
)
from aiops_agent.actions.validation import validate_rendered_action
from aiops_agent.contracts.diagnosis import (
    EvidenceFact,
    EvidenceIndex,
    RootCauseAssessment,
    SolutionDraft,
)
from aiops_agent.contracts.turn_answer import (
    DbaSufficiencyAssessment,
    TurnEvidenceFact,
)
from aiops_agent.workers.change_handlers import (
    ActionPlanHandler,
    ChatActionPlanHandler,
    ProposalSnapshotHandler,
    _object_in_scope,
)
from platform_core.contracts.aiops import (
    MeasurementSemantics,
    SufficiencyStatus,
)
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.change import ActionVerification
from platform_core.identity import uuid7


class ActionCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ActionRegistry.load()

    def test_registry_has_exact_oracle_and_mysql_variants(self) -> None:
        self.assertEqual(len(self.registry.templates), 55)
        modes = {
            item.definition.action_template_id: item.definition.execution_mode
            for item in self.registry.templates
        }
        self.assertEqual("MANUAL_ONLY", modes["db.table.drop"])
        self.assertEqual("MANUAL_ONLY", modes["db.table.truncate"])
        self.assertEqual("MANUAL_ONLY", modes["db.archive.cleanup"])
        self.assertEqual("MANUAL_ONLY", modes["db.recovery.recover"])
        self.assertEqual("MANUAL_ONLY", modes["db.backup.delete"])
        self.assertEqual("MANUAL_ONLY", modes["db.ha.failover"])
        self.assertEqual("UNSUPPORTED", modes["db.listener.stop"])
        self.assertEqual("UNSUPPORTED", modes["db.database.upgrade"])
        oracle = self.registry.resolve(
            action_template_id="db.session.terminate",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"session_management"},
            entitlements=set(),
            environment="PROD",
        )
        rendered = ActionRenderer().render(
            oracle,
            {"session_id": 42, "serial_number": 9, "instance_id": 1},
        )
        self.assertEqual(
            rendered.command_text,
            "ALTER SYSTEM DISCONNECT SESSION '42,9,@1' IMMEDIATE",
        )
        manual = self.registry.resolve(
            action_template_id="db.table.truncate",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities=set(),
            entitlements=set(),
            environment="PROD",
        )
        manual_rendered = ActionRenderer().render(
            manual,
            {
                "table_ref": {
                    "schema": "APP",
                    "object_type": "TABLE",
                    "object_name": "ORDERS_STAGING",
                }
            },
        )
        self.assertEqual("MANUAL_ONLY", manual_rendered.execution_mode)
        self.assertEqual("NONE", manual_rendered.executor_kind)
        self.assertEqual(
            'TRUNCATE TABLE "APP"."ORDERS_STAGING"',
            manual_rendered.command_text,
        )

    def test_destructive_recovery_commands_are_manual_only(self) -> None:
        expected = {
            "db.recovery.restore": "RESTORE DATABASE",
            "db.recovery.recover": "RECOVER DATABASE",
            "db.backup.delete": "DELETE NOPROMPT BACKUP",
            "db.ha.failover": (
                "ALTER DATABASE ACTIVATE PHYSICAL STANDBY DATABASE"
            ),
        }
        for action_id, command in expected.items():
            with self.subTest(action_id=action_id):
                template = self.registry.resolve(
                    action_template_id=action_id,
                    version="1.0.0",
                    db_type="ORACLE",
                    db_version="19.0.0",
                    capabilities=set(),
                    entitlements=set(),
                    environment="PROD",
                )
                rendered = ActionRenderer().render(template, {})
                self.assertEqual("MANUAL_ONLY", rendered.execution_mode)
                self.assertEqual("NONE", rendered.executor_kind)
                self.assertEqual(command, rendered.command_text)

    def test_planned_inventory_is_visible_but_never_resolvable(self) -> None:
        visible = self.registry.compatible(
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities=set(),
            entitlements=set(),
            environment="PROD",
        )
        planned = {
            item.definition.action_template_id
            for item in visible
            if item.definition.status == "PLANNED"
        }
        self.assertTrue(
            {
                "db.storage.datafile.add",
                "db.backup.start",
                "db.ha.log_apply.start",
                "db.pdb.open",
                "db.listener.stop",
                "db.patch.apply",
            }
            <= planned
        )
        with self.assertRaises(LookupError):
            self.registry.resolve(
                action_template_id="db.patch.apply",
                version="1.0.0",
                db_type="ORACLE",
                db_version="19.0.0",
                capabilities=set(),
                entitlements=set(),
                environment="PROD",
            )

    def test_oracle_partition_rebuild_uses_quoted_verified_identifiers(
        self,
    ) -> None:
        template = self.registry.resolve(
            action_template_id="db.index.partition.rebuild",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dba_catalog_views", "index_maintenance"},
            entitlements=set(),
            environment="PROD",
        )
        rendered = ActionRenderer().render(
            template,
            {
                "index_ref": {
                    "schema": "APP",
                    "object_type": "INDEX",
                    "object_name": "IX_ORDERS",
                    "partition": "P_202609",
                },
                "partition_name": "P_202609",
                "online": True,
            },
        )
        self.assertEqual(
            'ALTER INDEX "APP"."IX_ORDERS" REBUILD PARTITION "P_202609" ONLINE',
            rendered.command_text,
        )
        with self.assertRaisesRegex(ValueError, "分区引用"):
            ActionRenderer().render(
                template,
                {
                    "index_ref": {
                        "schema": "APP",
                        "object_type": "INDEX",
                        "object_name": "IX_ORDERS",
                        "partition": "P_202609",
                    },
                    "partition_name": "P_OTHER",
                    "online": True,
                },
            )

    def test_oracle_index_coalesce_uses_exact_verified_object(self) -> None:
        template = self.registry.resolve(
            action_template_id="db.index.coalesce",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dba_catalog_views", "index_maintenance"},
            entitlements=set(),
            environment="PROD",
        )
        rendered = ActionRenderer().render(
            template,
            {
                "index_ref": {
                    "schema": "APP",
                    "object_type": "INDEX",
                    "object_name": "IX_ORDERS",
                }
            },
        )
        self.assertEqual(
            'ALTER INDEX "APP"."IX_ORDERS" COALESCE', rendered.command_text
        )

    def test_oracle_storage_actions_are_growth_only_and_bounded(self) -> None:
        cases = {
            "db.storage.datafile.resize": (
                {"file_name": "+DATA/DB/data01.dbf", "new_size_mb": 2048},
                "ALTER DATABASE DATAFILE '+DATA/DB/data01.dbf' RESIZE 2048M",
            ),
            "db.storage.tempfile.resize": (
                {"file_name": "/u02/oradata/temp01.dbf", "new_size_mb": 4096},
                "ALTER DATABASE TEMPFILE '/u02/oradata/temp01.dbf' RESIZE 4096M",
            ),
            "db.storage.datafile.autoextend": (
                {
                    "file_name": "+DATA/DB/data01.dbf",
                    "next_mb": 128,
                    "max_size_mb": 8192,
                },
                "ALTER DATABASE DATAFILE '+DATA/DB/data01.dbf' AUTOEXTEND "
                "ON NEXT 128M MAXSIZE 8192M",
            ),
            "db.storage.tempfile.autoextend": (
                {
                    "file_name": "/u02/oradata/temp01.dbf",
                    "next_mb": 256,
                    "max_size_mb": 16384,
                },
                "ALTER DATABASE TEMPFILE '/u02/oradata/temp01.dbf' AUTOEXTEND "
                "ON NEXT 256M MAXSIZE 16384M",
            ),
        }
        for action_id, (parameters, command) in cases.items():
            with self.subTest(action_id=action_id):
                template = self.registry.resolve(
                    action_template_id=action_id,
                    version="1.0.0",
                    db_type="ORACLE",
                    db_version="19.0.0",
                    capabilities={"dba_catalog_views"},
                    entitlements=set(),
                    environment="PROD",
                )
                self.assertEqual(
                    command, ActionRenderer().render(template, parameters).command_text
                )
        template = self.registry.resolve(
            action_template_id="db.storage.datafile.resize",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dba_catalog_views", "dynamic_performance_views"},
            entitlements=set(),
            environment="PROD",
        )
        with self.assertRaises(ValueError):
            ActionRenderer().render(
                template,
                {"file_name": "x.dbf'; DROP TABLE T;--", "new_size_mb": 2048},
            )

    def test_parameter_resource_and_privilege_commands_are_exact(self) -> None:
        parameter = self.registry.resolve(
            action_template_id="db.parameter.set",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dba_catalog_views", "dynamic_performance_views"},
            entitlements=set(),
            environment="PROD",
        )
        self.assertEqual(
            "ALTER SYSTEM SET cursor_sharing = FORCE SCOPE=MEMORY",
            ActionRenderer().render(
                parameter,
                {"parameter_name": "cursor_sharing", "parameter_value": "FORCE"},
            ).command_text,
        )
        with self.assertRaises(ValueError):
            ActionRenderer().render(
                parameter,
                {"parameter_name": "cursor_sharing", "parameter_value": "ALL"},
            )

        resource_plan = self.registry.resolve(
            action_template_id="db.resource_manager.plan.switch",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dba_catalog_views", "dynamic_performance_views"},
            entitlements=set(),
            environment="PROD",
        )
        self.assertEqual(
            "ALTER SYSTEM SET RESOURCE_MANAGER_PLAN = 'APP_PLAN' SCOPE=BOTH",
            ActionRenderer().render(
                resource_plan, {"resource_plan_name": "APP_PLAN"}
            ).command_text,
        )

        templates = {
            item.definition.variant: item
            for item in self.registry.templates
            if item.definition.action_template_id == "db.user.privilege.grant"
        }
        system_grant = ActionRenderer().render(
            templates["oracle_registered_system_privilege"],
            {"grantee_name": "APPUSER", "privilege": "CREATE SESSION"},
        )
        self.assertEqual('GRANT CREATE SESSION TO "APPUSER"', system_grant.command_text)
        object_grant = ActionRenderer().render(
            templates["oracle_registered_object_privilege"],
            {
                "privilege": "SELECT",
                "object_ref": {
                    "schema": "APP",
                    "object_type": "TABLE",
                    "object_name": "ORDERS",
                },
                "grantee_name": "REPORTER",
            },
        )
        self.assertEqual(
            'GRANT SELECT ON "APP"."ORDERS" TO "REPORTER"',
            object_grant.command_text,
        )
        with self.assertRaises(LookupError):
            self.registry.resolve(
                action_template_id="db.user.privilege.grant",
                version="1.0.0",
                db_type="ORACLE",
                db_version="19.0.0",
                capabilities={"dba_catalog_views"},
                entitlements=set(),
                environment="PROD",
            )
        resolved = self.registry.resolve(
            action_template_id="db.user.privilege.grant",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dba_catalog_views"},
            entitlements=set(),
            environment="PROD",
            template_hash=templates[
                "oracle_registered_object_privilege"
            ].template_hash,
        )
        self.assertEqual("oracle_registered_object_privilege", resolved.definition.variant)

    def test_sensitive_action_targets_require_agent_registration(self) -> None:
        policy = {
            "object_scopes": {
                "schemas": ["APP"],
                "dynamic_parameters": [
                    {
                        "name": "cursor_sharing",
                        "allowed_values": ["EXACT", "FORCE"],
                    }
                ],
                "resource_manager_plans": ["APP_PLAN"],
                "privilege_grantees": ["REPORTER"],
                "system_privileges": ["CREATE SESSION"],
                "object_privileges": ["SELECT"],
            }
        }
        self.assertTrue(
            _object_in_scope(
                {
                    "parameter_name": "cursor_sharing",
                    "parameter_value": "FORCE",
                },
                policy,
            )
        )
        self.assertFalse(
            _object_in_scope(
                {
                    "parameter_name": "statistics_level",
                    "parameter_value": "ALL",
                },
                policy,
            )
        )
        self.assertTrue(
            _object_in_scope(
                {
                    "privilege": "SELECT",
                    "object_ref": {
                        "schema": "APP",
                        "object_type": "TABLE",
                        "object_name": "ORDERS",
                    },
                    "grantee_name": "REPORTER",
                },
                policy,
            )
        )
        self.assertFalse(
            _object_in_scope(
                {"grantee_name": "REPORTER", "privilege": "CREATE TABLE"},
                policy,
            )
        )

    def test_oracle_cancel_sql_has_exact_typed_command(self) -> None:
        template = self.registry.resolve(
            action_template_id="db.session.cancel_sql",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dynamic_performance_views", "session_management"},
            entitlements=set(),
            environment="PROD",
        )
        rendered = ActionRenderer().render(
            template,
            {
                "session_id": 42,
                "serial_number": 9,
                "instance_id": 1,
                "sql_id": "0abc123def456",
            },
        )
        self.assertEqual(
            "ALTER SYSTEM CANCEL SQL '42,9,@1,0abc123def456' IMMEDIATE",
            rendered.command_text,
        )

    def test_oracle_compile_uses_matching_typed_object_reference(self) -> None:
        template = self.registry.resolve(
            action_template_id="db.object.compile",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dba_catalog_views"},
            entitlements=set(),
            environment="PROD",
        )
        parameters = {
            "object_type": "PROCEDURE",
            "object_ref": {
                "schema": "APP",
                "object_type": "PROCEDURE",
                "object_name": "PROC_A",
            },
        }

        rendered = ActionRenderer().render(template, parameters)

        self.assertEqual(
            'ALTER PROCEDURE "APP"."PROC_A" COMPILE',
            rendered.command_text,
        )
        parameters["object_type"] = "FUNCTION"
        with self.assertRaisesRegex(ValueError, "类型.*不一致"):
            ActionRenderer().render(template, parameters)

    def test_oracle_compile_supports_schema_object_kinds(self) -> None:
        template = self.registry.resolve(
            action_template_id="db.object.compile",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dba_catalog_views"},
            entitlements=set(),
            environment="PROD",
        )
        expected = {
            "VIEW": 'ALTER VIEW "APP"."OBJECT_A" COMPILE',
            "TRIGGER": 'ALTER TRIGGER "APP"."OBJECT_A" COMPILE',
            "TYPE": 'ALTER TYPE "APP"."OBJECT_A" COMPILE',
            "PACKAGE BODY": 'ALTER PACKAGE "APP"."OBJECT_A" COMPILE BODY',
            "TYPE BODY": 'ALTER TYPE "APP"."OBJECT_A" COMPILE BODY',
        }
        for object_type, command in expected.items():
            with self.subTest(object_type=object_type):
                rendered = ActionRenderer().render(
                    template,
                    {
                        "object_type": object_type,
                        "object_ref": {
                            "schema": "APP",
                            "object_type": object_type,
                            "object_name": "OBJECT_A",
                        },
                    },
                )
                self.assertEqual(command, rendered.command_text)

    def test_oracle_table_statistics_uses_fixed_gather_strategy(self) -> None:
        template = self.registry.resolve(
            action_template_id="db.statistics.gather",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dba_catalog_views"},
            entitlements=set(),
            environment="PROD",
        )
        parameters = {
            "table_ref": {
                "schema": "APP",
                "object_type": "TABLE",
                "object_name": "ORDERS",
            }
        }

        rendered = ActionRenderer().render(template, parameters)

        self.assertEqual(
            "BEGIN DBMS_STATS.GATHER_TABLE_STATS(ownname => 'APP', "
            "tabname => 'ORDERS', estimate_percent => "
            "DBMS_STATS.AUTO_SAMPLE_SIZE, method_opt => "
            "'FOR ALL COLUMNS SIZE AUTO', cascade => TRUE, "
            "no_invalidate => DBMS_STATS.AUTO_INVALIDATE); END;",
            rendered.command_text,
        )
        parameters["table_ref"]["schema"] = "APP'); DELETE FROM USERS;--"
        with self.assertRaisesRegex(ValueError, "标识符无效"):
            ActionRenderer().render(template, parameters)

    def test_oracle_table_statistics_lock_commands_are_exact(self) -> None:
        expected = {
            "db.statistics.lock": (
                "BEGIN DBMS_STATS.LOCK_TABLE_STATS(ownname => 'APP', "
                "tabname => 'ORDERS'); END;"
            ),
            "db.statistics.unlock": (
                "BEGIN DBMS_STATS.UNLOCK_TABLE_STATS(ownname => 'APP', "
                "tabname => 'ORDERS'); END;"
            ),
        }
        for action_id, command in expected.items():
            with self.subTest(action_id=action_id):
                template = self.registry.resolve(
                    action_template_id=action_id,
                    version="1.0.0",
                    db_type="ORACLE",
                    db_version="19.0.0",
                    capabilities={"dba_catalog_views"},
                    entitlements=set(),
                    environment="PROD",
                )
                rendered = ActionRenderer().render(
                    template,
                    {
                        "table_ref": {
                            "schema": "APP",
                            "object_type": "TABLE",
                            "object_name": "ORDERS",
                        }
                    },
                )
                self.assertEqual(command, rendered.command_text)

    def test_oracle_scheduler_run_binds_exact_registered_job(self) -> None:
        template = self.registry.resolve(
            action_template_id="db.scheduler.job.run",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"dba_catalog_views"},
            entitlements=set(),
            environment="PROD",
        )

        rendered = ActionRenderer().render(
            template,
            {
                "job_ref": {
                    "schema": "APP",
                    "object_type": "SCHEDULER_JOB",
                    "object_name": "NIGHTLY_JOB",
                },
                "previous_run_count": 7,
                "previous_failure_count": 1,
            },
        )

        self.assertEqual(
            "BEGIN DBMS_SCHEDULER.RUN_JOB(job_name => "
            "'\"APP\".\"NIGHTLY_JOB\"', use_current_session => FALSE); END;",
            rendered.command_text,
        )

    def test_oracle_scheduler_state_commands_are_exact(self) -> None:
        expected = {
            "db.scheduler.job.enable": (
                "BEGIN DBMS_SCHEDULER.ENABLE(name => "
                "'\"APP\".\"NIGHTLY_JOB\"'); END;"
            ),
            "db.scheduler.job.disable": (
                "BEGIN DBMS_SCHEDULER.DISABLE(name => "
                "'\"APP\".\"NIGHTLY_JOB\"', force => FALSE); END;"
            ),
            "db.scheduler.job.stop": (
                "BEGIN DBMS_SCHEDULER.STOP_JOB(job_name => "
                "'\"APP\".\"NIGHTLY_JOB\"', force => FALSE); END;"
            ),
        }
        for action_id, command in expected.items():
            with self.subTest(action_id=action_id):
                template = self.registry.resolve(
                    action_template_id=action_id,
                    version="1.0.0",
                    db_type="ORACLE",
                    db_version="19.0.0",
                    capabilities={"dba_catalog_views"},
                    entitlements=set(),
                    environment="PROD",
                )
                rendered = ActionRenderer().render(
                    template,
                    {
                        "job_ref": {
                            "schema": "APP",
                            "object_type": "SCHEDULER_JOB",
                            "object_name": "NIGHTLY_JOB",
                        }
                    },
                )
                self.assertEqual(command, rendered.command_text)

    def test_oracle_user_state_commands_are_exact(self) -> None:
        expected = {
            "db.user.lock": 'ALTER USER "APPUSER" ACCOUNT LOCK',
            "db.user.unlock": 'ALTER USER "APPUSER" ACCOUNT UNLOCK',
            "db.user.password.expire": (
                'ALTER USER "APPUSER" PASSWORD EXPIRE'
            ),
        }
        for action_id, command in expected.items():
            with self.subTest(action_id=action_id):
                template = self.registry.resolve(
                    action_template_id=action_id,
                    version="1.0.0",
                    db_type="ORACLE",
                    db_version="19.0.0",
                    capabilities={"dba_catalog_views"},
                    entitlements=set(),
                    environment="PROD",
                )
                rendered = ActionRenderer().render(
                    template,
                    {
                        "user_ref": {
                            "schema": "APPUSER",
                            "object_type": "USER",
                            "object_name": "APPUSER",
                        }
                    },
                )
                self.assertEqual(command, rendered.command_text)


class OracleIndexActionCompilerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ActionRegistry.load()

    def _assessment(self, *, tool_id: str, columns: tuple[str, ...], row: tuple):
        return DbaSufficiencyAssessment(
            status=SufficiencyStatus.ANSWERABLE,
            evidence=(
                TurnEvidenceFact(
                    evidence_ref="artifact:index#row-0",
                    artifact_id="index-artifact",
                    source_id="oracle.index",
                    step_id="index-health",
                    tool_id=tool_id,
                    trust_level="SOURCE_VERIFIED",
                    measurement_semantics=MeasurementSemantics.CURRENT_ACTIVITY,
                    presentation_kind="TABLE",
                    captured_at="2026-09-02T00:00:00+00:00",
                    columns=tuple({"name": name} for name in columns),
                    rows=(row,),
                    row_count=1,
                ),
            ),
        )

    def test_partition_rebuild_requires_verified_space_and_lock_context(self):
        columns = (
            "owner",
            "index_name",
            "partition_name",
            "status",
            "partitioned",
            "index_type",
            "space_sufficient",
            "online_supported",
            "active_table_locks",
        )
        compiler = ActionCompilerRegistry()
        compiled = compiler.compile_turn(
            compiler_id="oracle-index-partition-rebuild.v1",
            assessment=self._assessment(
                tool_id="db.index.partition.health",
                columns=columns,
                row=(
                    "APP",
                    "IX_ORDERS",
                    "P_202609",
                    "UNUSABLE",
                    "YES",
                    "NORMAL",
                    "YES",
                    "YES",
                    2,
                ),
            ),
            db_type="ORACLE",
        )
        self.assertIsNotNone(compiled)
        self.assertEqual(
            compiled.parameters["index_ref"]["partition"], "P_202609"
        )
        self.assertTrue(compiled.parameters["online"])

        insufficient = compiler.compile_turn(
            compiler_id="oracle-index-partition-rebuild.v1",
            assessment=self._assessment(
                tool_id="db.index.partition.health",
                columns=columns,
                row=(
                    "APP",
                    "IX_ORDERS",
                    "P_202609",
                    "UNUSABLE",
                    "YES",
                    "NORMAL",
                    "NO",
                    "YES",
                    0,
                ),
            ),
            db_type="ORACLE",
        )
        self.assertIsNone(insufficient)

    def test_index_coalesce_requires_valid_unlocked_normal_index(self):
        columns = (
            "owner",
            "index_name",
            "status",
            "partitioned",
            "index_type",
            "active_table_locks",
        )
        compiler = ActionCompilerRegistry()
        compiled = compiler.compile_turn(
            compiler_id="oracle-index-coalesce.v1",
            assessment=self._assessment(
                tool_id="db.index.coalesce_candidate",
                columns=columns,
                row=("APP", "IX_ORDERS", "VALID", "NO", "NORMAL", 0),
            ),
            db_type="ORACLE",
        )
        self.assertIsNotNone(compiled)
        self.assertEqual(
            "IX_ORDERS", compiled.parameters["index_ref"]["object_name"]
        )
        blocked = compiler.compile_turn(
            compiler_id="oracle-index-coalesce.v1",
            assessment=self._assessment(
                tool_id="db.index.coalesce_candidate",
                columns=columns,
                row=("APP", "IX_ORDERS", "VALID", "NO", "NORMAL", 1),
            ),
            db_type="ORACLE",
        )
        self.assertIsNone(blocked)

    def test_storage_compilers_reject_shrink_and_unbounded_autoextend(self):
        columns = (
            "file_name",
            "current_size_mb",
            "current_max_size_mb",
            "autoextensible",
            "current_next_mb",
            "requested_size_mb",
            "requested_next_mb",
            "requested_max_size_mb",
            "status",
            "online_status",
        )
        compiler = ActionCompilerRegistry()
        grow = compiler.compile_turn(
            compiler_id="oracle-datafile-resize.v1",
            assessment=self._assessment(
                tool_id="db.storage.datafile.action_state",
                columns=columns,
                row=("+DATA/DB/data01.dbf", 1024, 1024, "NO", 0, 2048, 0, 0, "AVAILABLE", "ONLINE"),
            ),
            db_type="ORACLE",
        )
        self.assertEqual(2048, grow.parameters["new_size_mb"])
        shrink = compiler.compile_turn(
            compiler_id="oracle-datafile-resize.v1",
            assessment=self._assessment(
                tool_id="db.storage.datafile.action_state",
                columns=columns,
                row=(
                    "+DATA/DB/data01.dbf",
                    1024,
                    1024,
                    "NO",
                    0,
                    512,
                    0,
                    0,
                    "AVAILABLE",
                    "ONLINE",
                ),
            ),
            db_type="ORACLE",
        )
        self.assertIsNone(shrink)
        bounded = compiler.compile_turn(
            compiler_id="oracle-datafile-autoextend.v1",
            assessment=self._assessment(
                tool_id="db.storage.datafile.action_state",
                columns=columns,
                row=(
                    "+DATA/DB/data01.dbf",
                    1024,
                    1024,
                    "NO",
                    0,
                    0,
                    128,
                    4096,
                    "AVAILABLE",
                    "ONLINE",
                ),
            ),
            db_type="ORACLE",
        )
        self.assertEqual(4096, bounded.parameters["max_size_mb"])
        unbounded = compiler.compile_turn(
            compiler_id="oracle-datafile-autoextend.v1",
            assessment=self._assessment(
                tool_id="db.storage.datafile.action_state",
                columns=columns,
                row=(
                    "+DATA/DB/data01.dbf",
                    1024,
                    1024,
                    "NO",
                    0,
                    0,
                    128,
                    0,
                    "AVAILABLE",
                    "ONLINE",
                ),
            ),
            db_type="ORACLE",
        )
        self.assertIsNone(unbounded)

    def test_parameter_resource_and_privilege_compilers_use_verified_state(self):
        compiler = ActionCompilerRegistry()
        parameter = compiler.compile_turn(
            compiler_id="oracle-dynamic-parameter-set.v1",
            assessment=self._assessment(
                tool_id="db.parameter.dynamic_state",
                columns=(
                    "parameter_name",
                    "current_value",
                    "issys_modifiable",
                    "requested_value",
                ),
                row=("cursor_sharing", "EXACT", "IMMEDIATE", "FORCE"),
            ),
            db_type="ORACLE",
        )
        self.assertEqual(
            {"parameter_name": "cursor_sharing", "parameter_value": "FORCE"},
            parameter.parameters,
        )
        invalid_pair = compiler.compile_turn(
            compiler_id="oracle-dynamic-parameter-set.v1",
            assessment=self._assessment(
                tool_id="db.parameter.dynamic_state",
                columns=(
                    "parameter_name",
                    "current_value",
                    "issys_modifiable",
                    "requested_value",
                ),
                row=("cursor_sharing", "EXACT", "IMMEDIATE", "ALL"),
            ),
            db_type="ORACLE",
        )
        self.assertIsNone(invalid_pair)
        resource_plan = compiler.compile_turn(
            compiler_id="oracle-resource-manager-plan-switch.v1",
            assessment=self._assessment(
                tool_id="db.resource_manager.plan_state",
                columns=("resource_plan_name", "status", "current_plan_name"),
                row=("APP_PLAN", "ACTIVE", "OLD_PLAN"),
            ),
            db_type="ORACLE",
        )
        self.assertEqual("APP_PLAN", resource_plan.parameters["resource_plan_name"])
        system_grant = compiler.compile_turn(
            compiler_id="oracle-system-privilege-grant.v1",
            assessment=self._assessment(
                tool_id="db.user.system_privilege_state",
                columns=(
                    "grantee_name",
                    "privilege",
                    "is_granted",
                    "oracle_maintained",
                    "common",
                ),
                row=("APPUSER", "CREATE SESSION", "NO", "N", "NO"),
            ),
            db_type="ORACLE",
        )
        self.assertEqual("CREATE SESSION", system_grant.parameters["privilege"])
        object_revoke = compiler.compile_turn(
            compiler_id="oracle-object-privilege-revoke.v1",
            assessment=self._assessment(
                tool_id="db.user.object_privilege_state",
                columns=(
                    "owner",
                    "object_name",
                    "object_type",
                    "grantee_name",
                    "privilege",
                    "is_granted",
                    "oracle_maintained",
                    "common",
                ),
                row=("APP", "ORDERS", "TABLE", "REPORTER", "SELECT", "YES", "N", "NO"),
            ),
            db_type="ORACLE",
        )
        self.assertEqual("ORDERS", object_revoke.parameters["object_ref"]["object_name"])

    def test_offline_rebuild_is_not_compiled_while_table_is_locked(self):
        columns = (
            "owner",
            "index_name",
            "status",
            "partitioned",
            "index_type",
            "space_sufficient",
            "online_supported",
            "active_table_locks",
        )
        compiled = ActionCompilerRegistry().compile_turn(
            compiler_id="oracle-index-rebuild.v1",
            assessment=self._assessment(
                tool_id="db.index.health",
                columns=columns,
                row=(
                    "APP",
                    "IX_ORDERS",
                    "UNUSABLE",
                    "NO",
                    "NORMAL",
                    "YES",
                    "NO",
                    1,
                ),
            ),
            db_type="ORACLE",
        )
        self.assertIsNone(compiled)

    def test_cancel_sql_compiler_only_consumes_dedicated_verified_fact(self):
        columns = (
            "instance_id",
            "session_id",
            "serial_number",
            "sql_id",
            "status",
        )
        compiled = ActionCompilerRegistry().compile_turn(
            compiler_id="oracle-session-cancel-sql.v1",
            assessment=self._assessment(
                tool_id="db.session.current_sql",
                columns=columns,
                row=(1, 42, 9, "0abc123def456", "ACTIVE"),
            ),
            db_type="ORACLE",
        )
        self.assertEqual(
            {
                "session_id": 42,
                "serial_number": 9,
                "instance_id": 1,
                "sql_id": "0abc123def456",
            },
            compiled.parameters,
        )

    def test_object_compile_requires_verified_invalid_plsql_object(self):
        compiler = ActionCompilerRegistry()
        columns = (
            "owner",
            "object_name",
            "object_type",
            "status",
            "last_ddl_time",
        )
        compiled = compiler.compile_turn(
            compiler_id="oracle-object-compile.v1",
            assessment=self._assessment(
                tool_id="db.object.status",
                columns=columns,
                row=("APP", "PROC_A", "PROCEDURE", "INVALID", None),
            ),
            db_type="ORACLE",
        )

        self.assertIsNotNone(compiled)
        self.assertEqual(
            {
                "object_type": "PROCEDURE",
                "object_ref": {
                    "schema": "APP",
                    "object_type": "PROCEDURE",
                    "object_name": "PROC_A",
                },
            },
            compiled.parameters,
        )
        already_valid = compiler.compile_turn(
            compiler_id="oracle-object-compile.v1",
            assessment=self._assessment(
                tool_id="db.object.status",
                columns=columns,
                row=("APP", "PROC_A", "PROCEDURE", "VALID", None),
            ),
            db_type="ORACLE",
        )
        self.assertIsNone(already_valid)

    def test_statistics_gather_requires_stale_unlocked_regular_table(self):
        compiler = ActionCompilerRegistry()
        columns = (
            "owner",
            "table_name",
            "partitioned",
            "temporary",
            "last_analyzed",
            "stale_stats",
            "stattype_locked",
        )
        compiled = compiler.compile_turn(
            compiler_id="oracle-table-statistics-gather.v1",
            assessment=self._assessment(
                tool_id="db.table.statistics",
                columns=columns,
                row=(
                    "APP",
                    "ORDERS",
                    "NO",
                    "N",
                    "2026-08-01T00:00:00+00:00",
                    "YES",
                    None,
                ),
            ),
            db_type="ORACLE",
        )

        self.assertIsNotNone(compiled)
        self.assertEqual(
            {
                "table_ref": {
                    "schema": "APP",
                    "object_type": "TABLE",
                    "object_name": "ORDERS",
                }
            },
            compiled.parameters,
        )
        locked = compiler.compile_turn(
            compiler_id="oracle-table-statistics-gather.v1",
            assessment=self._assessment(
                tool_id="db.table.statistics",
                columns=columns,
                row=(
                    "APP",
                    "ORDERS",
                    "NO",
                    "N",
                    "2026-08-01T00:00:00+00:00",
                    "YES",
                    "ALL",
                ),
            ),
            db_type="ORACLE",
        )
        self.assertIsNone(locked)

    def test_statistics_lock_compilers_use_action_specific_facts(self):
        columns = (
            "owner",
            "table_name",
            "partitioned",
            "temporary",
            "last_analyzed",
            "stale_stats",
            "stattype_locked",
        )
        cases = (
            (
                "oracle-table-statistics-lock.v1",
                "db.table.statistics.lock_candidate",
                "2026-09-02T00:00:00Z",
                None,
            ),
            (
                "oracle-table-statistics-unlock.v1",
                "db.table.statistics.unlock_candidate",
                "2026-09-02T00:00:00Z",
                "ALL",
            ),
        )
        for compiler_id, tool_id, last_analyzed, locked in cases:
            with self.subTest(compiler_id=compiler_id):
                compiled = ActionCompilerRegistry().compile_turn(
                    compiler_id=compiler_id,
                    assessment=self._assessment(
                        tool_id=tool_id,
                        columns=columns,
                        row=(
                            "APP",
                            "ORDERS",
                            "NO",
                            "N",
                            last_analyzed,
                            "NO",
                            locked,
                        ),
                    ),
                    db_type="ORACLE",
                )
                self.assertIsNotNone(compiled)

    def test_scheduler_run_requires_enabled_scheduled_job(self):
        compiler = ActionCompilerRegistry()
        columns = (
            "owner",
            "job_name",
            "enabled",
            "state",
            "last_start_date",
            "last_run_duration",
            "run_count",
            "failure_count",
        )
        compiled = compiler.compile_turn(
            compiler_id="oracle-scheduler-job-run.v1",
            assessment=self._assessment(
                tool_id="db.scheduler.job.status",
                columns=columns,
                row=(
                    "APP",
                    "NIGHTLY_JOB",
                    "TRUE",
                    "SCHEDULED",
                    None,
                    None,
                    7,
                    1,
                ),
            ),
            db_type="ORACLE",
        )

        self.assertIsNotNone(compiled)
        self.assertEqual(7, compiled.parameters["previous_run_count"])
        self.assertEqual(1, compiled.parameters["previous_failure_count"])
        self.assertEqual(
            "SCHEDULER_JOB",
            compiled.parameters["job_ref"]["object_type"],
        )
        running = compiler.compile_turn(
            compiler_id="oracle-scheduler-job-run.v1",
            assessment=self._assessment(
                tool_id="db.scheduler.job.status",
                columns=columns,
                row=(
                    "APP",
                    "NIGHTLY_JOB",
                    "TRUE",
                    "RUNNING",
                    None,
                    None,
                    7,
                    1,
                ),
            ),
            db_type="ORACLE",
        )
        self.assertIsNone(running)

    def test_scheduler_state_compilers_use_action_specific_facts(self):
        columns = (
            "owner",
            "job_name",
            "enabled",
            "state",
            "last_start_date",
            "last_run_duration",
            "run_count",
            "failure_count",
        )
        cases = (
            (
                "oracle-scheduler-job-enable.v1",
                "db.scheduler.job.enable_candidate",
                "FALSE",
                "DISABLED",
            ),
            (
                "oracle-scheduler-job-disable.v1",
                "db.scheduler.job.disable_candidate",
                "TRUE",
                "SCHEDULED",
            ),
            (
                "oracle-scheduler-job-stop.v1",
                "db.scheduler.job.stop_candidate",
                "TRUE",
                "RUNNING",
            ),
        )
        for compiler_id, tool_id, enabled, state in cases:
            with self.subTest(compiler_id=compiler_id):
                compiled = ActionCompilerRegistry().compile_turn(
                    compiler_id=compiler_id,
                    assessment=self._assessment(
                        tool_id=tool_id,
                        columns=columns,
                        row=(
                            "APP",
                            "NIGHTLY_JOB",
                            enabled,
                            state,
                            None,
                            None,
                            0,
                            0,
                        ),
                    ),
                    db_type="ORACLE",
                )
                self.assertIsNotNone(compiled)
                self.assertEqual(
                    "NIGHTLY_JOB",
                    compiled.parameters["job_ref"]["object_name"],
                )

    def test_user_state_compilers_exclude_system_and_wrong_state(self):
        columns = (
            "username",
            "account_status",
            "lock_date",
            "expiry_date",
            "profile",
            "authentication_type",
            "oracle_maintained",
            "common",
        )
        cases = (
            (
                "oracle-user-lock.v1",
                "db.user.lock_candidate",
                "OPEN",
            ),
            (
                "oracle-user-unlock.v1",
                "db.user.unlock_candidate",
                "LOCKED(TIMED)",
            ),
        )
        for compiler_id, tool_id, status in cases:
            with self.subTest(compiler_id=compiler_id):
                compiled = ActionCompilerRegistry().compile_turn(
                    compiler_id=compiler_id,
                    assessment=self._assessment(
                        tool_id=tool_id,
                        columns=columns,
                        row=(
                            "APPUSER",
                            status,
                            None,
                            None,
                            "DEFAULT",
                            "PASSWORD",
                            "N",
                            "NO",
                        ),
                    ),
                    db_type="ORACLE",
                )
                self.assertIsNotNone(compiled)
                self.assertEqual(
                    "APPUSER", compiled.parameters["user_ref"]["object_name"]
                )

        system_user = ActionCompilerRegistry().compile_turn(
            compiler_id="oracle-user-lock.v1",
            assessment=self._assessment(
                tool_id="db.user.lock_candidate",
                columns=columns,
                row=(
                    "SYS",
                    "OPEN",
                    None,
                    None,
                    "DEFAULT",
                    "PASSWORD",
                    "Y",
                    "YES",
                ),
            ),
            db_type="ORACLE",
        )
        self.assertIsNone(system_user)

        password_expire = ActionCompilerRegistry().compile_turn(
            compiler_id="oracle-user-password-expire.v1",
            assessment=self._assessment(
                tool_id="db.user.password_expire_candidate",
                columns=columns,
                row=(
                    "APPUSER",
                    "OPEN",
                    None,
                    None,
                    "DEFAULT",
                    "PASSWORD",
                    "N",
                    "NO",
                ),
            ),
            db_type="ORACLE",
        )
        self.assertIsNotNone(password_expire)

    def test_renderer_rejects_unregistered_command_shape(self) -> None:
        oracle = self.registry.resolve(
            action_template_id="db.session.terminate",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"session_management"},
            entitlements=set(),
            environment="PROD",
        )
        with self.assertRaisesRegex(ValueError, "Allowlist"):
            validate_rendered_action(
                "ALTER SYSTEM SET open_cursors=999",
                definition=oracle.definition,
            )


class ActionPlanHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ActionRegistry.load()

    def test_source_verified_blocker_becomes_advisory_plan(self) -> None:
        context = self._context()
        plan = asyncio.run(
            ActionPlanHandler(
                registry=self.registry,
                execution_enabled=False,
            ).execute(context)
        )
        self.assertEqual(plan.decision, "ADVISORY")
        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(
            plan.actions[0].canonical_parameters,
            {"session_id": 42, "serial_number": 9, "instance_id": 1},
        )
        proposal = asyncio.run(
            ProposalSnapshotHandler().execute(
                context.__class__(
                    **{
                        **context.__dict__,
                        "task_id": "proposal-task",
                        "task_key": "change:proposal",
                        "input_artifacts": (
                            {
                                "schema_version": "ACTION_PLAN.v1",
                                "payload": plan.model_dump(mode="json"),
                            },
                        ),
                    }
                )
            )
        )
        self.assertEqual(proposal.status, "CREATED")
        self.assertEqual(proposal.proposal.mode, "ADVISORY")

    def test_user_only_result_cannot_supply_action_parameters(self) -> None:
        context = self._context(user_only=True)
        plan = asyncio.run(
            ActionPlanHandler(
                registry=self.registry,
                execution_enabled=True,
            ).execute(context)
        )
        self.assertEqual(plan.decision, "NO_ACTION")
        self.assertIn(
            "VERIFIED_ACTION_PARAMETERS_UNAVAILABLE",
            plan.decision_reasons,
        )

    def test_automatic_diagnosis_never_creates_executable_action(self) -> None:
        context = self._context()
        context = context.__class__(
            **{**context.__dict__, "trigger_type": "ALERT"}
        )
        plan = asyncio.run(
            ActionPlanHandler(
                registry=self.registry,
                execution_enabled=True,
            ).execute(context)
        )
        self.assertEqual("ADVISORY", plan.decision)
        self.assertEqual("ADVISORY", plan.actions[0].mode)

    def _context(self, *, user_only: bool = False) -> TaskExecutionContext:
        trust = "USER_PROVIDED" if user_only else "SOURCE_VERIFIED"
        blocker = EvidenceFact(
            fact_id="1" * 64,
            source_artifact_id="artifact-blocking",
            source_json_pointer="/rows/0",
            source_type=(
                "USER_RESULT" if user_only else "DATABASE_OBSERVATION"
            ),
            source_group_id="database:blocking",
            trust_level=trust,
            target_id="target-1",
            observed_subject="target-1",
            metric_or_fact_type="db.session.blocking_chain",
            value={
                "blocking_session_id": 42,
                "blocking_instance_id": 1,
            },
            fact_summary="会话 42 正在阻塞其他会话",
        )
        active = EvidenceFact(
            fact_id="2" * 64,
            source_artifact_id="artifact-active",
            source_json_pointer="/rows/0",
            source_type=(
                "USER_RESULT" if user_only else "DATABASE_OBSERVATION"
            ),
            source_group_id="database:active",
            trust_level=trust,
            target_id="target-1",
            observed_subject="target-1",
            metric_or_fact_type="db.session.active",
            value={
                "session_id": 42,
                "serial_number": 9,
                "instance_id": 1,
            },
            fact_summary="活动会话 42",
        )
        evidence = EvidenceIndex(
            target_id="target-1",
            facts=(blocker, active),
            fact_count=2,
            source_group_count=2,
            index_hash="3" * 64,
        )
        root = RootCauseAssessment(
            target_id="target-1",
            suggested_level="PROBABLE",
            eligible_ceiling="PROBABLE",
            effective_level="PROBABLE",
            primary_hypothesis_key="blocking-session",
            supporting_fact_refs=(blocker.fact_id, active.fact_id),
        )
        solution = SolutionDraft(
            immediate_mitigations=("终止已确认的阻塞会话",),
        )
        return TaskExecutionContext(
            run_id="run-1",
            task_id="action-task",
            task_key="change:action-plan",
            target_id="target-1",
            agent_id="agent-1",
            trigger_type="CHAT",
            actor_id="user-1",
            original_request="处理阻塞",
            trace_id="trace-1",
            attempt=1,
            deadline_at=None,
            plan_snapshot={
                "target": {
                    "db_type": "ORACLE",
                    "version_code": "19.0.0",
                    "environment": "PROD",
                    "row_version": 3,
                    "capabilities": {"session_management": True},
                },
                "binding": {
                    "allow_mutation": True,
                    "allowed_actions": ["db.session.terminate"],
                },
            },
            policy_snapshot={
                "rules": {"allow_agent_execution": True}
            },
            input_artifacts=(
                {
                    "schema_version": "EVIDENCE_INDEX.v1",
                    "payload": evidence.model_dump(mode="json"),
                },
                {
                    "schema_version": "ROOT_CAUSE_ASSESSMENT.v1",
                    "payload": root.model_dump(mode="json"),
                },
                {
                    "schema_version": "SOLUTION_DRAFT.v1",
                    "payload": solution.model_dump(mode="json"),
                },
            ),
        )


class ChatActionPlanHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ActionRegistry.load()

    def test_verified_turn_facts_create_approval_action(self) -> None:
        plan = asyncio.run(
            ChatActionPlanHandler(
                registry=self.registry,
                execution_enabled=True,
            ).execute(self._context())
        )
        self.assertEqual(plan.decision, "AGENT_EXECUTE")
        self.assertEqual(
            plan.actions[0].canonical_parameters,
            {"session_id": 42, "serial_number": 9, "instance_id": 1},
        )

    def test_user_provided_turn_facts_never_authorize_action(self) -> None:
        plan = asyncio.run(
            ChatActionPlanHandler(
                registry=self.registry,
                execution_enabled=True,
            ).execute(self._context(trust_level="USER_PROVIDED"))
        )
        self.assertEqual(plan.decision, "NO_ACTION")
        self.assertIn(
            "VERIFIED_ACTION_PARAMETERS_UNAVAILABLE",
            plan.decision_reasons,
        )

    def test_readonly_agent_never_creates_approval_action(self) -> None:
        plan = asyncio.run(
            ChatActionPlanHandler(
                registry=self.registry,
                execution_enabled=True,
            ).execute(self._context(allow_execution=False))
        )
        self.assertEqual(plan.decision, "NO_ACTION")
        self.assertIn("AGENT_EXECUTION_NOT_ALLOWED", plan.decision_reasons)

    def test_conflicting_actions_for_same_object_fail_closed(self) -> None:
        base = self._context()
        columns = (
            {"name": "owner"},
            {"name": "job_name"},
            {"name": "enabled"},
            {"name": "state"},
            {"name": "run_count"},
            {"name": "failure_count"},
        )
        facts = (
            TurnEvidenceFact(
                evidence_ref="artifact:enable#row-0",
                artifact_id="enable",
                source_id="oracle.scheduler",
                step_id="enable",
                tool_id="db.scheduler.job.enable_candidate",
                trust_level="SOURCE_VERIFIED",
                measurement_semantics=MeasurementSemantics.CURRENT_ACTIVITY,
                presentation_kind="TABLE",
                captured_at="2026-09-02T00:00:00+00:00",
                columns=columns,
                rows=(("APP", "JOB_A", "FALSE", "DISABLED", 0, 0),),
                row_count=1,
            ),
            TurnEvidenceFact(
                evidence_ref="artifact:disable#row-0",
                artifact_id="disable",
                source_id="oracle.scheduler",
                step_id="disable",
                tool_id="db.scheduler.job.disable_candidate",
                trust_level="SOURCE_VERIFIED",
                measurement_semantics=MeasurementSemantics.CURRENT_ACTIVITY,
                presentation_kind="TABLE",
                captured_at="2026-09-02T00:00:00+00:00",
                columns=columns,
                rows=(("APP", "JOB_A", "TRUE", "SCHEDULED", 0, 0),),
                row_count=1,
            ),
        )
        assessment = DbaSufficiencyAssessment(
            status=SufficiencyStatus.ANSWERABLE,
            evidence=facts,
        )
        plan_snapshot = {
            **base.plan_snapshot,
            "capability_snapshot": {
                "target_capabilities": ["dba_catalog_views"]
            },
            "change_context": {
                **base.plan_snapshot["change_context"],
                "controlled_action_execution": {
                    "enabled": True,
                    "allowed_action_ids": [
                        "db.scheduler.job.enable",
                        "db.scheduler.job.disable",
                    ],
                    "object_scopes": {
                        "schemas": ["APP"],
                        "exclude_system_objects": True,
                    },
                },
            },
        }
        context = base.__class__(
            **{
                **base.__dict__,
                "plan_snapshot": plan_snapshot,
                "input_artifacts": (
                    {
                        "schema_version": "DBA_SUFFICIENCY.v1",
                        "payload": assessment.model_dump(mode="json"),
                    },
                ),
            }
        )

        plan = asyncio.run(
            ChatActionPlanHandler(
                registry=self.registry,
                execution_enabled=True,
            ).execute(context)
        )

        self.assertEqual("NO_ACTION", plan.decision)
        self.assertIn("AMBIGUOUS_ACTION_INTENT", plan.decision_reasons)

    def test_next_action_is_released_only_after_verified_predecessor(self) -> None:
        first_plan = asyncio.run(
            ChatActionPlanHandler(
                registry=self.registry,
                execution_enabled=True,
            ).execute(self._context())
        )
        second = first_plan.actions[0].model_copy(update={"ordinal": 2})
        plan = first_plan.model_copy(
            update={"actions": (first_plan.actions[0], second)}
        )
        run_id = uuid7()
        target_id = uuid7()
        proposal_task_id = uuid7()
        plan_task = SimpleNamespace(
            task_key="change:action-plan",
            output_artifact_id=uuid7(),
            ops_task_id=uuid7(),
        )
        proposal_task = SimpleNamespace(
            task_key="change:proposal",
            output_artifact_id=uuid7(),
            ops_task_id=proposal_task_id,
        )
        added = []

        async def add_artifact(entity):
            entity.artifact_id = uuid7()
            added.append(entity)
            return entity

        uow = SimpleNamespace(
            runs=SimpleNamespace(
                list_tasks=AsyncMock(return_value=[plan_task, proposal_task]),
                get_artifact=AsyncMock(
                    return_value=SimpleNamespace(
                        schema_version="ACTION_PLAN.v1",
                        payload_json=plan.model_dump(mode="json"),
                    )
                ),
                add_artifact=AsyncMock(side_effect=add_artifact),
            ),
            changes=SimpleNamespace(
                get_proposal_by_ordinal=AsyncMock(return_value=None)
            ),
        )
        runtime = SimpleNamespace(
            _materialize_advisory_proposal=AsyncMock(),
            _append_sequenced_proposal_block=AsyncMock(),
        )
        verification = ActionVerification(
            proposal_id=str(uuid7()),
            source_run_id=str(run_id),
            result_artifact_id=str(uuid7()),
            status="VERIFIED",
            summary="第一条动作已验证",
        )
        asyncio.run(
            AIOpsRuntimeService._release_next_action_proposal(
                runtime,
                uow=uow,
                source_run=SimpleNamespace(
                    ops_run_id=run_id,
                    target_id=target_id,
                    plan_snapshot_json={
                        "target": {"row_version": 3, "security_level": 2}
                    },
                ),
                source_proposal=SimpleNamespace(
                    proposal_id=uuid7(),
                    command_ordinal=1,
                    ops_task_id=proposal_task_id,
                ),
                verification=verification,
                now=datetime(2026, 9, 2, tzinfo=UTC),
                trace_id="trace-sequence",
            )
        )

        self.assertEqual(1, len(added))
        released = added[0].payload_json["proposal"]
        self.assertEqual(2, released["command_ordinal"])
        runtime._materialize_advisory_proposal.assert_awaited_once()
        runtime._append_sequenced_proposal_block.assert_awaited_once()

    def _context(
        self,
        *,
        trust_level: str = "SOURCE_VERIFIED",
        allow_execution: bool = True,
    ) -> TaskExecutionContext:
        facts = (
            TurnEvidenceFact(
                evidence_ref="artifact:blocking#blocking",
                artifact_id="blocking",
                source_id="oracle.session.blocking_chain",
                step_id="blocking",
                tool_id="db.session.blocking_chain",
                trust_level=trust_level,
                measurement_semantics=MeasurementSemantics.CURRENT_ACTIVITY,
                presentation_kind="TABLE",
                captured_at="2026-08-28T00:00:00+00:00",
                columns=(
                    {"name": "blocking_instance_id"},
                    {"name": "blocking_session_id"},
                ),
                rows=((1, 42),),
                row_count=1,
            ),
            TurnEvidenceFact(
                evidence_ref="artifact:active#active",
                artifact_id="active",
                source_id="oracle.session.active",
                step_id="active",
                tool_id="db.session.active",
                trust_level=trust_level,
                measurement_semantics=MeasurementSemantics.CURRENT_ACTIVITY,
                presentation_kind="TABLE",
                captured_at="2026-08-28T00:00:00+00:00",
                columns=(
                    {"name": "instance_id"},
                    {"name": "session_id"},
                    {"name": "serial_number"},
                ),
                rows=((1, 42, 9),),
                row_count=1,
            ),
        )
        assessment = DbaSufficiencyAssessment(
            status=SufficiencyStatus.ANSWERABLE,
            evidence=facts,
        )
        return TaskExecutionContext(
            run_id="run-chat",
            task_id="action-chat",
            task_key="change:action-plan",
            target_id="target-1",
            agent_id="agent-1",
            trigger_type="CHAT",
            trace_id="trace-chat",
            attempt=1,
            deadline_at=None,
            plan_snapshot={
                "capability_snapshot": {
                    "target_capabilities": ["session_management"]
                },
                "answer_context": {
                    "task_frame": {"requires_change": True}
                },
                "change_context": {
                    "target": {
                        "db_type": "ORACLE",
                        "version_code": "19.0.0",
                        "environment": "PROD",
                        "status": "ENABLED",
                        "connectivity_status": "CONNECTED",
                        "execution_secret_configured": True,
                        "capabilities": {"session_management": True},
                    },
                    "policy": {
                        "rules": {}
                    },
                    "controlled_action_execution": {
                        "enabled": allow_execution,
                        "allowed_action_ids": (
                            ["db.session.terminate"]
                            if allow_execution
                            else []
                        ),
                        "object_scopes": {
                            "schemas": [],
                            "exclude_system_objects": True,
                        },
                    },
                },
            },
            policy_snapshot={},
            input_artifacts=(
                {
                    "schema_version": "DBA_SUFFICIENCY.v1",
                    "payload": assessment.model_dump(mode="json"),
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
