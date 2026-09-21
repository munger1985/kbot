"""Target 运维事实的配置写入、聊天确认和 Schema 契约测试。"""

from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.application.configuration.target_service import (
    TargetConfigurationMixin,
)
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.targets.facts import normalize_fact_value
from aiops_agent.application.turns import ConversationTurnService
from platform_core.contracts.aiops import (
    TargetFactConfirmCommand,
    TargetFactCreate,
)
from platform_core.identity import uuid7


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = ROOT / "database" / "oracle" / "aiops_agent"
REBUILD_SCRIPT = (
    ROOT
    / "database"
    / "oracle"
    / "generated"
    / "aiops_agent"
    / "rebuild_aiops_schema.sql"
)
MANIFEST = SCHEMA_DIR / "schema_manifest.json"


def _scope() -> ConfigurationScope:
    return ConfigurationScope(
        domain_id=100,
        principal_id="PORTAL:aiops",
        actor_id="operator-1",
        request_id="request-1",
        trace_id="trace-1",
    )


def _integrity_error() -> IntegrityError:
    return IntegrityError(
        "INSERT INTO KBOT_OPS_TARGET_FACT",
        {},
        Exception("unique constraint"),
    )


class _FactStore:
    def __init__(self, *, target_id: UUID, domain_id: int) -> None:
        self.target_id = target_id
        self.domain_id = domain_id
        self.rows = []
        self.raise_integrity = False

    async def get_scoped(self, *, target_id, domain_id, lock=False):
        del lock
        if target_id == self.target_id and domain_id == self.domain_id:
            return SimpleNamespace(target_id=target_id, domain_id=domain_id)
        return None

    async def list_target_facts(
        self, *, target_id, domain_id, active_only=True
    ):
        rows = [
            row
            for row in self.rows
            if row.target_id == target_id and row.domain_id == domain_id
        ]
        if active_only:
            rows = [row for row in rows if row.status == "ACTIVE"]
        return rows

    async def get_active_target_fact(
        self,
        *,
        target_id,
        domain_id,
        fact_type,
        fact_key,
        lock=False,
    ):
        del lock
        return next(
            (
                row
                for row in self.rows
                if row.target_id == target_id
                and row.domain_id == domain_id
                and row.fact_type == fact_type
                and row.fact_key == fact_key
                and row.status == "ACTIVE"
            ),
            None,
        )

    async def get_target_fact_scoped(
        self,
        *,
        fact_id,
        target_id,
        domain_id,
        lock=False,
    ):
        del lock
        return next(
            (
                row
                for row in self.rows
                if row.target_fact_id == fact_id
                and row.target_id == target_id
                and row.domain_id == domain_id
            ),
            None,
        )

    async def add_target_fact(self, entity):
        if self.raise_integrity:
            raise _integrity_error()
        self.rows.append(entity)
        return entity


class _Outbox:
    def __init__(self) -> None:
        self.rows = []

    async def add(self, row):
        self.rows.append(row)
        return row


class _Uow:
    def __init__(self, store: _FactStore) -> None:
        self.targets = store
        self.outbox = _Outbox()
        self.commit_count = 0
        self.conversation = None
        self.turn = None
        self.conversations = SimpleNamespace(
            get_conversation=self._get_conversation
        )
        self.turns = SimpleNamespace(get_turn=self._get_turn)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        return False

    async def commit(self):
        self.commit_count += 1

    async def _get_conversation(self, *, domain_id, conversation_id):
        row = self.conversation
        if (
            row is not None
            and int(row.domain_id) == domain_id
            and row.conversation_id == conversation_id
        ):
            return row
        return None

    async def _get_turn(self, *, domain_id, turn_id):
        row = self.turn
        if (
            row is not None
            and int(row.domain_id) == domain_id
            and row.turn_id == turn_id
        ):
            return row
        return None


def _mixin(uow: _Uow) -> TargetConfigurationMixin:
    service = object.__new__(TargetConfigurationMixin)

    async def execute_idempotently(**kwargs):
        return await kwargs["handler"](uow, datetime.now(UTC))

    service._idempotent = execute_idempotently

    class UowContext:
        async def __aenter__(inner_self):
            return uow

        async def __aexit__(inner_self, exc_type, exc, traceback):
            del exc_type, exc, traceback
            return False

    service._uow_factory = UowContext
    return service


class TargetFactNormalizeTest(unittest.TestCase):
    def test_key_must_match_whitelist_field_and_extra_values_are_dropped(
        self,
    ) -> None:
        key, value = normalize_fact_value(
            fact_type="ASM_DISKGROUP",
            fact_key="DATA",
            fact_value={
                "diskgroup_name": "DATA",
                "sql_text": "ALTER DATABASE DATAFILE '+DATA/orcl/datafile/users.dbf' RESIZE 10G",
                "directory": "/u01/oradata",
            },
        )
        self.assertEqual("DATA", key)
        self.assertEqual({"diskgroup_name": "DATA"}, value)

    def test_rejects_mismatched_key(self) -> None:
        with self.assertRaises(AIOpsApplicationError) as caught:
            normalize_fact_value(
                fact_type="DATAFILE_PATH",
                fact_key="/u01/oradata",
                fact_value={"directory": "/u02/oradata"},
            )
        self.assertEqual("OPS_VALIDATION_FAILED", caught.exception.code)
        self.assertEqual(
            "事实键必须与事实值中的关键字段一致",
            caught.exception.message,
        )

    def test_rejects_unknown_type(self) -> None:
        with self.assertRaises(AIOpsApplicationError) as caught:
            normalize_fact_value(
                fact_type="AUTOEXTEND",
                fact_key="USERS",
                fact_value={"tablespace_name": "USERS"},
            )
        self.assertEqual("OPS_VALIDATION_FAILED", caught.exception.code)


class TargetFactConfigurationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.target_id = uuid7()
        self.store = _FactStore(target_id=self.target_id, domain_id=100)
        self.uow = _Uow(self.store)
        self.service = _mixin(self.uow)
        self.scope = _scope()

    async def test_manual_create_is_listed_as_active(self) -> None:
        created = await self.service.create_target_fact(
            scope=self.scope,
            target_id=self.target_id,
            request=TargetFactCreate(
                fact_type="ASM_DISKGROUP",
                fact_key="DATA",
                fact_value={
                    "diskgroup_name": "DATA",
                    "sql_text": "ALTER TABLESPACE USERS ADD DATAFILE",
                },
            ),
            idempotency_key="fact-create-1",
        )
        listed = await self.service.list_target_facts(
            scope=self.scope,
            target_id=self.target_id,
        )
        self.assertEqual(1, len(listed.items))
        self.assertEqual(created.fact_id, listed.items[0].fact_id)
        self.assertEqual("ACTIVE", listed.items[0].status)
        self.assertEqual("MANUAL_CONFIRMED", listed.items[0].source)
        self.assertEqual({"diskgroup_name": "DATA"}, listed.items[0].fact_value)
        self.assertEqual(
            ["TARGET_FACT_CREATED"],
            [row.event_type for row in self.uow.outbox.rows],
        )

    async def test_duplicate_active_natural_key_is_rejected(self) -> None:
        request = TargetFactCreate(
            fact_type="DATAFILE_PATH",
            fact_key="/u01/oradata",
            fact_value={"directory": "/u01/oradata"},
        )
        await self.service.create_target_fact(
            scope=self.scope,
            target_id=self.target_id,
            request=request,
            idempotency_key="fact-create-1",
        )
        with self.assertRaises(AIOpsApplicationError) as caught:
            await self.service.create_target_fact(
                scope=self.scope,
                target_id=self.target_id,
                request=request,
                idempotency_key="fact-create-2",
            )
        self.assertEqual("OPS_STATE_CONFLICT", caught.exception.code)
        self.assertEqual(
            "配置自然键已存在或并发创建冲突",
            caught.exception.message,
        )
        self.assertEqual(1, len(self.store.rows))

    async def test_retire_hides_fact_from_default_list(self) -> None:
        created = await self.service.create_target_fact(
            scope=self.scope,
            target_id=self.target_id,
            request=TargetFactCreate(
                fact_type="TABLESPACE_PLACEMENT",
                fact_key="USERS",
                fact_value={"tablespace_name": "USERS"},
            ),
            idempotency_key="fact-create-1",
        )
        retired = await self.service.retire_target_fact(
            scope=self.scope,
            target_id=self.target_id,
            fact_id=created.fact_id,
            expected_version=created.row_version,
            idempotency_key="fact-retire-1",
        )
        listed = await self.service.list_target_facts(
            scope=self.scope,
            target_id=self.target_id,
        )
        self.assertEqual("RETIRED", retired.status)
        self.assertEqual("RETIRED", self.store.rows[0].status)
        self.assertEqual((), listed.items)
        self.assertEqual(
            ["TARGET_FACT_CREATED", "TARGET_FACT_RETIRED"],
            [row.event_type for row in self.uow.outbox.rows],
        )


class TargetFactConversationConfirmTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.target_id = uuid7()
        self.conversation_id = uuid7()
        self.turn_id = uuid7()
        self.store = _FactStore(target_id=self.target_id, domain_id=100)
        self.uow = _Uow(self.store)
        self.uow.conversation = SimpleNamespace(
            conversation_id=self.conversation_id,
            domain_id=100,
            target_id=self.target_id,
            created_by="dba@example.com",
        )
        self.uow.turn = SimpleNamespace(
            turn_id=self.turn_id,
            domain_id=100,
            conversation_id=self.conversation_id,
        )
        self.service = ConversationTurnService(uow_factory=lambda: self.uow)

    async def test_chat_confirm_persists_manual_confirmed_fact(self) -> None:
        self.assertEqual([], self.store.rows)
        created = await self.service.confirm_target_fact(
            domain_id=100,
            conversation_id=self.conversation_id,
            turn_id=self.turn_id,
            actor_id="dba@example.com",
            trace_id="trace-confirm",
            command=TargetFactConfirmCommand(
                target_id=self.target_id,
                fact_type="ASM_DISKGROUP",
                fact_key="DATA",
                fact_value={
                    "diskgroup_name": "DATA",
                    "command_preview": "ALTER DATABASE DATAFILE",
                },
                note="容量扩容需要 ASM 组",
            ),
        )
        self.assertEqual(1, self.uow.commit_count)
        self.assertEqual(1, len(self.store.rows))
        self.assertEqual("MANUAL_CONFIRMED", created.source)
        self.assertEqual("ACTIVE", created.status)
        self.assertEqual({"diskgroup_name": "DATA"}, created.fact_value)
        self.assertNotIn("command_preview", created.fact_value)
        self.assertEqual("dba@example.com", created.confirmed_by)
        self.assertEqual(
            ["TARGET_FACT_CREATED"],
            [row.event_type for row in self.uow.outbox.rows],
        )
        self.assertEqual(
            str(self.conversation_id),
            self.uow.outbox.rows[0].payload_json["details"]["conversation_id"],
        )

    async def test_unconfirmed_chat_does_not_persist(self) -> None:
        listed = await _mixin(self.uow).list_target_facts(
            scope=_scope(),
            target_id=self.target_id,
        )
        self.assertEqual((), listed.items)
        self.assertEqual([], self.store.rows)
        self.assertEqual(0, self.uow.commit_count)

    async def test_integrity_error_maps_to_state_conflict(self) -> None:
        self.store.raise_integrity = True
        with self.assertRaises(AIOpsApplicationError) as caught:
            await self.service.confirm_target_fact(
                domain_id=100,
                conversation_id=self.conversation_id,
                turn_id=self.turn_id,
                actor_id="dba@example.com",
                trace_id="trace-conflict",
                command=TargetFactConfirmCommand(
                    target_id=self.target_id,
                    fact_type="ASM_DISKGROUP",
                    fact_key="DATA",
                    fact_value={"diskgroup_name": "DATA"},
                ),
            )
        self.assertEqual("OPS_STATE_CONFLICT", caught.exception.code)
        self.assertEqual(
            "配置自然键已存在或并发创建冲突",
            caught.exception.message,
        )
        self.assertEqual(0, self.uow.commit_count)
        self.assertEqual([], self.store.rows)

    async def test_confirm_rejects_mismatched_target(self) -> None:
        with self.assertRaises(AIOpsApplicationError) as caught:
            await self.service.confirm_target_fact(
                domain_id=100,
                conversation_id=self.conversation_id,
                turn_id=self.turn_id,
                actor_id="dba@example.com",
                trace_id="trace-mismatch",
                command=TargetFactConfirmCommand(
                    target_id=uuid7(),
                    fact_type="DATAFILE_PATH",
                    fact_key="/u01/oradata",
                    fact_value={"directory": "/u01/oradata"},
                ),
            )
        self.assertEqual("OPS_VALIDATION_FAILED", caught.exception.code)
        self.assertEqual(
            "确认的 Target 必须与当前对话一致",
            caught.exception.message,
        )
        self.assertEqual([], self.store.rows)
        self.assertEqual(0, self.uow.commit_count)


class TargetFactSchemaContractTest(unittest.TestCase):
    def test_rebuild_schema_and_manifest_include_target_fact(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        rebuild = REBUILD_SCRIPT.read_text(encoding="utf-8")
        roots = (SCHEMA_DIR / "001_ops_roots.sql").read_text(encoding="utf-8")
        self.assertEqual(26, manifest["schema_version"])
        self.assertEqual("aiops-oracle-v16", manifest["contract_version"])
        self.assertIn("KBOT_OPS_TARGET_FACT", manifest["tables"])
        self.assertIn("UX_OPS_TARGET_FACT_ACTIVE", manifest["function_unique_indexes"])
        self.assertIn("CREATE TABLE KBOT_OPS_TARGET_FACT", rebuild)
        self.assertIn("CREATE TABLE KBOT_OPS_TARGET_FACT", roots)
        self.assertIn("不保存可执行 SQL", roots)
        self.assertIn("不保存可执行 SQL", rebuild)
        self.assertNotIn("command_preview", roots)


if __name__ == "__main__":
    unittest.main()
