"""统一资源绑定的定向升级不能丢失身份、归属和业务配置。"""

from copy import deepcopy
import asyncio
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("resource_binding_upgrade", ROOT / "scripts/deployment/upgrade_aiops_resource_bindings.py")
upgrade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upgrade)


def example():
    source_id, group, actor = uuid4(), uuid4(), uuid4()
    data = {key: [] for key in upgrade.SUFFIXES}
    data["source"] = [{"diagnostic_source_id": source_id, "domain_id": 11, "source_type": "PROMETHEUS"}]
    for i in range(3):
        resource_id, target_id, system_id = uuid4(), uuid4(), uuid4()
        data["resource"].append({"resource_id": resource_id, "domain_id": 11, "owner_group_id": group,
                                 "resource_type_key": "DATABASE", "created_by": actor, "updated_by": actor})
        data["database"].append({"resource_id": resource_id, "target_id": target_id})
        data["system"].append({"business_system_id": system_id, "domain_id": 11})
        data["link"].append({"resource_id": resource_id, "system_resource_id": uuid4(), "business_system_id": system_id,
                             "responsibility_group_id": group, "display_order": i, "status": "ACTIVE", "created_by": actor, "updated_by": actor})
        data["binding"].append({"target_source_binding_id": uuid4(), "target_id": target_id, "diagnostic_source_id": source_id,
                                "source_locator_key": f"db-{i}", "source_locator_json": {"target_key": f"db-{i}", "host_target_key": "node-88"},
                                "role": "PRIMARY", "priority": 100, "status": "ACTIVE", "query_budget_json": {"max_series": 20},
                                "created_by": actor, "updated_by": actor, "row_version": 7, "health_version": 3})
    return data


class ResourceBindingUpgradeTest(unittest.TestCase):
    def test_database_binding_ids_and_business_values_are_preserved(self):
        data = example()
        before = deepcopy(data)
        plan = upgrade.build_plan(data, {"node-88": "VIRTUAL_MACHINE"})
        old_ids = {row["target_source_binding_id"] for row in data["binding"]}
        databases = [row for row in plan["bindings"] if row["resource_source_binding_id"] in old_ids]
        self.assertEqual(len(databases), 3)
        for row in databases:
            self.assertNotIn("target_id", row)
            self.assertNotIn("host_target_key", row["source_locator_json"])
            self.assertEqual(row["row_version"], 7)
            self.assertEqual(row["health_version"], 3)
            self.assertEqual(row["query_budget_json"], {"max_series": 20})
        self.assertEqual(data, before)

    def test_shared_host_is_created_once_with_three_system_memberships(self):
        plan = upgrade.build_plan(example(), {"node-88": "VIRTUAL_MACHINE"})
        self.assertEqual([len(plan[key]) for key in ("resources", "computes", "links", "relations", "participants", "bindings")], [1, 1, 3, 3, 6, 4])
        self.assertEqual(plan["bindings"][1]["source_locator_json"], {"target_key": "node-88", "resource_scope": "HOST"})
        self.assertTrue(all(row["architecture_layer_key"] == "INFRASTRUCTURE" for row in plan["links"]))
        self.assertTrue(all(row["verification_status"] == "UNKNOWN" for row in plan["relations"]))

    def test_missing_resource_and_duplicate_target_mapping_are_rejected(self):
        for mutate in (lambda d: d["database"].pop(), lambda d: d["database"].append(dict(d["database"][0]))):
            data = example()
            mutate(data)
            with self.assertRaises(ValueError):
                upgrade.build_plan(data, {"node-88": "VIRTUAL_MACHINE"})

    def test_cross_domain_or_tenant_is_rejected(self):
        for field in ("domain_id", "tenant_id"):
            data = example()
            data["source"][0][field] = uuid4()
            with self.assertRaises(ValueError):
                upgrade.build_plan(data, {"node-88": "VIRTUAL_MACHINE"})

    def test_unknown_host_requires_explicit_deferral_and_is_not_fabricated(self):
        data = example()
        with self.assertRaises(ValueError):
            upgrade.build_plan(data, {})
        plan = upgrade.build_plan(data, {}, defer_unknown=True)
        self.assertEqual(len(plan["deferred"]), 3)
        self.assertEqual(len(plan["bindings"]), 3)
        self.assertEqual(len(plan["resources"]), 0)

    def test_non_prometheus_bindings_are_preserved_without_host_guessing(self):
        data = example()
        data["source"][0]["source_type"] = "OEM"
        for row in data["binding"]:
            row["source_locator_json"] = {"target_guid": "guid-value", "target_type": "oracle_database"}
        plan = upgrade.build_plan(data, {})
        self.assertEqual(len(plan["bindings"]), 3)
        self.assertEqual(len(plan["resources"]), 0)
        self.assertEqual(plan["bindings"][0]["source_locator_json"], {"target_guid": "guid-value", "target_type": "oracle_database"})

    def test_database_id_is_not_assumed_equal_to_resource_id(self):
        data = example()
        self.assertNotEqual(data["database"][0]["target_id"], data["database"][0]["resource_id"])
        plan = upgrade.build_plan(data, {"node-88": "VIRTUAL_MACHINE"})
        self.assertEqual({row["resource_id"] for row in plan["bindings"] if row.get("row_version") == 7}, {row["resource_id"] for row in data["database"]})

    def test_table_names_and_canonical_oracle_ddl_are_narrow(self):
        self.assertEqual(upgrade.table_name("agent_system", "postgresql"), "aiops_agent_version_business_system")
        self.assertEqual(upgrade.table_name("agent_system", "oracle"), "KBOT_OPS_AGENT_VERSION_SYSTEM")
        self.assertEqual(upgrade.archive_table_name("postgresql"), "aiops_upgrade_archive.aiops_bak_resource_binding_57")
        if (ROOT / "database/oracle/aiops_agent/001_ops_roots.sql").exists():
            self.assertEqual(len(upgrade.oracle_statements()), 3)
            self.assertEqual(len(upgrade.oracle_views()), 3)
            self.assertNotRegex("\n".join(upgrade.oracle_statements()), r"\b(?:CHECK|UNIQUE)\s*\(")

    def test_backup_round_trip_preserves_identifiers_and_timestamps(self):
        value = {"id": uuid4(), "created_at": datetime.now(UTC), "raw": b"example"}
        self.assertEqual(json.loads(upgrade.serialized(value), object_hook=upgrade.decode), value)

    def test_only_existing_invalid_aiops_views_are_recompiled(self):
        result = Mock()
        result.scalars.return_value = ["KBOT_V_OPS_CHAT_PENDING", "KBOT_V_OPS_PENDING_APPROVAL"]
        connection = Mock()
        connection.execute = AsyncMock(side_effect=[result, None, None])
        asyncio.run(upgrade.compile_invalid_views(connection))
        self.assertEqual([str(call.args[0]) for call in connection.execute.call_args_list[1:]],
                         ["ALTER VIEW KBOT_V_OPS_CHAT_PENDING COMPILE", "ALTER VIEW KBOT_V_OPS_PENDING_APPROVAL COMPILE"])


if __name__ == "__main__":
    unittest.main()
