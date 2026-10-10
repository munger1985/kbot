"""DBA 工作项合同、SLA 与指纹规则测试。"""

from datetime import UTC, datetime
from pathlib import Path
import unittest

from pydantic import ValidationError

from aiops_agent.application.work_items import _fingerprint, _sla
from platform_core.contracts.aiops import (
    WorkItemPriority,
    WorkItemStatus,
    WorkItemTransition,
)
from platform_core.identity import uuid7


ROOT = Path(__file__).resolve().parents[3]


class WorkItemContractTest(unittest.TestCase):
    def test_fingerprint_is_stable_for_semantically_equal_object_refs(self):
        target_id = uuid7()
        first = _fingerprint(
            domain_id=7, target_id=target_id,
            work_type="RISK_REMEDIATION", finding_type="TABLESPACE",
            object_ref={"object_kind": "TABLESPACE", "object_name": "USERS"},
            condition_code="usage_percent:>=",
        )
        second = _fingerprint(
            domain_id=7, target_id=target_id,
            work_type="RISK_REMEDIATION", finding_type="TABLESPACE",
            object_ref={"object_name": "USERS", "object_kind": "TABLESPACE"},
            condition_code="usage_percent:>=",
        )
        self.assertEqual(first, second)
        self.assertEqual(64, len(first))

    def test_priority_sla_has_distinct_acknowledgement_and_resolution(self):
        observed_at = datetime(2026, 10, 10, 0, 0, tzinfo=UTC)
        acknowledgement, resolution, verification = _sla(
            WorkItemPriority.P1, observed_at
        )
        self.assertLess(acknowledgement, resolution)
        self.assertLess(resolution, verification)

    def test_waiting_and_resolution_require_business_details(self):
        with self.assertRaises(ValidationError):
            WorkItemTransition(
                expected_row_version=1,
                status=WorkItemStatus.WAITING,
                phase="DIAGNOSIS",
            )
        with self.assertRaises(ValidationError):
            WorkItemTransition(
                expected_row_version=1,
                status=WorkItemStatus.RESOLVED,
                phase="VERIFICATION",
            )

    def test_canonical_ddl_has_no_business_check_or_unique_constraint(self):
        ddl = (ROOT / "database/oracle/aiops_agent/012_ops_work_items.sql").read_text(
            encoding="utf-8"
        ).upper()
        self.assertNotIn(" CHECK ", ddl)
        self.assertNotIn(" UNIQUE ", ddl)
        self.assertIn("KBOT_OPS_WORK_ITEM_ACTIVITY", ddl)


if __name__ == "__main__":
    unittest.main()
