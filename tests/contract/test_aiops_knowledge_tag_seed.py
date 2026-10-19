"""Oracle AIOps 知识标签初始词表离线契约测试。"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "database/oracle/bootstrap/aiops/initial_knowledge_tags.sql"
EXPECTED_COUNTS = {
    "RESOURCE_CATEGORY": 25,
    "TECHNOLOGY_STACK": 64,
    "VENDOR": 44,
}


class AIOpsKnowledgeTagSeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sql = SCRIPT.read_text(encoding="utf-8")
        blocks = re.findall(r"q'~(\[.*?\])~'", cls.sql, re.DOTALL)
        cls.block_sizes = [len(block.encode("utf-8")) for block in blocks]
        cls.tags = [item for block in blocks for item in json.loads(block)]

    def test_seed_has_complete_unique_catalog(self) -> None:
        counts = {
            category: sum(
                item["category"] == category for item in self.tags
            )
            for category in EXPECTED_COUNTS
        }
        normalized = {
            (
                item["category"],
                " ".join(item["display_name"].strip().upper().split()),
            )
            for item in self.tags
        }

        self.assertEqual(EXPECTED_COUNTS, counts)
        self.assertEqual(sum(EXPECTED_COUNTS.values()), len(self.tags))
        self.assertEqual(len(self.tags), len(normalized))

    def test_seed_covers_representative_operations_technologies(self) -> None:
        names = {item["display_name"] for item in self.tags}
        self.assertTrue({
            "关系型数据库",
            "Kubernetes 集群",
            "监控与可观测平台",
            "Oracle Database",
            "PostgreSQL",
            "MySQL",
            "Prometheus",
            "Ansible",
            "Oracle",
            "Huawei",
            "Veeam",
        }.issubset(names))

    def test_seed_is_insert_only_and_oracle_literals_stay_bounded(self) -> None:
        self.assertIn("WHEN NOT MATCHED THEN INSERT", self.sql)
        self.assertNotIn("WHEN MATCHED THEN UPDATE", self.sql)
        self.assertTrue(self.block_sizes)
        self.assertTrue(all(size < 4000 for size in self.block_sizes))


if __name__ == "__main__":
    unittest.main()
