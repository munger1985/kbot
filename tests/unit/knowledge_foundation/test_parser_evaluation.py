"""评测入口使用纯合成 UTF-8 和隔离子进程，不访问模型、数据库或业务文档。"""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests/unit/knowledge_algorithms")]
from knowledge_algorithm_fixtures import parse_profile
from knowledge_algorithms.contracts.wire import encode

_spec = importlib.util.spec_from_file_location("knowledge_parser_evaluation",
    ROOT / "tests/evaluation/knowledge_algorithms/evaluate_parser.py")
_evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluation)


class ParserEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def evaluate(self, *, expected_content=("合成资料",), cases=True):
        with tempfile.TemporaryDirectory(prefix="knowledge-evaluation-fixture-") as directory:
            root = Path(directory)
            (root / "synthetic.txt").write_text("合成资料。\n\n第二段。", encoding="utf-8")
            manifest = {"owner_app_id": "knowledge_retrieval",
                "profile": encode(parse_profile(timeout_seconds=8.0)),
                "executor": {"memory_bytes": 536870912, "cpu_seconds": 10},
                "cases": [{"case_id": "synthetic", "path": "synthetic.txt",
                    "mime_type": "text/plain", "expected_content": list(expected_content),
                    "expected_kinds": ["PARAGRAPH"]}] if cases else []}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch.dict(os.environ, {"PYTHONPATH": os.pathsep.join(sys.path)}):
                return await _evaluation.evaluate(path)

    async def test_utf8_event_contract_and_content_are_evaluated(self):
        report = await self.evaluate()
        self.assertTrue(report["passed"], report)
        self.assertFalse(report["retrieval_quality_proven"])
        self.assertEqual(1, report["case_count"])
        self.assertGreater(report["results"][0]["node_count"], 0)

    async def test_missing_expected_content_stays_failed(self):
        report = await self.evaluate(expected_content=("不存在的预期内容",))
        self.assertFalse(report["passed"])
        self.assertIn("MISSING_CONTENT:不存在的预期内容", report["results"][0]["failures"])

    async def test_no_cases_is_not_a_successful_evaluation(self):
        report = await self.evaluate(cases=False)
        self.assertFalse(report["passed"])
        self.assertEqual(0, report["case_count"])


if __name__ == "__main__":
    unittest.main()
