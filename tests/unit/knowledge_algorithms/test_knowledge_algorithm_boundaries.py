"""算法包必须能在仅有标准库的隔离解释器中导入，测试目录不遮蔽发行包。"""

import ast
from dataclasses import fields
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "packages/knowledge_algorithms/src/knowledge_algorithms"
sys.path.insert(0, str(PACKAGE.parent))

from knowledge_algorithms.contracts.query import QueryInput
from knowledge_algorithms.parsing.events import ParseEvent


class BoundaryTests(unittest.TestCase):
    def test_repository_checker_rejects_algorithm_business_dependencies(self):
        candidate = ROOT / "tests/acceptance/check_4_0_boundaries.py"
        if not candidate.exists():
            candidate = ROOT / "tests/acceptance/check_architecture_boundaries.py"
        spec = importlib.util.spec_from_file_location("algorithm_boundary_checker", candidate)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        self.assertIn(PACKAGE, checker.ACTIVE_ROOTS)
        for source in ("import sqlalchemy", "import platform_core", "import knowledge_core", "import docling"):
            with self.subTest(source=source), patch.object(Path, "read_text", return_value=source):
                self.assertTrue(checker.check_file(PACKAGE / "probe.py"))
        with patch.object(Path, "read_text", return_value="from .contracts import identity\nimport math"):
            self.assertEqual(checker.check_file(PACKAGE / "probe.py"), [])

    def test_only_neutral_standard_library_imports(self):
        allowed = {"dataclasses", "enum", "hashlib", "json", "math", "re", "typing", "types",
                   "asyncio", "time", "base64", "hmac", "codecs", "collections", "unicodedata", "urllib"}
        for path in PACKAGE.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [(node.module or "").split(".")[0]]
                else:
                    continue
                with self.subTest(path=path.name, names=names):
                    self.assertTrue(set(names) <= allowed)

    def test_no_dynamic_code_or_file_access(self):
        forbidden = {"open", "eval", "exec", "__import__", "compile"}
        for path in PACKAGE.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, forbidden, str(path))

    def test_import_without_site_packages_models_or_ripgrep(self):
        program = (
            "import sys, pkgutil, importlib; "
            f"sys.path.insert(0, {str(PACKAGE.parent)!r}); "
            "import knowledge_algorithms as p; "
            "modules = [m.name for m in pkgutil.walk_packages(p.__path__, p.__name__ + '.')]; "
            "[importlib.import_module(name) for name in modules]; "
            f"assert p.__file__.startswith({str(PACKAGE)!r}); "
            "assert not any(k.split('.')[0] in {'docling','torch','sqlalchemy','fastapi','platform_core'} "
            "for k in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-I", "-S", "-c", program], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_no_app_router_or_reranker_fields(self):
        forbidden = {"app_ids", "target_app", "table_name", "session", "reranker", "do_rerank"}
        for contract in (QueryInput, ParseEvent):
            self.assertFalse({f.name for f in fields(contract)} & forbidden)
        self.assertFalse(list(PACKAGE.rglob("*rerank*")))

    def test_workspace_and_release_registration(self):
        for relative in ("pyproject.toml", "scripts/deployment/install_workspace.sh",
                         "scripts/deployment/workspace_fingerprint.py", "scripts/release/verify_release.py"):
            with self.subTest(path=relative):
                self.assertIn("packages/knowledge_algorithms", (ROOT / relative).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
