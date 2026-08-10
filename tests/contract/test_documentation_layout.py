"""KBot 4.0 当前文档结构与本地链接契约。"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ROOT / "docs"
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


class DocumentationLayoutTest(unittest.TestCase):
    def test_only_navigation_remains_at_docs_root(self):
        self.assertEqual(
            {"README.md"},
            {
                path.name
                for path in DOCS_ROOT.glob("*.md")
            },
        )

    def test_current_document_groups_are_explicit(self):
        self.assertEqual(
            {
                "agent-runtime.md",
                "aiops-agent.md",
                "knowledge-core.md",
                "model-serving.md",
                "overview.md",
                "repository-layout.md",
                "security-and-api.md",
            },
            {
                path.name
                for path in (DOCS_ROOT / "architecture").glob("*.md")
            },
        )
        self.assertEqual(
            {
                "deployment.md",
                "oracle-linux-nginx-http-apex-ords.md",
                "oracle-linux-nginx-https-gateway.md",
            },
            {
                path.name
                for path in (DOCS_ROOT / "operations").glob("*.md")
            },
        )
        self.assertEqual(
            {
                "agent-chat.md",
                "aiops-agent.md",
                "knowledge-lifecycle.md",
                "slack-integration.md",
            },
            {
                path.name
                for path in (DOCS_ROOT / "product").glob("*.md")
            },
        )
        self.assertFalse(any((DOCS_ROOT / "migrations").glob("*.md")))

    def test_obsolete_design_directories_do_not_return(self):
        self.assertFalse((DOCS_ROOT / "kbot_4.0_design").exists())
        self.assertFalse((DOCS_ROOT / "kbot_4.0_showcase").exists())
        self.assertFalse((DOCS_ROOT / "install").exists())

    def test_all_local_markdown_links_resolve(self):
        missing: list[str] = []
        paths = [ROOT / "README.md", *DOCS_ROOT.rglob("*.md")]
        for path in paths:
            source = path.read_text(encoding="utf-8")
            for raw_target in _MARKDOWN_LINK.findall(source):
                target = raw_target.split("#", 1)[0].strip()
                if (
                    not target
                    or "://" in target
                    or target.startswith("mailto:")
                ):
                    continue
                if not (path.parent / target).resolve().exists():
                    missing.append(f"{path.relative_to(ROOT)} -> {target}")
        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
