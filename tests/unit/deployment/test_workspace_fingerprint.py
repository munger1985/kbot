"""启动包内容指纹测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.deployment import workspace_fingerprint


class WorkspaceFingerprintTest(unittest.TestCase):
    def test_equal_installed_tree_has_no_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            package = workspace_fingerprint.WorkspacePackage(
                "packages/example", "kbot-example", "example"
            )
            source = root / package.member / "src" / package.module
            installed = root / "site-packages" / package.module
            source.mkdir(parents=True)
            installed.mkdir(parents=True)
            (source / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
            (installed / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
            with (
                patch.object(workspace_fingerprint, "ROOT", root),
                patch.object(
                    workspace_fingerprint,
                    "WORKSPACE_PACKAGES",
                    (package,),
                ),
                patch.object(
                    workspace_fingerprint,
                    "installed_module_root",
                    return_value=installed,
                ),
            ):
                self.assertEqual(
                    [], workspace_fingerprint.compare_workspace_packages()
                )

    def test_same_version_with_old_content_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            package = workspace_fingerprint.WorkspacePackage(
                "services/example", "kbot-example", "example"
            )
            source = root / package.member / "src" / package.module
            installed = root / "site-packages" / package.module
            source.mkdir(parents=True)
            installed.mkdir(parents=True)
            (source / "service.py").write_text("VALUE = 2\n", encoding="utf-8")
            (installed / "service.py").write_text("VALUE = 1\n", encoding="utf-8")
            with (
                patch.object(workspace_fingerprint, "ROOT", root),
                patch.object(
                    workspace_fingerprint,
                    "WORKSPACE_PACKAGES",
                    (package,),
                ),
                patch.object(
                    workspace_fingerprint,
                    "installed_module_root",
                    return_value=installed,
                ),
            ):
                self.assertEqual(
                    ["kbot-example:CONTENT_MISMATCH"],
                    workspace_fingerprint.compare_workspace_packages(),
                )

    def test_non_packaged_readme_does_not_trigger_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            package = workspace_fingerprint.WorkspacePackage(
                "services/example", "kbot-example", "example"
            )
            source = root / package.member / "src" / package.module
            installed = root / "site-packages" / package.module
            source.mkdir(parents=True)
            installed.mkdir(parents=True)
            (source / "__init__.py").write_text("", encoding="utf-8")
            (installed / "__init__.py").write_text("", encoding="utf-8")
            (source / "README.md").write_text("说明", encoding="utf-8")
            with (
                patch.object(workspace_fingerprint, "ROOT", root),
                patch.object(
                    workspace_fingerprint,
                    "WORKSPACE_PACKAGES",
                    (package,),
                ),
                patch.object(
                    workspace_fingerprint,
                    "installed_module_root",
                    return_value=installed,
                ),
            ):
                self.assertEqual(
                    [], workspace_fingerprint.compare_workspace_packages()
                )


if __name__ == "__main__":
    unittest.main()
