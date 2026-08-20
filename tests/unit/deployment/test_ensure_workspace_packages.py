"""启动前内部包自动更新测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.deployment import ensure_workspace_packages


class EnsureWorkspacePackagesTest(unittest.TestCase):
    def test_mismatch_installs_once_and_rechecks(self):
        with TemporaryDirectory() as directory:
            with (
                patch.object(
                    ensure_workspace_packages,
                    "ROOT",
                    Path(directory),
                ),
                patch.object(
                    ensure_workspace_packages,
                    "compare_workspace_packages",
                    return_value=["kbot-example:CONTENT_MISMATCH"],
                ),
                patch.object(
                    ensure_workspace_packages, "_install"
                ) as install,
                patch.object(
                    ensure_workspace_packages,
                    "_verify_in_fresh_interpreter",
                    return_value=True,
                ),
                patch(
                    "sys.argv",
                    [
                        "ensure_workspace_packages.py",
                        "--mode",
                        "production",
                    ],
                ),
                patch.dict(
                    "os.environ", {"KBOT_AUTO_INSTALL_PACKAGES": "true"}
                ),
            ):
                self.assertEqual(0, ensure_workspace_packages.main())
                install.assert_called_once_with("production")

    def test_disabled_auto_install_fails_closed(self):
        with TemporaryDirectory() as directory:
            with (
                patch.object(
                    ensure_workspace_packages,
                    "ROOT",
                    Path(directory),
                ),
                patch.object(
                    ensure_workspace_packages,
                    "compare_workspace_packages",
                    return_value=["kbot-example:CONTENT_MISMATCH"],
                ),
                patch.object(
                    ensure_workspace_packages, "_install"
                ) as install,
                patch(
                    "sys.argv",
                    [
                        "ensure_workspace_packages.py",
                        "--mode",
                        "production",
                    ],
                ),
                patch.dict(
                    "os.environ", {"KBOT_AUTO_INSTALL_PACKAGES": "false"}
                ),
            ):
                self.assertEqual(1, ensure_workspace_packages.main())
                install.assert_not_called()


if __name__ == "__main__":
    unittest.main()
