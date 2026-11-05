"""发布与上下线是业务系统唯一的用户可见运行状态入口。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_business_lifecycle_does_not_add_independent_target_toggles():
    if (ROOT / "ui/aiops").exists():
        directory = (ROOT / "ui/aiops/js/aiops-business-systems.js").read_text()
        workspace = (ROOT / "ui/aiops/js/aiops-business-system-workspace.js").read_text()
        agent = (ROOT / "ui/aiops/js/aiops-agents.js").read_text()
    else:
        base = ROOT / "apps/portal-web/src/features/aiops/management"
        directory = (base / "AIOpsBusinessSystemsView.vue").read_text()
        workspace = (base / "BusinessSystemWorkspace.vue").read_text()
        agent = (base / "AIOpsAgentsView.vue").read_text()
    assert "发布资源" in workspace and "已发布并上线" in workspace
    assert "下线业务系统" in directory and "重新上线" in directory
    for view in (directory, workspace):
        for control in ("data-enable-target", "data-disable-target", "启用 Target", "停用 Target"):
            assert control not in view
    assert "所属业务系统的上线或重新上线统一管理" in agent
    assert "Target 未启用" not in agent
