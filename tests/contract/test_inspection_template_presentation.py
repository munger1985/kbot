"""巡检卡片共享网格和项目风格模拟报告抽屉的两端静态回归。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IS_KBOT = ROOT.name == "kbot4"
STYLES = (ROOT / ("ui/aiops/css/ammolite-shared.css" if IS_KBOT else "apps/portal-web/src/shared/ui/styles.css")).read_text(encoding="utf-8")
SOURCE = (ROOT / ("ui/aiops/js/aiops-report-templates.js" if IS_KBOT else "apps/portal-web/src/features/aiops/management/AIOpsReportTemplatesView.vue")).read_text(encoding="utf-8")


def test_catalog_facts_have_complete_grid_lines_and_full_width_odd_tail():
    assert ".management-catalog-card dl {" in STYLES
    rules = STYLES.split(".management-catalog-card dl {", 1)[1].split("}", 1)[0]
    assert "border: 1px solid var(--line)" in rules
    assert "dl > div:nth-child(even) { border-left: 1px solid var(--line)" in STYLES
    assert "dl > div:nth-child(n + 3) { border-top: 1px solid var(--line)" in STYLES
    assert "dl > div:last-child:nth-child(odd) { grid-column: 1 / -1" in STYLES
    assert ".management-catalog-card dl > div + div" not in STYLES


def test_preview_uses_project_drawer_pinned_version_and_no_new_browser_tab():
    assert "window.open(" not in SOURCE and "templatePreview" not in SOURCE
    assert "重试预览" in SOURCE
    assert "template-readonly-preview" not in SOURCE and "inspection-readonly-preview" not in SOURCE
    assert "previewVersion" in SOURCE and "previewSequence" in SOURCE
    assert "seq===previewSequence" in SOURCE.replace(" ", "")
    assert "document.body.style.overflow" in SOURCE
    if IS_KBOT:
        markup = (ROOT / "ui/aiops/report-templates.html").read_text(encoding="utf-8")
        assert "previewVersion=version" in SOURCE
        assert "previewDrawer.showModal()" in SOURCE
        assert 'addEventListener("cancel"' in SOURCE
        assert 'addEventListener("close"' in SOURCE
        assert "closeInspectionPreview(false)" in SOURCE
        assert 'anchor.getAttribute("href").slice(1)' in SOURCE
    else:
        markup = SOURCE
        assert "previewVersion.value = selectedVersion.value" in SOURCE
        assert 'Teleport to="body"' in SOURCE
        assert ":inert=" in SOURCE and "previewKeydown" in SOURCE
        assert "onBeforeUnmount" in SOURCE and "previewTrigger" in SOURCE
        assert "event.preventDefault()" in SOURCE and "scrollIntoView" in SOURCE
    for value in ["admin-drawer-wide", "admin-drawer-sectioned", "inspection-template-preview-drawer",
                  "admin-drawer-header", "admin-drawer-body", "关闭模拟报告预览", "不代表实际巡检结果"]:
        assert value in markup
