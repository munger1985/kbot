"""巡检卡片共享网格和独立模拟报告页的两端静态回归。"""

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


def test_preview_opens_separate_page_pins_version_and_has_no_inline_report():
    assert "window.open(" in SOURCE and "noopener,noreferrer" in SOURCE
    assert "inspection-template-preview" in SOURCE and "所选发布版本不存在" in SOURCE
    assert "重试预览" in SOURCE and "返回模板详情" in SOURCE
    assert "不代表实际巡检结果" in SOURCE
    assert "template-readonly-preview" not in SOURCE and "inspection-readonly-preview" not in SOURCE
    if IS_KBOT:
        assert 'url.searchParams.set("version",version)' in SOURCE
        assert 'if(!previewParameters())showInspectionDetail()' in SOURCE
    else:
        assert "templateVersionId:selectedVersion.value.template_version_id" in SOURCE
        assert "window.open(url, '_blank'" in SOURCE
