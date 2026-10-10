"""AIOps Agent 编辑器静态交互边界测试。"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "ui" / "aiops" / "js" / "aiops-agents.js"
PAGE = ROOT / "ui" / "aiops" / "agents.html"
INTERNAL_API = ROOT / "services" / "aiops_agent" / "src" / "aiops_agent" / "api" / "agents" / "routes.py"
CLIENT = ROOT / "packages" / "platform_clients" / "src" / "platform_clients" / "aiops.py"
MAIN_API = ROOT / "services" / "main_api" / "src" / "main_api" / "api" / "aiops_app.py"


class AIOpsAgentEditorUiTest(unittest.TestCase):
    def test_agent_only_selects_targets_for_resource_scope(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        page = PAGE.read_text(encoding="utf-8")

        self.assertIn('name="target_ids"', script)
        self.assertIn("至少选择一个 Target", script)
        self.assertNotIn("diagnostic_source_ids", script)
        self.assertNotIn('id="agent-sources"', page)

    def test_page_uses_repaired_agent_editor_bundle(self) -> None:
        page = PAGE.read_text(encoding="utf-8")

        self.assertIn("aiops-agents.js?v=20261011-1", page)

    def test_agent_does_not_maintain_monitor_mappings(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertNotIn("assertExistingBindings", script)
        self.assertNotIn("source_locator_key", script)
        self.assertNotIn("source_locator:", script)
        self.assertNotIn("ensureSourceBindings", script)
        self.assertNotIn("/source-bindings", script)

    def test_image_capability_selects_use_catalog_categories(self) -> None:
        page = PAGE.read_text(encoding="utf-8")
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('name="ocr_model_id"', page)
        self.assertIn('name="vlm_model_id"', page)
        self.assertIn('renderImageModelOptions("agent-ocr-model", 6', script)
        self.assertIn('renderImageModelOptions("agent-vlm-model", 5', script)
        self.assertIn("image_capabilities: imageCapabilities", script)
        self.assertIn("allowed_model_ids: [ocrModelId]", script)
        self.assertIn("allowed_model_ids: [vlmModelId]", script)

    def test_agent_list_exposes_versioned_delete(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('data-agent-action="delete"', script)
        self.assertIn('method: "DELETE"', script)
        self.assertIn('"If-Match": `"rv-${agent.row_version}"`', script)

    def test_agent_delete_is_an_end_to_end_archive_operation(self) -> None:
        internal_api = INTERNAL_API.read_text(encoding="utf-8")
        client = CLIENT.read_text(encoding="utf-8")
        main_api = MAIN_API.read_text(encoding="utf-8")

        self.assertIn('@router.delete("/{agent_id}", status_code=204)', internal_api)
        self.assertIn("ArchiveAIOpsAgentCommand", internal_api)
        self.assertIn("async def delete_private_agent(", client)
        self.assertIn('@router.delete("/agents/{agent_id}"', main_api)


if __name__ == "__main__":
    unittest.main()
