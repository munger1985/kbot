"""单一部署配置可显式开启知识表示和资源预算，不接受任意服务覆盖。"""

from pathlib import Path
import tempfile
import unittest

from platform_core.config import load_settings
from knowledge_core.config import KnowledgeCoreSettings
from knowledge_retrieval_app.config import KnowledgeRetrievalAppSettings


class KnowledgeConfigurationTests(unittest.TestCase):
    def test_fixed_sections_are_loaded_by_their_owner_only(self):
        oracle = (Path(__file__).resolve().parents[3] / "database/oracle").is_dir()
        compute_name = "knowledge_core" if oracle else "ammolite_knowledge_core"
        app_name = "knowledge_retrieval_app" if oracle else "ammolite_knowledge_retrieval_app"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.toml"
            path.write_text("environment='development'\ndata_dir='/tmp/knowledge-config-fixture'\n"
                "[knowledge.computation]\nmemory_bytes=1073741824\nmax_jobs=1\n"
                "[knowledge.knowledge_retrieval]\npayload_tokens=8192\npayload_images=3\n"
                "[knowledge.aiops]\npayload_tokens=4096\n", encoding="utf-8")
            computation = load_settings(KnowledgeCoreSettings, service=compute_name, config_file=path)
            app = load_settings(KnowledgeRetrievalAppSettings, service=app_name, config_file=path)
            self.assertEqual(1073741824, computation.compute.memory_bytes)
            self.assertEqual(1, computation.compute.max_jobs)
            self.assertEqual(8192, app.knowledge.payload_tokens)
            self.assertEqual(3, app.knowledge.payload_images)
            self.assertFalse(hasattr(computation, "knowledge"))
            with path.open("a", encoding="utf-8") as output:
                output.write("[knowledge.foreign_app]\nurl='https://invalid.example'\n")
            with self.assertRaisesRegex(ValueError, "固定能力"):
                load_settings(KnowledgeRetrievalAppSettings, service=app_name, config_file=path)


if __name__ == "__main__":
    unittest.main()
