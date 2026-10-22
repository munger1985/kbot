import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SHA256_SCRIPT = ROOT / "ui" / "shared" / "kbot-sha256.js"
UPLOAD_PAGE = ROOT / "ui" / "aiops" / "operations-knowledge.html"
UPLOAD_SCRIPT = ROOT / "ui" / "aiops" / "js" / "aiops-operations-knowledge.js"


class AIOpsUploadDigestContractTest(unittest.TestCase):
    def test_http_context_uses_sha256_fallback(self) -> None:
        program = r"""
const fs = require("node:fs");
const vm = require("node:vm");
const sandbox = { Uint8Array, Uint32Array, DataView, Object, Array };
vm.runInNewContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const samples = [
  [[], "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"],
  [[97, 98, 99], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"],
];
Promise.all(samples.map(async ([input, expected]) => {
  const actual = await sandbox.KBotSha256.bytes(new Uint8Array(input));
  if (actual !== expected) throw new Error(`${actual} != ${expected}`);
})).catch((error) => { console.error(error); process.exit(1); });
"""
        subprocess.run(
            ["node", "-e", program, str(SHA256_SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_upload_page_loads_digest_utility_before_upload_logic(self) -> None:
        page = UPLOAD_PAGE.read_text(encoding="utf-8")
        utility = "../shared/kbot-sha256.js?v=20261019_1"
        upload = "./js/aiops-operations-knowledge.js?v=20261022_1"
        self.assertIn(utility, page)
        self.assertLess(page.index(utility), page.index(upload))
        self.assertIn(
            "const digest = (file) => KBotSha256.file(file);",
            UPLOAD_SCRIPT.read_text(encoding="utf-8"),
        )

    def test_upload_copy_and_delete_confirmation_match_approval_first_flow(self) -> None:
        page = UPLOAD_PAGE.read_text(encoding="utf-8")
        script = UPLOAD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("上传技术文档后先由人工审批", page)
        self.assertIn('id="submit-upload" class="primary" type="submit">上传</button>', page)
        self.assertNotIn("上传并识别标签", page)
        self.assertIn("资料已上传，等待人工审批", script)
        self.assertIn('id="delete-asset-dialog"', page)
        self.assertIn("同时删除物理文件", page)
        self.assertIn("共享文件会保留", page)
        self.assertIn("purge_source_objects=true", script)
        self.assertNotIn("confirm(`确认删除运维资料", script)


if __name__ == "__main__":
    unittest.main()
