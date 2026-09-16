"""生成契约的 JSON 传输必须能承载图片字节。"""

import json
import unittest

from platform_core.contracts import GeneratedImageArtifact, ImageGenerationResult


JPEG = b"\xff\xd8\xff\xdb" + b"\x00" * 32
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


class GenerativeImageJsonTest(unittest.TestCase):
    def test_jpeg_artifact_json_roundtrip_keeps_bytes(self):
        result = ImageGenerationResult(
            status="COMPLETED",
            artifacts=(GeneratedImageArtifact(mime_type="image/jpeg", content=JPEG),),
        )
        payload = result.model_dump(mode="json")
        encoded = json.dumps(payload)
        self.assertNotIn("\ufffd", encoded)
        restored = ImageGenerationResult.model_validate(json.loads(encoded))
        self.assertEqual(JPEG, restored.artifacts[0].content)
        self.assertEqual("image/jpeg", restored.artifacts[0].mime_type)

    def test_png_artifact_json_dump_does_not_use_utf8(self):
        artifact = GeneratedImageArtifact(mime_type="image/png", content=PNG)
        payload = artifact.model_dump(mode="json")
        self.assertIsInstance(payload["content"], str)
        restored = GeneratedImageArtifact.model_validate(payload)
        self.assertEqual(PNG, restored.content)


if __name__ == "__main__":
    unittest.main()
