"""AIOps 对话附件暂存与解析边界测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from aiops_agent.adapters.conversation_uploads import LocalConversationUploadStore
from aiops_agent.application.conversation_inputs import ConversationInputResolver


async def _chunks(*values: bytes):
    for value in values:
        yield value


class _ImageModel:
    def __init__(self) -> None:
        self.calls = []

    async def process(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "text": "截图显示 ORA-27157，发生于 03:16:10。",
            "model_revision": "vlm-test-r1",
        }


class _ImagePrompt:
    content = "从图片提取经过核对的运维证据。"

    @staticmethod
    def ref() -> dict[str, str]:
        return {
            "prompt_id": "aiops_agent.image_evidence_extract",
            "prompt_version": "1.0.0",
            "prompt_sha256": "a" * 64,
            "prompt_version_id": "01946b49-9f24-7f14-8000-000000000002",
            "prompt_source": "DATABASE",
        }


class _ImagePromptRegistry:
    def __init__(self) -> None:
        self.calls = []

    async def resolve(self, prompt_id: str):
        self.calls.append(prompt_id)
        return _ImagePrompt()


class ConversationUploadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = LocalConversationUploadStore(
            Path(self.temporary.name), max_bytes=1024, ttl_seconds=300
        )

    async def asyncTearDown(self):
        self.temporary.cleanup()

    async def test_text_upload_is_scoped_and_integrity_checked(self):
        stored = await self.store.store(
            domain_id=7,
            actor_id="user-1",
            file_name="../alert.log",
            media_type="text/plain; charset=utf-8",
            chunks=_chunks(b"ORA-", b"27157"),
        )
        self.assertEqual("alert.log", stored.file_name)
        self.assertEqual(b"ORA-27157", self.store.read(stored))
        preserved = self.store.preserve(stored)
        self.assertTrue(preserved.preserved)
        self.assertEqual(
            b"ORA-27157",
            self.store.read(
                self.store.get(
                    upload_id=stored.upload_id,
                    domain_id=7,
                    actor_id="user-1",
                )
            ),
        )
        self.assertEqual(
            b"ORA-27157",
            self.store.read_artifact(
                payload_uri=preserved.payload_uri,
                content_hash=preserved.content_hash,
                byte_size=preserved.byte_size,
            ),
        )
        with self.assertRaises(PermissionError):
            self.store.get(
                upload_id=stored.upload_id,
                domain_id=7,
                actor_id="user-2",
            )

    async def test_rejects_unsupported_media_type_and_oversize(self):
        with self.assertRaisesRegex(ValueError, "仅支持"):
            await self.store.store(
                domain_id=7,
                actor_id="user-1",
                file_name="archive.zip",
                media_type="application/zip",
                chunks=_chunks(b"zip"),
            )
        with self.assertRaisesRegex(ValueError, "超过"):
            await self.store.store(
                domain_id=7,
                actor_id="user-1",
                file_name="large.log",
                media_type="text/plain",
                chunks=_chunks(b"x" * 1025),
            )

    async def test_resolver_decodes_text_and_invokes_configured_vlm(self):
        text_upload = await self.store.store(
            domain_id=7,
            actor_id="user-1",
            file_name="top.sql",
            media_type="application/sql",
            chunks=_chunks("select * from v$sqlstats".encode()),
        )
        image_upload = await self.store.store(
            domain_id=7,
            actor_id="user-1",
            file_name="error.png",
            media_type="image/png",
            chunks=_chunks(b"not-a-real-image-but-bounded"),
        )
        image_model = _ImageModel()
        prompts = _ImagePromptRegistry()
        resolver = ConversationInputResolver(
            upload_store=self.store,
            image_model_client=image_model,
            prompt_registry=prompts,
            max_extracted_chars=1000,
        )
        content, uploads = await resolver.resolve(
            domain_id=7,
            actor_id="user-1",
            content=(
                {
                    "content_type": "FILE",
                    "upload_id": text_upload.upload_id,
                },
                {
                    "content_type": "IMAGE",
                    "upload_id": image_upload.upload_id,
                },
            ),
            image_capabilities={
                "vlm": {
                    "default_model_id": "01946b49-9f24-7f14-8000-000000000001"
                }
            },
        )
        self.assertIn("v$sqlstats", content[0]["text"])
        self.assertIn("ORA-27157", content[1]["text"])
        self.assertEqual("TEXT_DECODE", uploads[0].extraction_mode)
        self.assertEqual("VLM", uploads[1].extraction_mode)
        self.assertIsInstance(uploads[1].model_id, UUID)
        self.assertEqual(1, len(image_model.calls))
        self.assertEqual(["image_evidence_extract"], prompts.calls)
        self.assertEqual(_ImagePrompt.content, image_model.calls[0]["prompt_content"])
        self.assertEqual(
            "aiops_agent.image_evidence_extract",
            uploads[1].prompt_ref["prompt_id"],
        )


if __name__ == "__main__":
    unittest.main()
