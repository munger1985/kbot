"""模型客户端结构化输出参数边界测试。"""

import json
import unittest

from platform_clients.model import AIModelClient


class _RecordingModelClient(AIModelClient):
    def __init__(self):
        super().__init__(caller_service="unit-test")
        self.request_kwargs = None

    async def call_llm_model(self, served_model_name, prompt, **kwargs):
        self.request_kwargs = kwargs
        yield json.dumps(
            {
                "choices": [
                    {"message": {"content": '{"status":"ok"}'}}
                ]
            }
        )


class AIModelClientJsonTest(unittest.IsolatedAsyncioTestCase):
    async def test_json_call_uses_model_configured_max_tokens(self):
        client = _RecordingModelClient()

        result = await client.get_llm_json(
            served_model_name="configured-model",
            prompt="返回状态",
        )

        self.assertEqual({"status": "ok"}, result)
        self.assertNotIn("max_tokens", client.request_kwargs)
        self.assertEqual(False, client.request_kwargs["stream"])


if __name__ == "__main__":
    unittest.main()
