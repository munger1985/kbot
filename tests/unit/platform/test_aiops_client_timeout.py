"""AIOps Client 独立超时语义测试。"""

import unittest

import aiohttp

from platform_clients.aiops import _BaseAIOpsClient


class _Auth:
    def headers(self, context):
        del context
        return {}


class _Response:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        return None

    async def json(self):
        return {"status": "ok"}


class _SharedSession:
    def __init__(self):
        self.closed = False
        self.request_kwargs = None

    def request(self, method, url, **kwargs):
        del method, url
        self.request_kwargs = kwargs
        return _Response()


class AIOpsClientTimeoutTest(unittest.IsolatedAsyncioTestCase):
    async def test_shared_session_does_not_override_aiops_timeout(self):
        session = _SharedSession()
        client = _BaseAIOpsClient(
            base_url="http://aiops.internal",
            auth=_Auth(),
            timeout_seconds=17,
            session=session,
        )

        result = await client._json(
            "GET",
            "/internal/v1/test",
            auth_context=object(),
        )

        self.assertEqual({"status": "ok"}, result)
        timeout = session.request_kwargs["timeout"]
        self.assertIsInstance(timeout, aiohttp.ClientTimeout)
        self.assertEqual(17, timeout.total)


if __name__ == "__main__":
    unittest.main()
