"""AIOps 事件流的空闲保活、慢上游、续传分页与取消清理回归。"""

import asyncio
import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from main_api.api.aiops_events import stream_aiops_event_pages
from main_api.api import ops
from platform_core.identity import uuid7


class AIOpsEventStreamTest(unittest.IsolatedAsyncioTestCase):
    def request(self):
        self.disconnected = False
        settings = SimpleNamespace(
            sse_batch_size=2, sse_poll_interval_seconds=0.001,
            sse_heartbeat_seconds=0.01,
        )
        service_settings = SimpleNamespace(api=settings)
        return SimpleNamespace(
            is_disconnected=AsyncMock(side_effect=lambda: self.disconnected),
            state=SimpleNamespace(auth_context=object()),
            app=SimpleNamespace(state=SimpleNamespace(
                main_api_settings=service_settings,
                ammolite_main_api_settings=service_settings,
            )),
        ), settings

    def stream(self, fetch_page, cursor=0):
        request, settings = self.request()
        return stream_aiops_event_pages(
            request=request, fetch_page=fetch_page, cursor=cursor, settings=settings
        )

    async def test_idle_pages_emit_heartbeat_without_business_sequence(self):
        fetch = AsyncMock(return_value={"events": [], "terminal": False})
        stream = self.stream(fetch)
        self.assertEqual(": connected\n\n", await anext(stream))
        heartbeat = await asyncio.wait_for(anext(stream), 1)
        self.assertEqual(": heartbeat\n\n", heartbeat)
        self.assertGreater(fetch.await_count, 1)
        self.assertEqual((0, 2), fetch.await_args.args)
        await stream.aclose()

    async def test_slow_upstream_keeps_one_request_and_is_cancelled_on_close(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def fetch(*_args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        fetch_mock = AsyncMock(side_effect=fetch)
        stream = self.stream(fetch_mock, cursor=9)
        await anext(stream)
        for _ in range(2):
            self.assertEqual(": heartbeat\n\n", await asyncio.wait_for(anext(stream), 1))
        self.assertTrue(started.is_set())
        fetch_mock.assert_awaited_once_with(9, 2)
        await stream.aclose()
        self.assertTrue(cancelled.is_set())

    async def test_disconnect_cancels_pending_upstream(self):
        cancelled = asyncio.Event()

        async def fetch(*_args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        stream = self.stream(fetch)
        await anext(stream)
        await anext(stream)
        self.disconnected = True
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)
        self.assertTrue(cancelled.is_set())

    async def test_cancelling_consumer_cleans_up_pending_request(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def fetch(*_args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        stream = self.stream(fetch)
        await anext(stream)
        consumer = asyncio.create_task(anext(stream))
        await started.wait()
        consumer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await consumer
        self.assertTrue(cancelled.is_set())

    async def test_resume_drains_terminal_pages_before_done(self):
        def event(sequence):
            return {"sequence_no": sequence, "event_type": "answer.delta",
                    "payload": {"delta": "已验证的结果"}}

        fetch = AsyncMock(side_effect=[
            {"events": [event(8), event(9)], "terminal": True},
            {"events": [event(10)], "terminal": True},
        ])
        frames = [frame async for frame in self.stream(fetch, cursor=7)]
        self.assertEqual([7, 9], [call.args[0] for call in fetch.await_args_list])
        self.assertEqual(3, sum("event: answer.delta" in frame for frame in frames))
        self.assertEqual('event: done\ndata: {"sequence_no": 10}\n\n', frames[-1])
        self.assertEqual(1, sum("event: done" in frame for frame in frames))

    async def test_upstream_error_is_not_disguised_as_done(self):
        stream = self.stream(AsyncMock(side_effect=RuntimeError("读取失败")))
        await anext(stream)
        with self.assertRaisesRegex(RuntimeError, "读取失败"):
            await anext(stream)

    async def test_turn_route_keeps_access_check_and_resume_context(self):
        module = ops if hasattr(ops, "stream_conversation_turn_events") else importlib.import_module("main_api.api.aiops_app")
        request, _ = self.request()
        client = SimpleNamespace(list_conversation_turn_events=AsyncMock(
            return_value={"events": [], "terminal": True}
        ))
        conversation_id, turn_id = uuid7(), uuid7()
        with patch.object(module, "_conversation_with_access", new_callable=AsyncMock) as access, patch.object(module, "_client", return_value=client):
            response = await module.stream_conversation_turn_events(
                conversation_id, turn_id, request, last_event_id="7"
            )
            frames = [frame async for frame in response.body_iterator]
        access.assert_awaited_once_with(request, conversation_id)
        client.list_conversation_turn_events.assert_awaited_once_with(
            conversation_id, turn_id, after_sequence=7, limit=2,
            auth_context=request.state.auth_context,
        )
        self.assertEqual("no", response.headers["X-Accel-Buffering"])
        self.assertIn("event: done", frames[-1])

    async def test_run_route_uses_same_heartbeat_and_resume_contract(self):
        request, _ = self.request()
        client = SimpleNamespace(list_run_events=AsyncMock(
            return_value={"events": [], "terminal": True}
        ))
        run_id = uuid7()
        with patch.object(ops, "_client", return_value=client):
            response = await ops.stream_ops_run_events(run_id, request, last_event_id="3")
            frames = [frame async for frame in response.body_iterator]
        client.list_run_events.assert_awaited_once_with(
            run_id, after_sequence=3, limit=2, auth_context=request.state.auth_context
        )
        self.assertIn("event: done", frames[-1])


if __name__ == "__main__":
    unittest.main()
