"""原生正文流、有限校正、失败撤回及单次诊断复用的回归。"""

import asyncio
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from aiops_agent.adapters.model_serving import AIOpsModelError, AIOpsStructuredModelClient
from aiops_agent.ports.model import ModelTextDelta
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler, _model_answer_stream
from platform_core.identity import uuid7


class Output(BaseModel):
    answer_markdown: str
    verified_count: int


def frame(content=None, *, finish=None, reasoning=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning:
        delta["reasoning_content"] = reasoning
    return ("data: " + json.dumps({
        "id": "provider-stream",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }, ensure_ascii=False) + "\r\n\r\n").encode()


class Response:
    status = 200

    def __init__(self, parts, *, gate=None):
        self.parts, self.gate = parts, gate
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def iter_any(self):
        for index, part in enumerate(self.parts):
            if index == 1 and self.gate is not None:
                await self.gate.wait()
            # 每字节分片同时覆盖中文、emoji、转义及 CRLF 跨网络片段。
            for byte in part:
                yield bytes([byte])


class Session:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.payloads = []

    def post(self, _url, **kwargs):
        self.payloads.append(kwargs["json"])
        return self.responses.pop(0)


def client(session):
    return AIOpsStructuredModelClient(base_url="http://model", audience="model",
        caller_service="aiops", timeout_seconds=30, session=session)


async def generate(session, callback, *, output_model=Output):
    prompt = "只依据已授权证据。"
    with patch("aiops_agent.adapters.model_serving.build_internal_auth_headers", return_value={}):
        return await client(session).generate_structured(
            purpose="test", output_model=output_model,
            model_snapshot={"technical_name": "test", "model_id": uuid7(), "tenant_id": uuid7(), "revision": "1"},
            prompt_ref={"prompt_id": "test", "prompt_version": "1.0.0", "content": prompt,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()},
            input_payload={}, deadline=None, idempotency_key="test",
            text_field="answer_markdown", on_text_delta=callback)


def stream_response(payload, *, finish="stop", gate=None):
    content = json.dumps(payload, ensure_ascii=False)
    return Response([frame(content, reasoning="不得泄露的推理"), frame(finish=finish), b"data: [DONE]\r\n\r\n"], gate=gate)


def test_first_text_arrives_before_remaining_structured_response():
    async def exercise():
        gate, first = asyncio.Event(), asyncio.Event()
        markdown = '## 分析\n中文😀与转义引号"、反斜线\\保持原文。\n\n## 方案\n先核对证据。'
        encoded = json.dumps({"answer_markdown": markdown, "verified_count": 1}, ensure_ascii=False)
        split = encoded.index(', "verified_count"')
        session = Session(Response([frame(encoded[:split], reasoning="不能显示"), frame(encoded[split:], finish="stop")], gate=gate))
        deltas = []

        async def callback(delta):
            deltas.append(delta)
            if delta.delta:
                first.set()

        task = asyncio.create_task(generate(session, callback))
        try:
            await asyncio.wait_for(first.wait(), 2)
            assert not task.done(), "正文不应等待完整 JSON 或结构化核验"
            assert "".join(item.delta for item in deltas) == markdown
            gate.set()
            result = await asyncio.wait_for(task, 2)
        finally:
            gate.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        assert result.output.answer_markdown == markdown
        assert deltas[0].reset
        assert session.payloads[0]["stream"] is True
        assert session.payloads[0]["response_format"] == {"type": "json_object"}
        assert "max_tokens" not in session.payloads[0]
        assert result.receipt.provider_request_id == "provider-stream"
        assert not any("不能显示" in item.delta or "verified_count" in item.delta for item in deltas)
    asyncio.run(exercise())


def test_done_finishes_without_waiting_for_connection_close():
    class DoneResponse(Response):
        async def iter_any(self):
            yield frame(json.dumps({"answer_markdown": "正文", "verified_count": 1}), finish="stop")
            yield b"data: [DONE]\r\n\r\n"
            raise AssertionError("收到 DONE 后不应继续等待连接关闭")

    async def exercise():
        result = await generate(Session(DoneResponse([])), AsyncMock())
        assert result.output.answer_markdown == "正文"
    asyncio.run(exercise())


def test_schema_repair_resets_previous_draft_and_uses_same_bounded_call():
    async def exercise():
        session = Session(
            stream_response({"answer_markdown": "未通过草稿\n", "verified_count": "invalid"}),
            stream_response({"answer_markdown": "已修正正文\n", "verified_count": 1}))
        visible, history = "", []

        async def callback(delta):
            nonlocal visible
            visible = ("" if delta.reset else visible) + delta.delta
            history.append(delta)

        result = await generate(session, callback)
        assert visible == result.output.answer_markdown == "已修正正文\n"
        assert sum(delta.reset for delta in history) == 2
        assert len(session.payloads) == 2
    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["interrupted", "length", "content_filter", "invalid"])
def test_incomplete_or_invalid_stream_never_returns_success(failure):
    async def exercise():
        payload = {"answer_markdown": "临时草稿\n", "verified_count": 1}
        if failure == "interrupted":
            response = Response([frame('{"answer_markdown":"临时草稿\\n"')])
        else:
            if failure == "invalid":
                payload["verified_count"] = "invalid"
            response = stream_response(payload, finish=failure if failure != "invalid" else "stop")
        session = Session(response, response)
        callback = AsyncMock()
        with pytest.raises(AIOpsModelError) as error:
            await generate(session, callback)
        assert error.value.code == ("MODEL_STREAM_INTERRUPTED" if failure == "interrupted" else "MODEL_OUTPUT_INVALID")
        assert len(session.payloads) <= 2
    asyncio.run(exercise())


def test_sql_checkpoint_keeps_original_text_and_compose_does_not_regenerate_or_replay():
    from tests.unit.aiops_agent.test_aiops_sql_review import setup, review
    from tests.support.aiops_sql_review import diagnostic_response

    async def exercise():
        fact, task_frame, ctx, model, prompts, assessment = setup()
        gate = asyncio.Event()
        native = "## 分析\n已有观测支持局部判断。\n\n## 方案\n暂不变更。\n"

        async def generate_review(**kwargs):
            await kwargs["on_text_delta"](ModelTextDelta(reset=True))
            await kwargs["on_text_delta"](ModelTextDelta(delta=native))
            await gate.wait()
            output = diagnostic_response(review(fact)).model_dump()
            output["answer_markdown"] = native
            return SimpleNamespace(output=kwargs["output_model"].model_validate(output), receipt=None)

        model.generate_structured.side_effect = generate_review
        assessor = DbaEvidenceAssessmentHandler(model_client=model, prompts=prompts)

        async def execute(ctx, *, on_text_delta):
            return await assessor._review_sql(ctx, assessment, task_frame, on_text_delta=on_text_delta)

        stream = _model_answer_stream(ctx, execute)
        reset, delta = await anext(stream), await anext(stream)
        assert reset.payload["reset"] and delta.payload["delta"] == native
        assert delta.payload["validation_status"] == "PENDING"
        gate.set()
        assessed = await anext(stream)
        await stream.aclose()
        assert assessed.sql_checkpoint.answer_markdown == native
        assert assessed.sql_checkpoint.streamed_run_id == str(ctx.run_id)
        answer_ctx = replace(ctx, input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": assessed.model_dump(mode="json")},))
        composer = DbaAnswerComposeHandler(model_client=model, prompts=prompts)
        items = [item async for item in composer.execute_stream(answer_ctx)]
        assert len(items) == 1 and items[0].answer_streamed
        blocks = [block for block in items[0].blocks if block.block_type == "MARKDOWN"]
        assert len(blocks) == 1 and blocks[0].payload["markdown"] == native
        assert blocks[0].payload["raw_model_output"]
        assert "执行计划：" not in blocks[0].payload["markdown"]
        # 不同运行复用历史检查点时必须回放正文，不能误判为已向本轮输出。
        new_ctx = replace(answer_ctx, run_id=uuid7())
        replay = [item async for item in composer.execute_stream(new_ctx)]
        assert any(getattr(item, "event_type", None) == "answer.delta" for item in replay)
        model.generate_structured.assert_awaited_once()
    asyncio.run(exercise())


def test_sql_validation_failure_retracts_draft():
    from tests.unit.aiops_agent.test_aiops_sql_review import setup

    async def exercise():
        _, task_frame, ctx, model, prompts, assessment = setup()
        deltas = []

        async def generate_review(**kwargs):
            await kwargs["on_text_delta"](ModelTextDelta(delta="未核验内容"))
            raise AIOpsModelError("MODEL_OUTPUT_INVALID")

        async def callback(delta):
            deltas.append(delta)

        model.generate_structured.side_effect = generate_review
        result = await DbaEvidenceAssessmentHandler(model_client=model, prompts=prompts)._review_sql(
            ctx, assessment, task_frame, on_text_delta=callback)
        assert deltas[-1].reset
        assert result.sql_checkpoint is None
        assert result.sql_diagnosis_error == "MODEL_OUTPUT_INVALID"
    asyncio.run(exercise())


@pytest.mark.parametrize("valid", [True, False])
def test_general_diagnosis_stream_preserves_record_validation(valid):
    from tests.unit.aiops_agent.test_aiops_investigation_record import draft, evidence
    from tests.unit.aiops_agent.test_aiops_sql_review import setup
    from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment

    async def exercise():
        _, _, ctx, model, prompts, _ = setup()
        ctx.plan_snapshot["answer_context"]["task_frame"] = {"objectives": ["DIAGNOSE"]}
        assessment = DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(evidence(),))
        ctx = replace(ctx, input_artifacts=({"schema_version": "DBA_SUFFICIENCY.v1", "payload": assessment.model_dump(mode="json")},))
        native = "## 分析\n会话 12 正在阻塞会话 88。\n\n## 方案\n先核对事务。\n"

        async def generate_answer(**kwargs):
            await kwargs["on_text_delta"](ModelTextDelta(reset=True))
            await kwargs["on_text_delta"](ModelTextDelta(delta=native))
            payload = draft().model_dump()
            if not valid:
                payload["claims"][0]["evidence_fields"][0]["field"] = "nonexistent"
            return SimpleNamespace(output=kwargs["output_model"].model_validate({**payload, "answer_markdown": native}),
                receipt=SimpleNamespace(model_dump=lambda **_: {}))

        model.generate_structured.side_effect = generate_answer
        items = [item async for item in DbaAnswerComposeHandler(model_client=model, prompts=prompts).execute_stream(ctx)]
        result = items[-1]
        if valid:
            assert result.answer_streamed
            assert result.investigation_record.claims[0].statement == "会话 12 正在阻塞会话 88。"
            assert next(block for block in result.blocks if block.payload.get("raw_model_output")).payload["markdown"] == native
        else:
            resets = [item for item in items[:-1] if item.event_type == "answer.delta" and item.payload.get("reset")]
            assert len(resets) == 2
            assert not any(block.payload.get("raw_model_output") for block in result.blocks)
        model.generate_structured.assert_awaited_once()
    asyncio.run(exercise())


def test_closing_progress_generator_cancels_background_model():
    async def exercise():
        cancelled = asyncio.Event()

        async def execute(ctx, *, on_text_delta):
            try:
                await on_text_delta(ModelTextDelta(delta="生成中"))
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        stream = _model_answer_stream(SimpleNamespace(attempt=1), execute)
        assert (await anext(stream)).payload["delta"] == "生成中"
        await stream.aclose()
        assert cancelled.is_set()
    asyncio.run(exercise())
