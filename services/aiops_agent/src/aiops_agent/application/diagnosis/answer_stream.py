"""单次模型正文的追加式交付；不把原文保存等同于证据核验通过。"""

from aiops_agent.contracts.turn_answer import AnswerNarrative
from aiops_agent.ports.model import ModelTextDelta, TextDeltaCallback


class AnswerStreamCapture:
    def __init__(self, *, run_id, callback: TextDeltaCallback | None = None):
        self.run_id = str(run_id)
        self.callback = callback
        self.chunks: list[str] = []

    @property
    def text(self) -> str:
        return "".join(self.chunks)

    async def on_delta(self, chunk: ModelTextDelta) -> None:
        if chunk.reset:
            if self.chunks:
                raise ValueError("已经开始交付的正文不能重新生成或清空")
            return
        if chunk.delta:
            self.chunks.append(chunk.delta)
            if self.callback is not None:
                await self.callback(chunk)

    def snapshot(self, *, complete=False, verified=False, notice=None) -> AnswerNarrative | None:
        if not self.chunks:
            return None
        return AnswerNarrative(markdown=self.text,
            generation_status="COMPLETED" if complete else "INTERRUPTED",
            validation_status="VERIFIED" if verified else "UNVERIFIED",
            streamed_run_id=self.run_id if self.callback is not None else None, notice=notice)


def narrative_payload(narrative: AnswerNarrative) -> dict[str, str | bool | None]:
    """展示状态不能充当调查结论、可执行 SQL 或审批依据。"""
    return {"markdown": narrative.markdown, "raw_model_output": True,
            "generation_status": narrative.generation_status,
            "validation_status": narrative.validation_status,
            "answer_notice": narrative.notice}
