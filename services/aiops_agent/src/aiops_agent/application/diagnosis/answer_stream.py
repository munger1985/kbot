"""单次模型正文的追加式交付；不把原文保存等同于证据核验通过。"""

from aiops_agent.contracts.turn_answer import AnswerNarrative, AIOpsTurnResult, DbaSufficiencyAssessment, TurnAnswerBlock
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


def assessment_has_answer(assessment: DbaSufficiencyAssessment) -> bool:
    """正文已经生成后，补证建议不再触发本轮重新调查和生成。"""
    return bool(
        assessment.answer_narrative and assessment.answer_narrative.markdown
        or assessment.sql_checkpoint and assessment.sql_checkpoint.answer_markdown
    )


def preserve_published_answer(result: AIOpsTurnResult, published: str) -> AIOpsTurnResult:
    """持久化与用户已见正文一致；核验失败不能撤回原文或授予执行权限。"""
    if not published:
        return result
    narrative_types = {"MARKDOWN", "ANALYSIS_MARKDOWN", "SOLUTION_MARKDOWN"}
    narrative = [block for block in result.blocks if block.block_type in narrative_types]
    final_text = "\n\n".join(str(block.payload.get("markdown", "")) for block in narrative)
    if final_text.startswith(published):
        return result.model_copy(update={"answer_streamed": True})
    replacement = TurnAnswerBlock(
        block_type="MARKDOWN", schema_version="AIOPS_MARKDOWN_BLOCK.v1",
        payload={"markdown": published, "raw_model_output": True,
                 "generation_status": "INTERRUPTED", "validation_status": "UNVERIFIED",
                 "answer_notice": "模型正文已保留；最终诊断未完成，不作为已核验结论或执行授权。"},
    )
    blocks = []
    inserted = False
    for block in result.blocks:
        if block.block_type in narrative_types:
            if not inserted:
                blocks.append(replacement)
                inserted = True
        else:
            blocks.append(block)
    if not inserted:
        blocks.insert(0, replacement)
    return result.model_copy(update={
        "blocks": tuple(blocks), "answer_streamed": True,
        "status": "WAITING_USER" if result.status == "WAITING_USER" else "PARTIAL",
    })
