"""LLM Reranker — 复用现有 LLM，判断 chunk 是否与问题相关。

替代已移除的 Cross-Encoder Reranker (TxtBaseRerank)。
在原始 chunk 内容上执行判断（先 rerank 后上下文扩展），
OP 零新组件部署。
"""
import asyncio
from loguru import logger
from .result import TxtBaseSearchResult
from utils.clients import AIModelClient


class LLMReranker:
    """LLM 逐条相关性判断 — 复用现有 LLM，零新部署，OP 友好"""

    JUDGE_PROMPT = """判断以下文档片段是否能帮助回答用户的问题。
只回答 YES 或 NO。

用户问题：{question}

文档片段（标题: {header}，章节: {hierarchy}）：
{content}

这条文档片段能帮助回答用户的问题吗？"""

    def __init__(self):
        self.model_client = AIModelClient()

    async def rerank(
        self,
        results: list[TxtBaseSearchResult],
        question: str,
        llm_model: str,
        top_k: int = 15,
        parallel: int = 5,
    ) -> list[TxtBaseSearchResult]:
        """对检索结果的原始 chunk 内容逐条判断相关性。

        关键：在上下文扩展之前执行，使用原始 chunk content。
        NO 的结果仅降权不删除（防误杀）。

        Args:
            results: 检索结果列表
            question: 用户原始问题
            llm_model: 用于判断的 LLM 模型名
            top_k: 对前 top_k 条做判断
            parallel: 并发数

        Returns:
            按 rerank_score 降序排列的结果列表
        """
        candidates = results[:top_k]

        sem = asyncio.Semaphore(parallel)

        async def judge_one(r: TxtBaseSearchResult) -> tuple[TxtBaseSearchResult, str]:
            async with sem:
                hierarchy = " > ".join(getattr(r, 'hierarchy_path', []) or [])
                prompt = self.JUDGE_PROMPT.format(
                    question=question,
                    header=r.header,
                    hierarchy=hierarchy or "根目录",
                    content=r.content[:800],  # 原始 chunk，非扩展后
                )
                try:
                    verdict = await self.model_client.get_llm_answer(
                        model_name=llm_model,
                        prompt=prompt,
                        temperature=0,
                        max_tokens=5,
                    )
                    return r, verdict.strip().upper()
                except Exception:
                    return r, "YES"  # 调用失败时保留，避免误删

        tasks = [judge_one(r) for r in candidates]
        judged = await asyncio.gather(*tasks)

        kept: list[TxtBaseSearchResult] = []
        yes_count = 0
        for r, verdict in judged:
            if "YES" in verdict:
                r.rerank_score = r.score * 1.2
                yes_count += 1
            else:
                r.rerank_score = r.score * 0.3  # 大幅降权但不删除
            kept.append(r)

        kept.sort(key=lambda x: x.rerank_score, reverse=True)
        logger.debug(
            f"[LLMReranker] 判断完成: {yes_count}/{len(kept)} 条确认为相关"
        )
        return kept
