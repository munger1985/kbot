"""LLM Reranker — 复用现有 LLM，判断 chunk 是否与问题相关。

替代已移除的 Cross-Encoder Reranker (TxtBaseRerank)。
使用 origin_content（原始未扩展 chunk）做相关性判断，避免被上下文扩展污染。
通过自适应阈值过滤低相关结果，保底保留 min_keep 条防误杀。
OP 零新组件部署。
"""
import asyncio
from loguru import logger
from .result import TxtBaseSearchResult
from utils.clients import AIModelClient


class LLMReranker:
    """LLM 逐条相关性判断 — 支持自适应阈值过滤，保底 min_keep 条"""

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
        min_keep: int = 3,
    ) -> list[TxtBaseSearchResult]:
        """对检索结果的原始 chunk 内容逐条判断相关性。

        使用 origin_content（原始未扩展内容）而非 content（可能已被上下文扩展覆盖）。
        YES 结果得分上浮 20%，NO 结果降权至 30%。
        通过自适应阈值过滤低相关结果，保留至少 min_keep 条。

        Args:
            results: 检索结果列表
            question: 用户原始问题
            llm_model: 用于判断的 LLM 模型名
            top_k: 对前 top_k 条做判断
            parallel: 并发数
            min_keep: 过滤后最少保留条数（防误杀）

        Returns:
            按 rerank_score 降序排列的相关结果列表
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
                    content=(getattr(r, 'origin_content', None) or r.content)[:800],  # 优先使用扩展前的原始 chunk
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

        scored: list[TxtBaseSearchResult] = []
        yes_count = 0
        no_count = 0
        for r, verdict in judged:
            if "YES" in verdict:
                r.rerank_score = r.score * 1.2
                yes_count += 1
            else:
                r.rerank_score = r.score * 0.3
                no_count += 1
            scored.append(r)

        scored.sort(key=lambda x: x.rerank_score, reverse=True)

        # === 自适应阈值过滤 ===
        if not scored:
            return scored

        if len(scored) <= min_keep:
            logger.info(
                f"[LLMReranker] 判断完成: YES={yes_count}, NO={no_count}, "
                f"总数={len(scored)} ≤ 保底={min_keep}，免过滤"
            )
            return scored

        max_score = scored[0].rerank_score
        # 阈值 = 最高分的 35%
        # NO 的结果分数暴跌 (×0.3 vs YES ×1.2, 差距 4 倍)，
        # 35% 阈值能精准滤除绝大多数 NO 结果
        threshold = max_score * 0.35

        kept = [r for r in scored if r.rerank_score >= threshold]

        # 防误杀兜底：过滤后不足 min_keep 条，回填前 min_keep 条
        if len(kept) < min_keep:
            logger.warning(
                f"[LLMReranker] 过滤后仅 {len(kept)} 条，不足保底线 {min_keep}，"
                f"触发防误杀机制，回填保留前 {min_keep} 条"
            )
            return scored[:min_keep]

        logger.info(
            f"[LLMReranker] 判断完成: YES={yes_count}, NO={no_count}, "
            f"阈值={threshold:.4f}, {len(scored)} → {len(kept)} 条"
        )
        return kept
