# services/doc_service.py
import time
import asyncio
import re
from loguru import logger

from dao.repositories import AgentConfRepository
from services.search import TxtBaseSearchResult, TxtBaseSearch
from services.search.reranker import LLMReranker
from platform_clients import AIModelClient
from platform_core.exceptions import ParamValueError, InternalServerError
from services.kb import ModelParams
from platform_core.database import db_instance

# ---------------------------------------------------------------------------
# Phase 5: 多查询生成 prompt
# ---------------------------------------------------------------------------

_SUB_QUERY_PROMPT = """你是一个查询优化专家。请从不同角度改写用户问题，生成 2 个互补的检索查询。

要求：
1. 每个查询从不同角度或使用不同措辞表达相同的语义需求
2. 查询应简洁、具体，适合用于向量检索和关键词检索
3. 仅输出查询本身，每行一个，不要编号、引号或额外解释

用户问题：{question}

改写查询："""


def _dedup_by_chunk_id(
    results: list[TxtBaseSearchResult],
) -> list[TxtBaseSearchResult]:
    """按 chunk_id 去重，保留最高分的记录。"""
    seen: dict[str, TxtBaseSearchResult] = {}
    for r in results:
        if r.chunk_id not in seen or r.score > seen[r.chunk_id].score:
            seen[r.chunk_id] = r
    return list(seen.values())


class DocService:
    def __init__(self):
        self.tb_search = TxtBaseSearch()
        self.llm_reranker = LLMReranker()
        self.model_client = AIModelClient()

    async def get_knowledge_context(
        self,
        db_session,
        agent_id: int,
        question: str,
        keywords: str,
        security_level: int,
        model_params: ModelParams,
        tags: list[str] = [],
        enable_multi_query: bool = False,
    ) -> tuple[list[TxtBaseSearchResult], list[float]]:
        """
        核心业务流水线：多查询生成 → 向量化 → 并行检索 → 融池去重 → 重排序。

        Phase 5 改动：可选的多查询融合 — 从不同角度生成子查询，
        分别检索后合并去重，提升复杂问题的召回覆盖率。
        """
        # ------------------------------------------------------------------
        # Step 0 (Phase 5): 生成多角度子查询
        # ------------------------------------------------------------------
        sub_queries: list[tuple[str, str]] = []  # [(query_text, keywords_text), ...]

        # 主查询始终参与
        effective_keywords = keywords.strip() if keywords else ""
        if not effective_keywords:
            effective_keywords = question.strip() if question else ""
            logger.warning(
                f"[DocService] agent {agent_id} 的 keywords 为空，"
                f"使用原始问题作为检索词: '{effective_keywords[:80]}'"
            )

        sub_queries.append((question.strip(), effective_keywords))

        if enable_multi_query and len(question.strip()) > 10:
            try:
                variants = await self._generate_sub_queries(
                    question=question.strip(),
                    model_name=model_params.llm_model,
                )
                for v_text in variants:
                    if v_text and v_text != question.strip():
                        # 变体查询使用自身作为 keywords（也会经过 _process_keywords）
                        sub_queries.append((v_text, v_text))
                logger.debug(
                    f"[DocService] 多查询生成: {len(sub_queries)} 个查询变体"
                )
            except Exception as e:
                logger.warning(f"[DocService] 子查询生成失败，回退单查询模式: {e}")

        # ------------------------------------------------------------------
        # Step 1: 向量化所有子查询
        # ------------------------------------------------------------------
        logger.debug(
            f"[DocService] Step 1/4: 获取嵌入向量，model={model_params.txt_embedding_model}"
        )
        all_vecs: list[list[float]] = []
        for q_text, _ in sub_queries:
            vec = await self._get_embedding(
                q_text, model_params.txt_embedding_model, agent_id
            )
            all_vecs.append(vec)
        logger.debug(
            f"[DocService] Step 1/4: {len(all_vecs)} 个嵌入向量获取完成"
        )

        # ------------------------------------------------------------------
        # Step 2: 获取知识库配置
        # ------------------------------------------------------------------
        logger.debug(f"[DocService] Step 2/4: 获取知识库配置")
        conf_repo = AgentConfRepository(db_session)
        agent_confs = await conf_repo.get_by_agent(agent_id)
        logger.debug(
            f"[DocService] Step 2/4: 获取到 {len(agent_confs)} 个知识库配置"
        )

        # ------------------------------------------------------------------
        # Step 3: 并行检索（多查询 × 多知识库）
        # ------------------------------------------------------------------
        logger.info(f"开始为智能体 {agent_id} 执行知识库检索，安全等级：{security_level}")
        start_time = time.time()

        # 构建所有 (query_text, query_vec) × KB 配置的任务
        tb_tasks = []
        for q_idx, (q_text, q_keywords) in enumerate(sub_queries):
            query_vec = all_vecs[q_idx]
            for conf in agent_confs:
                tb_tasks.append(self._search_single_kb(
                    conf.kb_id,
                    str(q_keywords),
                    int(conf.search_top_k or 10),
                    float(conf.search_score_threshold or 0.0),
                    float(conf.tool_weight or 1.0),
                    int(security_level),
                    query_vec,
                    tags
                ))

        # 并行执行所有任务
        raw_results = await asyncio.gather(*tb_tasks, return_exceptions=True)
        logger.debug(
            f"[DocService] Step 3/4: 并行检索完成，{len(raw_results)} 个任务"
        )

        retrieved_results: list[TxtBaseSearchResult] = []
        for i, res in enumerate(raw_results):
            if isinstance(res, Exception):
                logger.error(f"知识库任务 {i} 执行失败：{res}")
                continue
            elif isinstance(res, list):
                retrieved_results.extend(res)
                logger.info(f"知识库任务 {i} 成功返回 {len(res)} 条结果")
            else:
                logger.warning(f"知识库任务 {i} 返回结果为空或格式错误")

        # Phase 5: 多查询结果去重（按 chunk_id 保留最高分）
        if len(sub_queries) > 1:
            before_dedup = len(retrieved_results)
            retrieved_results = _dedup_by_chunk_id(retrieved_results)
            # 按分数重新排序
            retrieved_results.sort(key=lambda x: x.score, reverse=True)
            logger.debug(
                f"[DocService] 多查询去重: {before_dedup} → {len(retrieved_results)}"
            )

        # ------------------------------------------------------------------
        # Step 4: 重排序
        # ------------------------------------------------------------------
        logger.debug(
            f"[DocService] Step 4/4: 开始重排序，待排文档数={len(retrieved_results)}"
        )
        # 使用主查询（第一个子查询）的问题文本做 rerank 相关性判断
        final_results = await self._apply_rerank(
            retrieved_results, sub_queries[0][0], model_params
        )

        logger.info(
            f"知识库检索完成，耗时：{time.time() - start_time:.2f}s，"
            f"最终返回 {len(final_results)} 条结果"
        )
        return final_results, all_vecs[0]

    # ------------------------------------------------------------------
    # Phase 5: 子查询生成
    # ------------------------------------------------------------------

    async def _generate_sub_queries(
        self, question: str, model_name: str
    ) -> list[str]:
        """
        使用 LLM 从不同角度生成 2 个互补检索查询。

        Returns:
            子查询文本列表（不含原始问题）
        """
        prompt = _SUB_QUERY_PROMPT.format(question=question)

        try:
            response = await self.model_client.get_llm_answer(
                model_name=model_name,
                prompt=prompt,
                temperature=0.3,
                max_tokens=200,
            )
        except Exception as e:
            logger.warning(f"[DocService] LLM 子查询生成调用失败: {e}")
            return []

        if not response:
            return []

        # 解析响应：每行一个查询
        lines = [
            line.strip()
            for line in str(response).split("\n")
            if line.strip()
        ]
        # 过滤掉明显的编号前缀和引号
        cleaned = []
        for line in lines:
            line = re.sub(r'^[\d]+[\.\、\)]\s*', '', line)
            line = line.strip('"\'""''').strip()
            if line and len(line) > 3:
                cleaned.append(line)

        logger.debug(f"[DocService] 生成子查询: {cleaned}")
        return cleaned[:2]  # 最多 2 个额外子查询

    # ------------------------------------------------------------------
    # 原有方法
    # ------------------------------------------------------------------

    async def _get_embedding(
        self, question: str, model_name: str, agent_id: int
    ) -> list[float]:
        question = question.strip() if question else ""
        if not question:
            raise ParamValueError(f"智能体 {agent_id} 的检索问题不能为空")

        vec = await self.model_client.get_embedding(model_name, question)
        if not vec:
            raise InternalServerError("嵌入向量生成失败")
        return vec

    async def _apply_rerank(
        self,
        results: list[TxtBaseSearchResult],
        question: str,
        params: ModelParams,
    ) -> list[TxtBaseSearchResult]:
        """LLM Reranker — 在原始 chunk 上逐条判断相关性"""
        # if not params.do_rerank or len(results) <= 1:
        #     logger.debug("检索结果数量不足或未启用重排序，直接返回原始结果")
        #     return results

        llm_model = params.llm_model_light or params.llm_model
        return await self.llm_reranker.rerank(
            results=results,
            question=question,
            llm_model=llm_model,
            top_k=params.rerank_top_k or 10,
            min_keep=3,
        )

    async def _search_single_kb(
        self, kb_id: int, keywords: str, top_k: int,
        threshold: float, weight: float, security: int,
        query_vec: list[float] | None, tags: list[str]
    ) -> list[TxtBaseSearchResult]:
        """单个知识库的检索封装，使用独立 session"""
        async with db_instance().get_session() as session:
            return await self.tb_search.search(
                kb_id=kb_id,
                keywords=keywords,
                search_top_k=top_k,
                threshold=threshold,
                weight=weight,
                security=security,
                query_vec=query_vec,
                tags=tags,
            )
