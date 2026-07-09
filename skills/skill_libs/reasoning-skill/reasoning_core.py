from loguru import logger
from typing import Any, AsyncGenerator
from utils.clients import AIModelClient
from agent.prompt import default_prompt
from core.config import get_prompt_config
from core.dictionary import PacketType
from skills import BaseSkill
from agent.common import ContextMemory
from services.basic import PromptService


class ReasoningSkill(BaseSkill):
    """
    Logical Analysis Skill: Integrates data and knowledge, supports streaming distribution of thought process and final answer.
    """
    
    def __init__(self):
        super().__init__()
        self.model_client = AIModelClient()
        self.prompt_service = PromptService()

    async def run_stream(
        self, 
        context: ContextMemory,
        **kwargs 
    ) -> AsyncGenerator[dict[str, Any], None]:
        # 1. Safely extract context
        target_goal = getattr(context, 'current_task', None) or context.get("standalone_query") or context.get("question")
        current_model = context["llm_model"]

        sql_results = context.get("sql_results") or []
        doc_results = context.get("doc_results") or []
        graph_results = context.get("graph_results") or []
        combined_kb_results = doc_results + graph_results

        # ---- 上下文尺寸安全阀（仅兜底，正常由 search_top_k / rerank_top_k 控制） ----
        MAX_CONTENT_LEN = 2000       # 单个 chunk 内容最大字符数
        MAX_KB_TEXT_CHARS = 48000    # kb_text 总字符数硬上限（约 12000 tokens），仅在异常大量结果时触发

        kb_parts: list[str] = []
        kb_chars = 0
        for i, d in enumerate(combined_kb_results):
            raw = d.get("content", "") if isinstance(d, dict) else ""
            truncated = raw[:MAX_CONTENT_LEN]
            if len(raw) > MAX_CONTENT_LEN:
                truncated += "…"
            part = f"[{i+1}] {truncated}"
            if kb_chars + len(part) > MAX_KB_TEXT_CHARS:
                kb_parts.append(f"…（共 {len(combined_kb_results)} 条文档，受上下文窗口限制仅展示前 {i} 条）")
                break
            kb_parts.append(part)
            kb_chars += len(part)

        kb_text = "\n".join(kb_parts) if kb_parts else "No reference documents"
        # --------------------------------------------------
        
        # Optimize data context presentation
        if sql_results and isinstance(sql_results, list):
            try:
                import pandas as pd
                # Convert to Markdown table and limit to top 10 items to prevent token overflow
                data_text = pd.DataFrame(sql_results[:10]).to_markdown(index=False)
            except ImportError:
                data_text = str(sql_results[:10])
        else:
            data_text = "No business data"
        
        # summary = context["session_state"].get("context_summary", "New session")
        question = context["standalone_query"] or context["question"]

        reasoning_prompt = await default_prompt.generate(
            get_prompt_config().reasoning
        )

        # 3. Get user prompt
        user_prompt = await self.prompt_service.get_prompt_by_agent_id(context["agent_id"])
        final_prompt = f"""
【用户的分析需求】
{user_prompt}

【后台查询到的结构化数据】
{data_text if data_text else "（暂无数据）"}

【后台检索到的相关知识库文档】
{kb_text if kb_text else "（暂无相关文档）"}

【本次分析任务】
{target_goal}

请基于以上数据与知识，按照系统指令要求进行分析并回答用户的提问：{question}
"""

        # 4. State machine to parse LLM output
        is_thinking = False
        output_buffer = ""

        # 从 agent 配置获取 max_tokens，兜底 4096
        model_params = context.get("model_params", {})
        if isinstance(model_params, dict):
            llm_params = model_params.get("llm_params", {}) or {}
            max_tokens = llm_params.get("max_tokens") or 4096
        else:
            max_tokens = 4096
        logger.debug(f"[ReasoningSkill] max_tokens={max_tokens}")

        try:
            async for chunk in self.model_client.get_llm_stream_parsed(
                model_name=current_model,
                prompt=[
                        {"role": "system", "content": reasoning_prompt},
                        {"role": "user", "content": final_prompt}
                    ],
                temperature=0.3,
                max_tokens=max_tokens,
            ):
                if not chunk: continue

                # A. Native reasoning content support (e.g., DeepSeek-R1)
                if hasattr(chunk, "reasoning_content") and chunk.reasoning_content:
                    yield {"type": PacketType.THOUGHT, "content": chunk.reasoning_content}
                    continue

                if not chunk.content: continue
                
                # B. Tag parsing stream
                output_buffer += chunk.content

                while output_buffer:
                    if not is_thinking:
                        if "<thought>" in output_buffer:
                            parts = output_buffer.split("<thought>", 1)
                            if parts[0].strip(): 
                                yield {"type": PacketType.ANSWER, "content": parts[0]}
                            is_thinking = True
                            output_buffer = parts[1]
                        elif "<" in output_buffer and any(tag.startswith(output_buffer.rsplit("<", 1)[1]) for tag in ["thought>", "/thought>"]):
                            # Hit tag prefix, wait for subsequent fragments
                            break
                        else:
                            yield {"type": PacketType.ANSWER, "content": output_buffer}
                            output_buffer = ""
                    else:
                        if "</thought>" in output_buffer:
                            parts = output_buffer.split("</thought>", 1)
                            if parts[0].strip(): 
                                yield {"type": PacketType.THOUGHT, "content": parts[0]}
                            is_thinking = False
                            output_buffer = parts[1]
                        elif "<" in output_buffer and "/thought>".startswith(output_buffer.rsplit("<", 1)[1]):
                            break
                        else:
                            yield {"type": PacketType.THOUGHT, "content": output_buffer}
                            output_buffer = ""

            # Process remaining buffer
            if output_buffer.strip():
                m_type = PacketType.THOUGHT if is_thinking else PacketType.ANSWER
                yield {"type": m_type, "content": output_buffer}

        except Exception as e:
            logger.error(f"ReasoningSkill runtime exception: {e}")
            content = f"⚠️ Analysis interrupted: {str(e)}\n"
            yield {"type": PacketType.ERROR, "content": content}
