import pandas as pd
from loguru import logger
from typing import Any, AsyncGenerator
from utils.clients import AIModelClient
from agent.prompt import default_prompt
from core.config import get_prompt_config
from core.dictionary import PacketType
from skills import BaseSkill, SkillMeta, SkillDomain, SkillRunMode
from agent.common import ContextMemory
from services.basic import PromptService


class ReasoningSkill(BaseSkill):
    """
    逻辑分析技能：整合数据与知识，支持思考过程(thought)与答案(answer)的流式分发。
    """
    meta = SkillMeta(
        name="reasoning-skill",
        description="整合数据与知识，支持思考过程(thought)与答案(answer)的流式分发",
        domain=SkillDomain.BUSINESS,
        run_mode=SkillRunMode.READ_ONLY
    )
    
    def __init__(self):
        super().__init__()
        self.model_client = AIModelClient()
        self.prompt_service = PromptService()

    async def run_stream(
        self, 
        context: ContextMemory,  # 统一使用 ContextMemory
        **kwargs 
    ) -> AsyncGenerator[dict[str, Any], None]:
        # 1. 安全提取上下文
        current_model = context["llm_model"]
        question = context["standalone_query"] or context["question"]
        # 从 current_execution 中取 planner 下发的 task_description 作为分析任务
        current_exec = context.get("current_execution") or {}
        task_desc = (current_exec.get("task_description") or current_exec.get("resolved_input") or question) if isinstance(current_exec, dict) else question
        sql_results = context.get("sql_results") or []
        doc_results = context.get("doc_results") or []
        graph_results = context.get("graph_results") or []
        combined_kb_results = doc_results + graph_results

        # 2. 构建 LLM 上下文文本：只提取 content 字段，忽略元数据（score/kb_id/bbox等）
        kb_parts: list[str] = []
        for i, d in enumerate(combined_kb_results):
            # 兼容 Pydantic 模型和普通 dict
            if hasattr(d, 'content'):
                text = d.content # type: ignore
            elif isinstance(d, dict):
                text = d.get('content', '')
            else:
                text = str(d)
            if text and text.strip():
                kb_parts.append(f"[{i+1}] {text}")
        kb_text = "\n".join(kb_parts) if kb_parts else "无参考文档"
        kb_count = len(kb_parts)

        # 优化数据上下文展示
        # 提取实际的数据集：sql_results 结构为 [{"sql": "...", "data": [dict1, dict2, ...]}]
        sql_data: list[dict[str, Any]] = []
        if isinstance(sql_results, list) and len(sql_results) > 0:
            # 优先取最后一条结果的 data 字段（最新的查询结果）
            last_result = sql_results[-1]
            if isinstance(last_result, dict) and "data" in last_result:
                raw_data = last_result["data"]
                if isinstance(raw_data, list):
                    sql_data = raw_data
            elif isinstance(last_result, list):
                # 兼容旧格式：sql_results 直接是数据列表
                sql_data = last_result

        if sql_data:
            try:
                data_text = pd.DataFrame(sql_data[:10]).to_markdown(index=False) # type: ignore
            except (ImportError, Exception) as e:
                logger.warning(f"SQL 数据转换 Markdown 表格失败: {e}")
                data_text = str(sql_data[:10]) # type: ignore
        else:
            data_text = "无业务数据"

        reasoning_prompt = await default_prompt.generate(
            get_prompt_config().reasoning,
            user_language=context.get("user_language", "English")
        )

        # 3. 获取用户提示词
        user_prompt = await self.prompt_service.get_prompt_by_agent_id(context["agent_id"])
        original_question = context.get("question", "")
        final_prompt = f"""
【系统指令】
{user_prompt}

【用户原始问题】
{original_question}

【后台查询到的结构化数据】
{data_text if data_text else "（暂无数据）"}

【后台检索到的相关知识库文档】
{kb_text if kb_text else "（暂无相关文档）"}

【本次分析任务】
{task_desc}

请基于以上数据与知识，按照系统指令要求进行分析并回答。
"""
        
        # 4. 流式输出前，先发送上下文摘要（不暴露原始文档内容）
        sql_rows = len(sql_data)
        summary_parts: list[str] = []
        if sql_rows > 0:
            summary_parts.append(f"已获取 {sql_rows} 条结构化查询结果")
        if kb_count > 0:
            summary_parts.append(f"检索到 {kb_count} 个相关知识文档")
        if graph_results:
            g_count = len(graph_results) if isinstance(graph_results, list) else 1
            summary_parts.append(f"匹配 {g_count} 条知识图谱数据")
        if summary_parts:
            yield {
                "type": PacketType.THOUGHT,
                "content": "正在综合分析：" + "，".join(summary_parts) + "。\n"
            }

        # 5. 状态机解析 LLM 输出
        is_thinking = False
        output_buffer = ""

        try:
            async for chunk in self.model_client.get_llm_stream_parsed(
                model_name=current_model,
                prompt=[
                        {"role": "system", "content": reasoning_prompt},
                        {"role": "user", "content": final_prompt}
                    ],
                temperature=0.3
            ):
                if not chunk: continue

                # A. 原生推理字段支持 (DeepSeek-R1 等)
                if hasattr(chunk, "reasoning_content") and chunk.reasoning_content:
                    yield {"type": PacketType.THOUGHT, "content": chunk.reasoning_content}
                    continue

                if not chunk.content: continue
                
                # B. 标签解析流
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
                            # 命中标签前缀，等待后续片段
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

            # 处理末尾残留
            if output_buffer.strip():
                m_type = PacketType.THOUGHT if is_thinking else PacketType.ANSWER
                yield {"type": m_type, "content": output_buffer}

        except Exception as e:
            logger.error(f"ReasoningSkill 运行异常: {e}")
            yield {"type": PacketType.ERROR, "content": f"⚠️ 分析中断: {str(e)}\n"}