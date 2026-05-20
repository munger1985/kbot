from loguru import logger
from typing import Any, AsyncGenerator
from utils.clients import AIModelClient
from agent.prompt import default_prompt
from core.config.settings import get_prompt_config
from core.dictionary import PacketType
from skills import BaseSkill
from agent.common import ContextMemory
from services.basic import PromptService


class ReasoningSkill(BaseSkill):
    """
    逻辑分析技能：整合数据与知识，支持思考过程(thought)与答案(answer)的流式分发。
    """
    
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
        target_goal = getattr(context, 'current_task', None) or context.get("standalone_query") or context.get("question")
        current_model = context["llm_model"]
        
        # --- 重点修改开始：增强数据提取逻辑 ---
        # A. 尝试从 Planner 定义的变量池中提取 (对应 AskDataSkill 写入的 output_var)
        vars_pool = context.get("variables", {})
        # 优先取 Planner 动态分配的键，或常见的默认键
        sql_data = vars_pool.get("inspection_records") or vars_pool.get("query_result")

        # B. 备选方案：从 context["sql_results"] 提取（兼容传统模式）
        if not sql_data:
            res_val = context.get("sql_results")
            if isinstance(res_val, dict):
                sql_data = res_val.get("data")
            elif isinstance(res_val, list) and len(res_val) > 0:
                # 如果是列表，取最后一项（最新的查询结果）
                last_item = res_val[-1]
                sql_data = last_item.get("data") if isinstance(last_item, dict) else last_item

        doc_results = context.get("doc_results") or []
        # --- 重点修改结束 ---

        # 2. 构建 LLM 上下文文本 (优化点：将 List 转为 Markdown 表格，LLM 更易理解)
        kb_text = "\n".join([f"[{i+1}] {d.get('content')}" for i, d in enumerate(doc_results)]) if doc_results else "无参考文档"
        
        # 优化数据上下文展示
        if sql_data and isinstance(sql_data, list):
            try:
                import pandas as pd
                # 转换为 Markdown 表格，并限制前 10 条，防止 Token 溢出
                data_text = pd.DataFrame(sql_data[:10]).to_markdown(index=False)
            except ImportError:
                data_text = str(sql_data[:10])
        else:
            data_text = "无业务数据"

        # 2. 构建 LLM 上下文文本 (优化点：避免直接 str(list))
        kb_text = "\n".join([f"[{i+1}] {d.get('content')}" for i, d in enumerate(doc_results)]) if doc_results else "无参考文档"
        # 数据表格建议转为 Markdown 表格或简易文本
        data_text = str(sql_data) if sql_data else "无业务数据"
        
        summary = context["session_state"].get("context_summary", "新会话")

        reasoning_prompt = await default_prompt.generate(
            get_prompt_config().reasoning,
            data_context=data_text,
            kb_context=kb_text,
            context_summary=summary,
            final_goal=target_goal
        )

        # 3. 获取用户提示
        user_prompt = await self.prompt_service.get_prompt_by_agent_id(context["agent_id"])
        final_prompt = f"{reasoning_prompt}\n{user_prompt}"

        # 4. 状态机解析 LLM 输出
        is_thinking = False
        output_buffer = ""

        try:
            async for chunk in self.model_client.get_llm_stream_parsed(
                model_name=current_model,
                prompt=final_prompt,
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
            logger.error(f"AnalysisSkill 异常: {e}")
            yield {"type": PacketType.ERROR, "content": f"分析中断: {str(e)}"}