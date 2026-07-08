import re
import json
import copy
from datetime import datetime, timezone
from typing import Any, AsyncGenerator
from loguru import logger

from core.dictionary import PacketType
# 引入两种 Context 的定义
from agent.common import ContextMemory, SkillExecutionContext, TaskStep, OpsContextMemory


class SkillRuntime:
    """
    Skill 运行时处理器 (4.0 双模多态版)
    负责处理强类型多维变量注入、状态追踪、执行结果回填以及数据总线交互。
    支持统一调度 Business 级通用业务 Context 与 AIOps 级强类型特化 OpsContext。
    """

    # LLM 常用变量名 → ContextMemory 实际 key 的别名映射
    _ALIAS_MAP: dict[str, str] = {
        "user_query": "standalone_query",
        "query": "standalone_query",
        "original_question": "question",
        "user_question": "standalone_query",
    }

    def __init__(self, context: ContextMemory | OpsContextMemory):
        # 在运行时，通过强制转型或通用字典兼容来保障统一处理流程
        self.ctx = context
        # 变量占位符正则：{{variable_name}} (双大括号)
        self.var_pattern = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")
        # 单大括号模式：{variable_name} (LLM 常见输出格式，与 JSON 语法区分)
        self._single_brace_pattern = re.compile(r"(?<!\")\{\s*([\w.]+)\s*\}(?!\")")

    def _resolve_variables(self, text: str) -> str:
        """
        解析并替换文本中的变量占位符。
        支持 {{variable_name}} 和 {variable_name} 两种语法。
        支持 'variables.prod_speed_data'、'global_inputs.line_id' 等多层级路径提取。
        强类型判定：若变量为 list/dict，自动转换为 JSON 字符串形式。
        """
        if not text or not isinstance(text, str):
            return text

        def replace_match(match):
            full_path = match.group(1).strip()
            path_parts = full_path.split('.')

            # 动态多层级路径提取导航
            current_value = self.ctx

            try:
                for part in path_parts:
                    if isinstance(current_value, dict):
                        current_value = current_value.get(part)
                    else:
                        current_value = getattr(current_value, part, None)

                if current_value is None:
                    # 💡 多模适配兜底：不管是 business 还是 ops，都去 variables 字典里找全路径
                    variables_pool = self.ctx.get("variables")
                    if isinstance(variables_pool, dict):
                        current_value = variables_pool.get(full_path)

                # 🔧 别名映射：LLM 常用的变量名 → ContextMemory 实际 key
                if current_value is None:
                    alias_key = self._ALIAS_MAP.get(full_path)
                    if alias_key:
                        current_value = self.ctx.get(alias_key)

                if current_value is None:
                    return match.group(0)  # 维持原样占位符

                if isinstance(current_value, (dict, list)):
                    return json.dumps(current_value, ensure_ascii=False)

                return str(current_value)

            except Exception as e:
                logger.warning(f"解析变量路径 {full_path} 时发生异常: {e}")
                return match.group(0)

        # 先尝试双大括号 {{var}}，再尝试单大括号 {var}
        text = self.var_pattern.sub(replace_match, text)
        text = self._single_brace_pattern.sub(replace_match, text)
        return text

    def _resolve_any(self, target: Any) -> Any:
        """递归清洗器：清洗任意嵌套结构中的 {{占位符}}"""
        if isinstance(target, str):
            return self._resolve_variables(target)
        elif isinstance(target, dict):
            return {k: self._resolve_any(v) for k, v in target.items()}
        elif isinstance(target, list):
            return [self._resolve_any(item) for item in target]
        return target

    def create_execution_context(self, step_config: TaskStep) -> SkillExecutionContext:
        """
        根据规划步骤初始化执行快照。
        """
        skill_name = step_config.get("skill", "UnknownSkill")
        
        # 深拷贝一份原始参数，排除掉路由控制元数据，剩下全当做业务参数清洗
        raw_params = copy.deepcopy(step_config)
        raw_params.pop("skill", None)
        raw_params.pop("step_id", None)
        raw_params.pop("output_var", None)
        raw_params.pop("condition", None)

        resolved_params = self._resolve_any(raw_params)

        # 兼容老框架的 resolved_input 字段
        resolved_input_legacy = (
            resolved_params.get("task_description") or 
            resolved_params.get("task_input") or 
            json.dumps(resolved_params, ensure_ascii=False)
        )

        # 构建执行快照
        execution: SkillExecutionContext = {
            "skill": skill_name,
            "task_description": step_config.get("task_description", ""),
            "resolved_input": str(resolved_input_legacy),
            "resolved_params": resolved_params,      
            "start_time": datetime.now(timezone.utc),
            "end_time": None,
            "status": "running",
            "output": None,
            "output_var": step_config.get("output_var"),
            "error": None
        }
        
        # 更新 Context 中的当前执行槽位（双模均支持此 key 设为可选或 None）
        self.ctx["current_execution"] = execution  # type: ignore
        return execution

    async def execute_skill(
        self, 
        skill_instance: Any, 
        execution: SkillExecutionContext
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        执行具体的 Skill 实例并捕获流式输出。
        """
        skill_name = execution["skill"]
        logger.info(f"Runtime 开始执行 Skill: {skill_name} | Input: {execution['resolved_input'][:100]}...")

        answer_text_accumulator = ""

        try:
            # 统一流式入口调度
            async for packet in skill_instance.run_stream(context=self.ctx):
                p_type = packet.get("type")
                content = packet.get("content")

                # 💡 核心改造：针对不同上下文的总线异构数据监听与存储隔离防御
                if p_type == PacketType.DOC_RESULTS:
                    if "doc_results" in self.ctx and self.ctx["doc_results"] is not None:
                        # 确保以 list 方式安全合并
                        target_list = content if isinstance(content, list) else [content]
                        self.ctx["doc_results"].extend(target_list) # type: ignore
                        
                # SQL_RESULTS 已统一在 root_orchestrator 层处理（避免与 skill_runtime 重复回填）
                
                # 收集最终产出
                if p_type == PacketType.DONE:
                    execution["output"] = content
                elif p_type == PacketType.ANSWER:
                    answer_text_accumulator += str(content or "")

                yield packet

            # 始终保留 answer 文本，供下游技能提取 SQL 等结构化产物
            if answer_text_accumulator:
                execution["answer"] = answer_text_accumulator.strip()
            if execution["output"] is None and answer_text_accumulator:
                execution["output"] = answer_text_accumulator.strip()

            execution["status"] = "success"

        except Exception as e:
            logger.exception(f"Skill {skill_name} 运行中崩溃: {e}")
            execution["status"] = "failed"
            execution["error"] = str(e)
            yield {"type": PacketType.ERROR, "content": f"组件 {skill_name} 执行异常: {str(e)}"}

        finally:
            execution["end_time"] = datetime.now(timezone.utc)
            
            # 变量回填逻辑
            if execution["output_var"] and execution["status"] == "success":
                # 双模上下文里均有一级成员 variables 字典
                if "variables" in self.ctx and isinstance(self.ctx["variables"], dict):
                    self.ctx["variables"][execution["output_var"]] = execution["output"]
            
            # 归档到历史记录快照
            if "execution_history" in self.ctx and isinstance(self.ctx["execution_history"], list):
                self.ctx["execution_history"].append(execution) # type: ignore
                
            self.ctx["current_execution"] = None # type: ignore