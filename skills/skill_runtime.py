import re
import json
import copy
from datetime import datetime, timezone
from typing import Any, AsyncGenerator
from loguru import logger

from core.dictionary import PacketType
from agent.common import ContextMemory, SkillExecutionContext, TaskStep


class SkillRuntime:
    """
    Skill 运行时处理器
    负责处理强类型多维变量注入、状态追踪、执行结果回填以及数据总线交互。
    """

    def __init__(self, context: ContextMemory):
        self.ctx = context
        # 变量占位符正则表达式，匹配 {{variable_name}}
        self.var_pattern = re.compile(r"\{\{\s*(\w+)\s*\}\}")

    def _resolve_variables(self, text: str) -> str:
        """
        解析并替换文本中的变量占位符。
        强类型判定：若变量为 list/dict（如 SQL 结果集），自动转换为 JSON 字符串形式。
        """
        if not text or not isinstance(text, str):
            return text

        def replace_match(match):
            var_name = match.group(1)
            # 优先从 ctx["variables"] 取值，找不到则保持原样占位
            value = self.ctx["variables"].get(var_name)
            if value is None:
                return match.group(0)
            
            # 如果是列表、字典等复杂数据结构，转为标准符合大模型直觉的 JSON 字符串
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            
            return str(value)

        return self.var_pattern.sub(replace_match, text)

    def _resolve_any(self, target: Any) -> Any:
        """
        【工业级新增】：递归清洗器。
        支持对任意嵌套的字符串、列表、字典进行全局占位符替换，完美支持多参数技能（如 compute）。
        """
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
        
        # 🟢 升级：深拷贝一份原始参数，排除掉路由控制元数据，剩下全当做业务参数清洗
        raw_params = copy.deepcopy(step_config)
        raw_params.pop("skill", None)
        raw_params.pop("step_id", None)
        raw_params.pop("output_var", None)
        raw_params.pop("condition", None)

        # 🟢 升级：利用递归清洗器，把诸如 formula, variables, task_input 里所有的 {{占位符}} 批量一网打尽
        resolved_params = self._resolve_any(raw_params)

        # 兼容老框架的 resolved_input 字段，若有 task_description 则用它，否则用配置全貌
        resolved_input_legacy = resolved_params.get("task_description") or resolved_params.get("task_input") or json.dumps(resolved_params, ensure_ascii=False)

        # 2. 构建执行快照
        execution: SkillExecutionContext = {
            "skill": skill_name,
            "task_description": step_config.get("task_description", ""),
            "resolved_input": resolved_input_legacy,  # 留给传统展现层/日志使用
            "resolved_params": resolved_params,      # 🟢 核心：塞给具体武器（Skill）开箱即用的多维纯净参数字典
            "start_time": datetime.now(timezone.utc),
            "end_time": None,
            "status": "running",
            "output": None,
            "output_var": step_config.get("output_var"),
            "error": None
        }
        
        # 3. 更新 Context 中的当前执行槽位
        self.ctx["current_execution"] = execution
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

        # 引入局部文本累加器，专门用来对付流式退化备选输出
        answer_text_accumulator = ""

        try:
            # 调用业务 Skill 的流式入口
            async for packet in skill_instance.run_stream(context=self.ctx):
                p_type = packet.get("type")
                content = packet.get("content")

                # 1. 总线数据自动监听（历史追溯）
                if p_type == PacketType.DOC_RESULTS:
                    self.ctx["doc_results"].extend(content if isinstance(content, list) else [])
                elif p_type == PacketType.SQL_RESULTS:
                    self.ctx["sql_results"].append(content)
                
                # 2. 收集最终产出
                if p_type == PacketType.DONE:
                    # 优先：业务 Skill 显式宣告执行完毕吐出的干净结构体
                    execution["output"] = content
                elif p_type == PacketType.ANSWER:
                    answer_text_accumulator += str(content or "")

                yield packet

            # 3. 判定最终落袋的 Output
            if execution["output"] is None and answer_text_accumulator:
                # 降级：如果业务没有显式给 DONE 包带料，则使用累加完的流文本作为产出
                execution["output"] = answer_text_accumulator.strip()

            execution["status"] = "success"

        except Exception as e:
            logger.exception(f"Skill {skill_name} 运行中崩溃: {e}")
            execution["status"] = "failed"
            execution["error"] = str(e)
            yield {"type": PacketType.ERROR, "content": f"组件 {skill_name} 执行异常: {str(e)}"}

        finally:
            # 4. 执行收尾记时
            execution["end_time"] = datetime.now(timezone.utc)
            
            # 5. 强类型变量回填逻辑 (送回全局总线变量池，供接下来的步骤消费)
            if execution["output_var"] and execution["status"] == "success":
                self.ctx["variables"][execution["output_var"]] = execution["output"]
            
            # 6. 归档到历史记录快照，清空当前活动槽位
            self.ctx["execution_history"].append(execution)
            self.ctx["current_execution"] = None