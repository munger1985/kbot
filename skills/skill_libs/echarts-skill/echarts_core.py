from typing import Any, AsyncGenerator
from loguru import logger
from skills import BaseSkill
from utils.clients import AIModelClient
from agent.prompt import default_prompt
from core.config import get_prompt_config
from agent.common import ContextMemory
from core.dictionary import PacketType


class EChartsSkill(BaseSkill):
    """
    可视化技能：根据数据分析结果，自动推荐并生成 ECharts 可视化配置。
    """
    
    def __init__(self):
        super().__init__()
        self.model_client = AIModelClient()

    async def run_stream(
        self, 
        context: ContextMemory,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        :param task_input: Planner 传来的绘图指令，例如 "将过去一周的产量对比绘制为折线图"
        :param context: 包含之前步骤获取的数据（如 sql_results）
        """
        
        
        # 1. 从上下文中获取实际要绘图的数据
        # 从 context 中提取参数
        task_input = context["current_execution"] or context["standalone_query"]
        model_name = context["llm_model"]

        logger.info(f"ImageSkill: 准备为数据生成可视化方案 -> {task_input}")
        yield {"type": PacketType.THOUGHT, "content": f"开始为数据生成可视化方案: `{task_input}`\n"}

        # 从 context 中获取数据
        raw_data = context["sql_results"] or task_input

        prompt = await default_prompt.generate(
            get_prompt_config().generate_chart,
            data_content=str(raw_data),
            user_requirement=task_input
        )

        try:
            # 3. 获取结构化 JSON 配置
            chart_config = await self.model_client.get_llm_json(
                model_name=model_name, 
                prompt=prompt,
                temperature=0.0  # 绘图配置需要高度严谨，不建议随机
            )
            
            # 4. 结果清洗与校验
            if "option" not in chart_config:
                raise ValueError("LLM 返回的 JSON 缺少 option 字段")
                
            logger.info(f"ImageSkill: 成功生成 {chart_config.get('chart_type')} 类型图表")
            yield {
                "type": PacketType.ECHARTS,
                "content": chart_config
            }

        except Exception as e:
            logger.error(f"ImageSkill 执行异常: {e}")
            yield {"type": PacketType.ERROR, "content": f"可视化生成失败: {str(e)}"}