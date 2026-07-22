from typing import Any, AsyncGenerator
from loguru import logger
from skills import BaseSkill
from platform_clients import AIModelClient
from agent.prompt import default_prompt
from platform_core.config import get_prompt_config
from agent.common import ContextMemory
from platform_core.dictionary import PacketType


class EChartsSkill(BaseSkill):
    """
    Visualization Skill: Automatically recommend and generate ECharts visualization configuration based on data analysis results.
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
        :param task_input: Drawing instruction from Planner, e.g., "Draw a line chart comparing production in the past week"
        :param context: Contains data obtained from previous steps (e.g. sql_results)
        """
        
        
        # 1. Get the actual data to be plotted from context
        # Extract parameters from context
        current_exec = context["current_execution"]
        if isinstance(current_exec, dict):
            task_input = current_exec.get("resolved_input") or current_exec.get("task_description") or ""
        else:
            task_input = current_exec or ""
        task_input = task_input or context["standalone_query"] or ""
        model_name = context["llm_model"]

        logger.info(f"ImageSkill: Preparing to generate visualization方案 for data -> {task_input}")
        content = f"Start generating visualization solution for data: `{task_input}`\n"
        yield {"type": PacketType.THOUGHT, "content": content}

        # Get data from context
        raw_data = context["sql_results"] or task_input

        prompt = await default_prompt.generate(
            get_prompt_config().generate_chart,
            data_content=str(raw_data),
            user_requirement=task_input
        )

        try:
            # 3. Get structured JSON configuration
            chart_config = await self.model_client.get_llm_json(
                model_name=model_name, 
                prompt=prompt,
                temperature=0.0  # Highly rigorous required for drawing configuration, randomness not recommended
            )
            
            # 4. Result cleaning and verification
            if "option" not in chart_config:
                raise ValueError("Missing option field in JSON returned by LLM")
                
            logger.info(f"ImageSkill: Successfully generated {chart_config.get('chart_type')} type chart")
            yield {
                "type": PacketType.ECHARTS,
                "content": chart_config
            }

        except Exception as e:
            logger.error(f"ImageSkill execution exception: {e}")
            content = f"⚠️ Visualization generation failed: {str(e)}\n"
            yield {"type": PacketType.ERROR, "content": content}
