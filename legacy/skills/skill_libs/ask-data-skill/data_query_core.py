import uuid
import aiohttp
from loguru import logger
from typing import Any, AsyncGenerator
from skills import *
from platform_core.dictionary import PacketType
from services.basic.agent_service import AgentService
from agent.common import ContextMemory
from platform_core.config.settings import get_ask_data_api_config



class AskDataSkill(BaseSkill):
    """
    问数技能：基于 SQLAgent 的结构化数据查询（完全对齐小写连字符规范）。
    """
    meta = SkillMeta(
        name="ask-data-skill",
        description="根据自然语言高命中率地查询目标数据库的数据",
        domain=SkillDomain.BUSINESS,
        run_mode=SkillRunMode.READ_ONLY
    )

    def __init__(self):
        super().__init__()

    async def run_stream(
        self,
        context: ContextMemory,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        
        current_execution = context["current_execution"]
        if not current_execution:
            yield {"type": PacketType.ERROR, "content": "问数组件的执行上下文快照丢失, 无法继续执行。\n"}
            return

        # 动态获取在大模型和 manager 中注册的严格小写名称（即 ask-data-skill）
        runtime_skill_name = current_execution.get("skill", "ask-data-skill")
        current_agent = context["agent_id"]
        current_user = context.get("user_id", "default_user")
        
        query_text = current_execution["resolved_input"]

        if not query_text:
            yield {"type": PacketType.ERROR, "content": "变量解析异常，未能获取有效的查询文本，无法继续执行。\n"}
            return
        
        if not current_agent:
            yield {"type": PacketType.ERROR, "content": "全局上下文缺失关键参数 agent_id。\n"}
            return

        logger.info(f"自治组件 [{runtime_skill_name}] 触发链条执行 | Agent: {current_agent} | Query: {query_text[:60]}...")

        yield {"type": PacketType.THOUGHT, "content": "正在为您连接数据库查询管理器，深度分析查询请求：'" + query_text + "'...\n"}
        
        final_sql_results = {
            "data": [],
            "is_empty": True,
            "error": None
        }

        try:
            # 从 Agent 获取 profile ID
            agent_service = AgentService()
            profile = await agent_service.get_agent_profile(current_agent)

            # 加载外部问数 API 配置
            api_config = get_ask_data_api_config()
            api_endpoint = api_config.api_endpoint
            api_key = api_config.api_key
            timeout = aiohttp.ClientTimeout(total=api_config.timeout)

            # 构造请求体
            payload = {
                "profile": profile,
                "user": current_user,
                "ask": query_text,
            }
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            logger.info(
                f"[{runtime_skill_name}] 调用外部问数接口 | endpoint={api_endpoint} | "
                f"profile={profile} | ask={query_text[:60]}..."
            )

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(api_endpoint, json=payload, headers=headers) as response:
                    if response.status != 200:
                        err_text = await response.text()
                        logger.error(f"[{runtime_skill_name}] 外部问数接口响应异常 {response.status}: {err_text}")
                        final_sql_results["error"] = f"问数接口返回异常状态 {response.status}"
                        yield {"type": PacketType.SQL_RESULTS, "content": final_sql_results}
                        yield {"type": PacketType.ERROR, "content": f"问数接口调用失败（HTTP {response.status}），请稍后重试。\n"}
                        return

                    res_json = await response.json()

            # 从 data[0].data 提取数据行
            response_data = res_json.get("data", [])
            raw_rows = response_data[0].get("data", []) if response_data else []

            logger.info(f"[{runtime_skill_name}] 外部问数接口返回 {len(raw_rows)} 行数据")

            # 填充 final_sql_results
            final_sql_results["data"] = raw_rows
            final_sql_results["is_empty"] = len(raw_rows) == 0

            # 1. 提取当前步骤定义的 output_var
            output_key = current_execution.get("output_var") or "inspection_records"
            clean_output_key = str(output_key).strip().strip("'").strip('"')

            # 2. 写入变量池（供变量替换使用）
            if "variables" not in context:
                context["variables"] = {}
            context["variables"][clean_output_key] = raw_rows

            # 3. 组装结构化结果并通过 SQL_RESULTS 包发送
            #    root_orchestrator 会从中提取 data 行 extend 至 context["sql_results"]
            formatted_res = {
                "sql": payload.get("ask"),
                "data": raw_rows,
            }
            yield {"type": PacketType.SQL_RESULTS, "content": formatted_res}
            return

        except aiohttp.ClientConnectorError:
            logger.error(f"[{runtime_skill_name}] 无法连接外部问数接口: {api_endpoint}")
            final_sql_results["error"] = "无法连接至外部问数接口"
            yield {"type": PacketType.SQL_RESULTS, "content": final_sql_results}
            yield {"type": PacketType.ERROR, "content": "网络故障：无法连接至外部问数接口，请检查网络或稍后重试。\n"}
        except Exception as e:
            logger.error(f"[{runtime_skill_name}] 运行时遭遇致命崩溃: {e}", exc_info=True)
            error_info = str(e)
            final_sql_results.update({"error": error_info})
            yield {"type": PacketType.SQL_RESULTS, "content": final_sql_results}
            yield {"type": PacketType.ERROR, "content": f"问数组件执行时发生系统异常：{error_info}。\n"}
