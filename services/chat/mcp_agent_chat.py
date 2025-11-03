import asyncio
from loguru import logger
from typing import Any
from agent_params import AgentParams, KBResult
from mcp_tool.mcp_tool import *



class Agent:
    """基于MCP的智能体类"""
    
    def __init__(self, agent_id: int, security: int, tags: list[str] | None = None):
        self.agent_id = agent_id
        self.security = security
        self.tags = tags
        self.agent_params = AgentParams()
        self.tool_registry = MCPToolRegistry()
        self.llm_client = None  # 初始化LLM客户端
        
    def register_tools(self, tools: list[MCPTool]):
        """注册工具"""
        for tool in tools:
            self.tool_registry.register(tool)
    
    async def _call_llm_for_tool_selection(self, question: str, context: dict[str, Any]) -> list[ToolCall]:
        """
        调用大模型选择要使用的工具
        
        Args:
            question: 用户问题
            context: 上下文信息
            
        Returns:
            list[ToolCall]: 工具调用列表
        """
        try:
            # 获取所有可用工具的schema
            tools_schema = self.tool_registry.get_tools_schema()
            
            # 构建提示词
            prompt = self._build_tool_selection_prompt(question, context, tools_schema)
            
            # 调用LLM进行工具选择
            llm_response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                tools=self._format_tools_for_llm(tools_schema),
                tool_choice="auto"
            )
            
            # 解析LLM返回的工具调用
            tool_calls = []
            if llm_response.tool_calls:
                for tool_call in llm_response.tool_calls:
                    tool_calls.append(ToolCall(
                        tool_type=ToolType(tool_call.function.name),
                        tool_name=tool_call.function.name,
                        parameters=tool_call.function.arguments,
                        description=tools_schema.get(tool_call.function.name, {}).get("description", "")
                    ))
            
            return tool_calls
            
        except Exception as e:
            logger.error(f"LLM工具选择失败: {e}")
            return []
    
    def _build_tool_selection_prompt(self, question: str, context: dict[str, Any], tools_schema: dict) -> str:
        """构建工具选择提示词"""
        prompt = f"""你是一个智能助手，需要根据用户问题选择合适的工具来解决问题。

        用户问题: {question}

        可用工具:
        """
        for tool_name, tool_info in tools_schema.items():
            prompt += f"- {tool_name}: {tool_info['description']}\n"
        
        prompt += """
        请根据用户问题选择最合适的工具，并给出调用参数。优先选择知识库搜索工具来获取准确信息。
        """
        return prompt
    
    def _format_tools_for_llm(self, tools_schema: dict) -> list[dict]:
        """格式化工具schema供LLM使用"""
        tools = []
        for tool_name, tool_info in tools_schema.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info["description"],
                    "parameters": tool_info["schema"]
                }
            })
        return tools
    
    async def _execute_tools_parallel(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """并行执行工具调用"""
        if not tool_calls:
            return []
        
        tasks = []
        for tool_call in tool_calls:
            tool = self.tool_registry.get_tool(tool_call.tool_name)
            if tool:
                tasks.append(tool.execute(tool_call.parameters))
            else:
                logger.warning(f"工具未找到: {tool_call.tool_name}")
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return self._process_tool_results(results)
        except Exception as e:
            logger.error(f"工具并行执行失败: {e}")
            return []
    
    def _process_tool_results(self, results: list[Any]) -> list[ToolResult]:
        """处理工具执行结果"""
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"工具执行失败: {result}")
            elif isinstance(result, ToolResult):
                processed_results.append(result)
            else:
                logger.warning(f"工具返回未知类型结果: {type(result)}")
        return processed_results
    
    async def _combine_and_rerank_results(self, question: str, tool_results: list[ToolResult]) -> list[KBResult]:
        """合并和重排结果"""
        all_kb_results = []
        
        # 提取所有知识库结果
        for tool_result in tool_results:
            if (tool_result.tool_type == ToolType.KB_SEARCH and 
                isinstance(tool_result.content, list)):
                all_kb_results.extend(tool_result.content)
        
        # 应用重排逻辑
        if len(all_kb_results) > 1 and self.agent_params.reranker_model_id:
            reranker = AgentRerank(self.agent_params)
            reranked = await reranker.rerank_kb(question, all_kb_results)
            if reranked:
                all_kb_results = [
                    item for item in reranked 
                    if item.reranker_score >= self.agent_params.reranker_score_threshold
                ]
        
        # 去重和排序
        seen = set()
        unique_results = []
        for item in all_kb_results:
            content = item.content.strip()
            if content not in seen:
                seen.add(content)
                unique_results.append(item)
        
        unique_results.sort(key=lambda x: x.weight, reverse=True)
        return unique_results