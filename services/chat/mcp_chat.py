import asyncio
from loguru import logger
from typing import Any
from .agent_params import AgentParams, KBResult, ToolParams
from .agent_rerank import AgentRerank
from ..search.chinese_preprocessor import preprocess_cn_query
from mcp_tools import *
from core.dictionary import MCPToolType, ToolType
from dao.repositories.kbot_md_agent_conf_repo import KbotMdAgentConfRepository
from dao.repositories.kbot_md_agent_repo import KbotMdAgentRepository
from utils.call_models import CallModel



class Agent:
    """基于MCP的智能体类"""
    
    def __init__(self, agent_id: int, security: int, tags: list[str] = []):
        """
        初始化智能体
        
        Args:
            agent_id: 智能体ID
            security: 安全级别
            tags: 标签列表. 默认为空列表
        """
        self.agent_id = agent_id
        self.security = security
        self.tags = tags
        self.agent_params = AgentParams()
        self.tool_registry = MCPToolRegistry()
    
    def register_tools(self, tools: list[MCPTool]):
        """注册工具"""
        for tool in tools:
            # 设置工具的安全级别和标签
            tool.security = self.security
            tool.tags = self.tags
            self.tool_registry.register(tool)
        logger.info(f"成功注册 {len(tools)} 个工具")
    
    async def _call_llm_for_tool_selection(self, question: str, context: dict[str, Any]) -> list[ToolCall]:
        """
        调用大模型选择要使用的工具
        
        Args:
            question: 用户问题
            context: 上下文信息
            
        Returns:
            List[ToolCall]: 工具调用列表
        """
        try:
            # 获取所有可用工具的schema
            tools_schema = self.tool_registry.get_tools_schema()
            
            if not tools_schema:
                logger.warning("没有可用的工具")
                return []
            
            # 构建提示词
            prompt = self._build_tool_selection_prompt(question, context, tools_schema)
            
            # 格式化工具供LLM使用
            tools = self._format_tools_for_llm(tools_schema)
            
            # 调用真实的LLM服务进行工具选择
            llm_response = await self._call_llm_for_tool_selection_internal(
                prompt=prompt,
                tools=tools
            )
            
            # 解析LLM返回的工具调用
            tool_calls = []
            if llm_response and hasattr(llm_response, 'choices') and llm_response.choices:
                message = llm_response.choices[0].message
                
                # 检查是否有工具调用
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_info = tools_schema.get(tool_name, {})
                        
                        # 解析参数
                        import json
                        try:
                            parameters = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            logger.warning(f"工具参数解析失败: {tool_call.function.arguments}")
                            parameters = {}
                        
                        # 确定工具类型
                        tool_type_str = tool_info.get('type', 'kb_search')
                        try:
                            tool_type = MCPToolType(tool_type_str)
                        except ValueError:
                            tool_type = MCPToolType.KB_SEARCH
                        
                        tool_calls.append(ToolCall(
                            tool_type=tool_type,
                            tool_name=tool_name,
                            parameters=parameters,
                            description=tool_info.get('description', '')
                        ))
                        
                        logger.debug(f"LLM选择工具: {tool_name} with params: {parameters}")
            
            return tool_calls
            
        except Exception as e:
            logger.error(f"LLM工具选择失败: {e}")
            return []

    async def _call_llm_for_tool_selection_internal(self, prompt: str, tools: list[dict[str, Any]]) -> Any:
        """
        内部方法：调用LLM进行工具选择
        
        Args:
            prompt: 提示词
            tools: 工具列表
            
        Returns:
            LLM响应对象
        """
        try:
            # 使用非流式模式调用LLM
            model_id = self.agent_params.llm_id  # 使用智能体配置的LLM ID
            if model_id is None:
                raise ValueError("智能体配置中未指定 LLM")
            
            # 调用LLM服务
            async for chunk in CallModel().call_llm_model(
                model_id=model_id,
                prompt=prompt,
                tools=tools,
                tool_choice="auto",
                stream=False,  # 工具选择使用非流式
                temperature=0.1,  # 使用较低的温度以获得更确定的工具选择
                max_tokens=1000
            ):
                response = chunk
            
            return response
            
        except Exception as e:
            logger.error(f"调用LLM服务失败: {e}")
            raise
    
    def _build_tool_selection_prompt(self, question: str, context: dict[str, Any], tools_schema: dict) -> str:
        """构建工具选择提示词"""
        prompt = f"""你是一个智能助手，需要根据用户问题选择合适的工具来解决问题。

用户问题: {question}

上下文信息:
- 安全级别: {context.get('security', 0)}
- 标签: {context.get('tags', [])}
- 预处理问题: {context.get('processed_question', 'N/A')}

可用工具:
"""
        for tool_name, tool_info in tools_schema.items():
            prompt += f"- {tool_name}: {tool_info['description']}\n"
            schema_str = str(tool_info.get('schema', {}))
            prompt += f"  参数schema: {schema_str}\n"
        
        prompt += """
请根据用户问题选择最合适的工具，并给出调用参数。优先选择知识库搜索工具来获取准确信息。
如果涉及数学计算，使用计算器工具。如果需要最新信息，使用网络搜索工具。
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
            if (tool_result.tool_type in [MCPToolType.KB_SEARCH, MCPToolType.INTERNET_SEARCH] and 
                isinstance(tool_result.content, list)):
                all_kb_results.extend(tool_result.content)
        
        logger.debug(f"合并后总结果数: {len(all_kb_results)}")
        
        # 应用重排逻辑
        if len(all_kb_results) > 1 and self.agent_params.reranker_model_id:
            logger.debug("开始重排结果")
            reranker = AgentRerank(self.agent_params)
            reranked = await reranker.rerank_kb(question, all_kb_results)
            if reranked:
                all_kb_results = [
                    item for item in reranked 
                    if item.reranker_score >= self.agent_params.reranker_score_threshold
                ]
                logger.debug(f"重排后结果数: {len(all_kb_results)}")
        
        # 去重和排序
        seen = set()
        unique_results = []
        for item in all_kb_results:
            content = item.content.strip()
            if content not in seen:
                seen.add(content)
                unique_results.append(item)
        
        unique_results.sort(key=lambda x: x.weight, reverse=True)
        logger.debug(f"去重后最终结果数: {len(unique_results)}")
        
        return unique_results
    
    def _setup_agent_params(self, agent):
        """设置智能体参数"""
        self.agent_params = AgentParams(
            domain_id=agent.domain_id,
            prompt_id=agent.prompt_id,
            llm_id=agent.llm_id,
            llm_params=agent.llm_params,
            feedback_similarity_flag=agent.feedback_similarity_flag == 1,
            synonym_similarity_flag=agent.synonym_similarity_flag == 1,
            reranker_model_id=agent.reranker_model_id,
            reranker_top_k=agent.reranker_topk,
            reranker_score_threshold=agent.reranker_score_threshold or 0.0
        )
    
    async def _initialize_tools(self, agent):
        """初始化工具"""
        # 获取智能体的工具配置
        agent_conf_repo = KbotMdAgentConfRepository()
        confs = await agent_conf_repo.get_by_agent_id(self.agent_id)
        
        tools = []
        for conf in confs:
            if conf.tool_type == ToolType.KB_SEARCH.value:
                tool_params = ToolParams.from_orm(conf)
                kb_tool = KBSearchTool(tool_params)
                tools.append(kb_tool)
                logger.debug(f"创建知识库搜索工具: {conf.tool_id}")
            elif conf.tool_type == ToolType.INTERNET_SEARCH.value:
                tools.append(InternetSearchTool())
                logger.debug("创建网络搜索工具")
            # 可以继续添加其他工具类型
        
        # 注册默认工具
        tools.append(CalculatorTool())
        
        self.register_tools(tools)
    
    async def _preprocess_question(self, question: str) -> str:
        """预处理问题"""
        if self.agent_params.synonym_similarity_flag:
            logger.debug("问题改写启用同义词扩展")
        else:
            logger.debug("问题改写禁用同义词扩展")

        expand_question = await preprocess_cn_query(
            query=question, 
            enable_synonym_expansion=self.agent_params.synonym_similarity_flag
        )
        
        if expand_question is None:
            logger.warning(f"问题扩展失败: {question}")
            return question
        else:
            return ' '.join(expand_question) if isinstance(expand_question, list) else str(expand_question)
    
    async def chat(self, question: str) -> list[KBResult] | None:
        """
        基于MCP的智能体对话处理
        
        Args:
            question: 用户问题
            
        Returns:
            list[KBResult] | None: 知识库结果列表或None
        """
        logger.info(f"开始处理问题: {question}")
        
        # 1. 初始化智能体配置
        agent = await KbotMdAgentRepository().get_by_id(self.agent_id)
        if not agent:
            logger.warning("未找到智能体")
            return None
        
        self._setup_agent_params(agent)
        logger.debug("智能体参数设置完成")
        
        # 2. 初始化工具注册表
        await self._initialize_tools(agent)
        logger.debug("工具初始化完成")
        
        # 3. 预处理问题
        processed_question = await self._preprocess_question(question)
        logger.debug(f"问题预处理完成: {processed_question}")
        
        # 4. 让LLM选择工具
        context = {
            "security": self.security,
            "tags": self.tags,
            "processed_question": processed_question
        }
        
        tool_calls = await self._call_llm_for_tool_selection(question, context)
        logger.info(f"LLM选择了 {len(tool_calls)} 个工具: {[tc.tool_name for tc in tool_calls]}")
        
        # 5. 并行执行工具
        tool_results = await self._execute_tools_parallel(tool_calls)
        logger.info(f"工具执行完成，获得 {len(tool_results)} 个结果")
        
        # 6. 合并和重排结果
        final_results = await self._combine_and_rerank_results(question, tool_results)
        logger.info(f"最终返回 {len(final_results)} 个结果")
        
        return final_results