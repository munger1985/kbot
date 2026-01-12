import asyncio
import json
from loguru import logger
from typing import Any
from .agent_params import AgentParams
from mcp_tools import *
from core.dictionary import MCPToolType
from dao.repositories.kbot_md_agent_repo import KbotMdAgentRepository
from utils.model_client import CallModel


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
        logger.info(f"成功注册 {len(tools)} 个工具: {[tool.tool_name for tool in tools]}")
    
    async def _call_llm_for_tool_selection(self, question: str, context: dict[str, Any]) -> list[Tool]:
        """
        调用大模型选择要使用的工具
        
        Args:
            question: 用户问题
            context: 上下文信息
            
        Returns:
            List[Tool]: 工具调用列表
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
            
            # 调用LLM服务进行工具选择
            model_id = self.agent_params.llm_id  # 使用智能体配置的LLM ID
            if model_id is None:
                raise ValueError("智能体配置中未指定 LLM")
            
            # 调用LLM服务
            try:
                async for chunk in CallModel().call_llm_model(
                    model_id=model_id,
                    prompt=prompt,
                    tools=tools,
                    tool_choice="auto",
                    stream=False,  # 工具选择使用非流式
                    temperature=0.1  # 使用较低的温度以获得更确定的工具选择
                ):
                    logger.debug(f"LLM工具选择响应chunk: {chunk}")
                    llm_response = chunk
                    
            except Exception as e:
                logger.error(f"LLM工具选择失败: {e}")
                return []
            
            # 解析LLM返回的工具调用
            tool_calls = []

            if not llm_response:
                logger.warning("LLM响应为空")
                return []
                
            logger.debug(f"LLM响应类型: {type(llm_response)}")
            
            # 检查响应格式并解析
            if isinstance(llm_response, str):
                # 如果是字符串，尝试解析JSON
                try:
                    llm_response = json.loads(llm_response)
                    logger.debug("成功将字符串响应解析为JSON")
                except json.JSONDecodeError:
                    logger.warning(f"LLM响应不是有效的JSON: {llm_response}")
                    return []
            
            # 直接从根级别解析 tool_calls
            if isinstance(llm_response, dict):
                if 'tool_calls' in llm_response:
                    response_tool_calls = llm_response['tool_calls']
                    logger.info(f"从根级别找到 {len(response_tool_calls)} 个工具调用")
                    
                    for tool_call_data in response_tool_calls:
                        try:
                            if isinstance(tool_call_data, dict) and 'function' in tool_call_data:
                                function_data = tool_call_data['function']
                                tool_name = function_data.get('name')
                                arguments_str = function_data.get('arguments', '{}')
                                
                                if tool_name:
                                    # 解析参数
                                    try:
                                        parameters = json.loads(arguments_str)
                                    except json.JSONDecodeError:
                                        logger.warning(f"工具参数解析失败: {arguments_str}")
                                        parameters = {}
                                    
                                    # 确定工具类型
                                    tool_info = tools_schema.get(tool_name, {})
                                    tool_type_str = tool_info.get('type', 'kb_search')
                                    try:
                                        tool_type = MCPToolType(tool_type_str)
                                    except ValueError:
                                        tool_type = MCPToolType.KB_SEARCH
                                    
                                    tool_calls.append(Tool(
                                        tool_type=tool_type,
                                        tool_name=tool_name,
                                        parameters=parameters,
                                        description=tool_info.get('description', '')
                                    ))
                                    
                                    logger.info(f"成功解析工具调用: {tool_name} with params: {parameters}")
                                else:
                                    logger.warning(f"工具调用缺少name字段: {tool_call_data}")
                            else:
                                logger.warning(f"工具调用格式不正确: {tool_call_data}")
                                
                        except Exception as e:
                            logger.error(f"解析工具调用失败: {e}, 数据: {tool_call_data}")
                            continue
                else:
                    logger.warning("响应中未找到tool_calls字段")
                    # 记录所有可用字段用于调试
                    logger.debug(f"响应字段: {list(llm_response.keys())}")
            else:
                logger.warning(f"LLM响应不是字典类型: {type(llm_response)}")
            
            logger.info(f"总共解析出 {len(tool_calls)} 个工具调用")
            return tool_calls
            
        except Exception as e:
            logger.error(f"LLM工具选择失败: {e}")
            return []

    
    def _build_tool_selection_prompt(self, question: str, context: dict[str, Any], tools_schema: dict) -> str:
        """构建工具选择提示词"""
        prompt = f"""你是一个智能助手，需要根据用户问题，改写问题以适用于知识库搜索，并选择合适的工具来解决问题。

用户问题: {question}

上下文信息:
- 安全级别: {context.get('security', 0)}
- 标签: {context.get('tags', [])}

可用工具:
"""
        for tool_name, tool_info in tools_schema.items():
            prompt += f"- {tool_name}: {tool_info['description']}\n"
            schema_str = str(tool_info.get('schema', {}))
            prompt += f"  参数schema: {schema_str}\n"
        
        prompt += """

## 问题改写优化指南

**原始问题**: "{question}"

请将问题改写成更适合知识库检索的版本：

### 改写原则：
1. **保持自然语言**：不要简单堆砌关键词，要使用完整的自然语句
2. **语义完整性**：确保改写后的问题能完整表达原问题的意思  
3. **检索友好**：既要便于语义理解，也要包含核心实体和概念

### 针对当前问题的改写建议：
- 避免简单的"关键词 关键词"堆砌
- 使用完整的疑问句或陈述句

### 搜索模式选择：
- **hybrid** (推荐): 兼顾语义理解和关键词匹配
- **vector**: 当问题复杂需要深度语义理解时
- **fulltext**: 当问题包含具体术语需要精确匹配时
- **summary**: 当问题宽泛需要主题检索时

请输出优化后的问题，确保既自然又适合检索。
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
    
    async def _execute_tools_parallel(self, tool_calls: list[Tool]) -> list[ToolResult]:
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
    
    
    def _setup_agent_params(self, agent, topk: int | None = None, score_threshold: float | None = None):
        """设置智能体参数"""
        self.agent_params = AgentParams(
            domain_id=agent.domain_id,
            prompt_id=agent.prompt_id,
            llm_id=agent.llm_id,
            llm_params=agent.llm_params,
            feedback_similarity_flag=agent.feedback_similarity_flag == 1,
            synonym_similarity_flag=agent.synonym_similarity_flag == 1,
            reranker_model_id=agent.reranker_model_id,
            reranker_top_k=topk or agent.reranker_topk,
            reranker_score_threshold=score_threshold or agent.reranker_score_threshold or 0.0,
        )
    
    async def _initialize_tools(self):
        """初始化工具"""

        tools = []
        # 注册知识库搜索工具
        kb_search_tool = KBSearchTool(self.agent_id)
        tools.append(kb_search_tool)
        
        # tools.append(InternetSearchTool())  # 互联网搜索工具需要付费API，暂不启用

        # 注册默认工具
        tools.append(CalculatorTool())
        
        self.register_tools(tools)
    
    async def chat(self, question: str, topk: int | None = None, score_threshold: float | None = None) -> list[ToolResult]:
        """
        基于MCP的智能体对话处理
        
        Args:
            question: 用户问题
            topk: 知识库检索TopK，默认None
            score_threshold: 知识库检索分数阈值，默认None
            
        Returns:
            list[ToolResult] | None: 工具结果列表或None
        """
        logger.info(f"开始处理问题: {question}")
        
        # 1. 初始化智能体配置
        agent = await KbotMdAgentRepository().get_by_id(self.agent_id)
        if not agent:
            logger.warning("未找到智能体")
            return []
        
        self._setup_agent_params(agent, topk, score_threshold)
        logger.debug("智能体参数设置完成")
        
        # 2. 初始化工具注册表
        await self._initialize_tools()
        logger.debug("工具初始化完成")
        
        # 3. 让LLM选择工具，同时预处理问题
        context = {
            "security": self.security,
            "tags": self.tags
        }
        
        tool_calls = await self._call_llm_for_tool_selection(question, context)
        logger.info(f"LLM选择了 {len(tool_calls)} 个工具: {[tc.tool_name for tc in tool_calls]}")
        
        # 4. 并行执行工具
        tool_results = await self._execute_tools_parallel(tool_calls)
        logger.info(f"工具执行完成，获得 {len(tool_results)} 个结果")
        
        return tool_results