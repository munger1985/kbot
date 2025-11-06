import asyncio
from typing import Any
from loguru import logger
from dao.repositories.kbot_md_agent_conf_repo import KbotMdAgentConfRepository
from dao.repositories.kbot_md_agent_repo import KbotMdAgentRepository
from core.dictionary import ToolType, YesNoEnum
from mcp_tools import KBSearchResult, KBSearchToolParams
from .agent_params import AgentParams
from .agent_rerank import AgentRerank
from ..search.kb_search import KBSearch
from ..search.fulltext_preprocessor import preprocess_for_fulltext


class Agent:
    """智能体类"""
    
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

    async def _run_kb_search_async(self, 
                                   tool_params: KBSearchToolParams, 
                                   vector_search_question: str, 
                                   full_text_question: str, 
                                   security: int,
                                   tags: list[str] = []
                                   ) -> list[KBSearchResult]:
        """
        异步运行KB搜索的方法
        
        Args:
            tool_params: 工具参数
            vector_search_question: 改写后的向量搜索问题
            full_text_question: 改写后的全文搜索问题
            security: 安全级别
            tags: 标签列表. 默认为空列表
            
        Returns:
            list[KBSearchResult]: 搜索结果列表
        """
        try:
            kb = KBSearch(tool_params)
            result = await kb.search(vector_search_question, full_text_question, security, tags=tags)
            return result or []
        except Exception as e:
            logger.error(f"KB搜索执行失败: {e}")
            return []

    async def _execute_kb_searches_parallel(self, kb_tasks: list[tuple]) -> list[list[KBSearchResult]]:
        """
        并行执行所有KB搜索任务（使用线程池）
        
        Args:
            kb_tasks: KB任务列表，每个任务为 (tool_params, vector_search_question, full_text_question, security)
            
        Returns:
            list[list[KBSearchResult]]: 搜索结果列表
        """

        if not kb_tasks:
            return []
        
        # 使用线程池而不是进程池，避免事件循环问题
        results = []
        
        # 创建所有异步任务
        async_tasks = []
        for tool_params, vector_search_question, full_text_question, security, tags in kb_tasks:
            async_tasks.append(
                self._run_kb_search_async(tool_params, vector_search_question, full_text_question, security, tags)
            )
        
        # 并行执行所有任务
        try:
            logger.debug(f"开始并行执行 {len(async_tasks)} 个KB搜索任务")
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            return self._process_kb_results(results)
        except Exception as e:
            logger.error(f"并行KB搜索执行失败: {e}")
            return [[] for _ in async_tasks]

    def _process_kb_results(self, results: list[Any]) -> list[list[KBSearchResult]]:
        """
        处理KB搜索结果
        
        Args:
            results: 原始结果列表
            
        Returns:
            list[list[KBSearchResult]]: 处理后的KBSearchResult列表
        """
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"KB搜索任务 {i} 执行失败: {result}")
                processed_results.append([])
            elif result is None:
                logger.warning(f"KB搜索任务 {i} 返回空结果")
                processed_results.append([])
            elif isinstance(result, list):
                logger.debug(f"KB搜索任务 {i} 找到 {len(result)} 条结果")
                processed_results.append(result)
            else:
                logger.warning(f"KB搜索任务 {i} 返回未知类型结果: {type(result)}")
                processed_results.append([])
        return processed_results

    async def _process_kb_tools(self, confs: list[Any], vector_search_question: str, full_text_question: str) -> tuple:
        """
        处理知识库工具
        
        Args:
            confs: 配置列表
            vector_search_question: 改写后的向量搜索问题
            full_text_question: 改写后的全文搜索问题
            
        Returns:
            tuple: (重排结果列表, 非重排结果列表)
        """
        
        kb_results_rerank: list[KBSearchResult] = []
        kb_results_non_rerank: list[KBSearchResult] = []
        
        # 收集所有KB搜索任务
        kb_tasks = []
        kb_configs = []  # 保存配置信息
        
        for conf in confs:
            if conf.tool_type == ToolType.KB_SEARCH.value:
                logger.debug(f"知识库工具ID: {conf.tool_id}")
                
                # 直接从ORM对象创建KBSearchToolParams
                tool_params = KBSearchToolParams.from_orm(conf)
                
                # 添加到并行任务列表
                kb_tasks.append((
                    tool_params,
                    vector_search_question,
                    full_text_question,
                    self.security,
                    self.tags
                ))
                kb_configs.append(conf)
        
        # 并行执行所有KB搜索
        if kb_tasks:
            all_kb_results = await self._execute_kb_searches_parallel(kb_tasks)
            
            # 处理搜索结果
            for conf, kb_result_list in zip(kb_configs, all_kb_results):
                if kb_result_list:
                    # 如果开启了重排，则将结果添加到重排列表中
                    if conf.reranker_flag == YesNoEnum.YES.value:
                        kb_results_rerank.extend(kb_result_list)
                        logger.debug(f"知识库 {conf.tool_id} 添加到重排列表: {len(kb_result_list)} 条结果")
                    # 如果没有开启重排，则将结果添加到非重排列表中
                    else:
                        kb_results_non_rerank.extend(kb_result_list)
                        logger.debug(f"知识库 {conf.tool_id} 添加到非重排列表: {len(kb_result_list)} 条结果")
                else:
                    logger.warning(f"知识库 {conf.tool_id} 搜索结果为空")
        
        logger.debug(f"重排结果数: {len(kb_results_rerank)}, 非重排结果数: {len(kb_results_non_rerank)}")
        return kb_results_rerank, kb_results_non_rerank

    async def _process_non_kb_tools(self, confs: list[Any], question: str) -> list[Any]:
        """
        处理非知识库工具
        
        Args:
            confs: 配置列表
            question: 用户问题
            
        Returns:
            list[Any]: 非KB工具结果列表
        """
        non_kb_results = []
        
        for conf in confs:
            # 函数调用工具
            if conf.tool_type == ToolType.FUNCTION_CALL.value:
                logger.debug("工具类型: 函数调用")
                # 这里可以添加函数调用逻辑
                pass
            
            # 网络搜索工具
            elif conf.tool_type == ToolType.INTERNET_SEARCH.value:
                logger.debug("工具类型: 网络搜索")
                # 这里可以添加网络搜索逻辑
                pass
            
            # 其他类型暂不支持
            else:
                logger.warning(f"不支持的工具类型: {conf.tool_type}")
        
        return non_kb_results

    async def _rerank_and_process_results(self, question: str, kb_results_rerank: list[KBSearchResult], 
                                         kb_results_non_rerank: list[KBSearchResult]) -> list[KBSearchResult]:
        """
        重排和处理最终结果
        
        Args:
            question: 用户问题
            kb_results_rerank: 需要重排的结果
            kb_results_non_rerank: 不需要重排的结果
            
        Returns:
            list[KBSearchResult]: 最终结果列表
        """
        kb_results: list[KBSearchResult] = []
        
        # 如果重排结果大于1个，则进行重排
        if len(kb_results_rerank) > 1:
            reranker = AgentRerank(self.agent_params)
            reranked = await reranker.rerank_kb(question, kb_results_rerank)
            if reranked:
                # 根据重排阈值，提取出大于等于阈值的重排结果
                kb_results.extend([item for item in reranked if item.reranker_score >= self.agent_params.reranker_score_threshold])

            # 计算重排结果的权重值
            if kb_results:
                total_weight = sum(item.weight for item in kb_results)
                avg_weight = total_weight / len(kb_results)
                for result in kb_results:
                    result.weight = avg_weight

        # 如果重排结果等于1个，则直接返回结果
        elif len(kb_results_rerank) == 1:
            logger.debug("只有1个结果，无需重排")
            kb_results.extend(kb_results_rerank)

        # 添加非重排结果
        if kb_results_non_rerank:
            kb_results.extend(kb_results_non_rerank)

        # 根据权重进行排序
        kb_results.sort(key=lambda x: x.weight, reverse=True)

        # 对内容进行去重
        seen = set()
        unique_kb_results = []
        for item in kb_results:
            content = item.content.strip()
            if content not in seen:
                seen.add(content)
                unique_kb_results.append(item)

        return unique_kb_results

    async def chat(self, question: str) -> list[KBSearchResult] | None:
        """
        智能体对话处理
        
        Args:
            question: 用户问题
            
        Returns:
            list[KBSearchResult] | None: 知识库结果列表或None
        """

        # 1. 获取智能体的默认配置信息
        agent = await KbotMdAgentRepository().get_by_id(self.agent_id)
        if not agent:
            logger.warning("未找到智能体")
            return None

        # 设置智能体参数
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

        # 2. 预处理问题，用于向量检索和全文检索，语义检索需要字符串，全文检索需要词元列表
        if self.agent_params.synonym_similarity_flag:
            logger.debug(f"问题改写启用同义词扩展")
        else:
            logger.debug(f"问题改写禁用同义词扩展")

        expand_question: str | None = None
        if agent.llm_id is not None:   
            expand_question = await preprocess_for_fulltext(model_id=agent.llm_id, query=question)
        else:
            logger.warning("智能体LLM模型ID为空，无法进行问题扩展")
        
        if expand_question is None:
            logger.warning(f"问题扩展失败: {question}")
            # vector_search_question = question
            full_text_question = question
        else:
            # vector_search_question = expand_question.get("semantic", question)
            full_text_question = expand_question

        # 3. 获取智能体包含的知识库或工具配置信息
        agent_conf_repo = KbotMdAgentConfRepository()
        confs = await agent_conf_repo.get_by_agent_id(self.agent_id)
        if not confs:
            logger.warning("未找到智能体配置")
            return None
        
        logger.debug(f"找到 {len(confs)} 个工具")
        
        # 4. 并行处理知识库工具
        kb_results_rerank, kb_results_non_rerank = await self._process_kb_tools(confs, question, full_text_question) # type: ignore
        
        # 5. 处理非知识库工具
        # TODO: 目前非知识库工具未实现具体功能
        # non_kb_results = await self._process_non_kb_tools(confs, question) # type: ignore
        
        # 6. 重排和处理最终结果
        # 如果重排模型ID为空，则直接返回结果
        if agent.reranker_model_id is None:
            logger.warning("智能体重排模型ID为空")
            final_results = kb_results_rerank + kb_results_non_rerank
        else:
            final_results = await self._rerank_and_process_results(question, kb_results_rerank, kb_results_non_rerank)
        
        # 7. 返回最终结果
        return final_results