from loguru import logger
from dao.repositories.kbot_md_agent_conf_repo import KbotMdAgentConfRepository
from dao.repositories.kbot_md_agent_repo import KbotMdAgentRepository
from dao.repositories.kbot_md_models_repo import KbotMdModelsRepository
from core.dictionary import ToolType, YesNoEnum
from .agent_params import AgentParams, ToolParams, KBResult
from ..search.kb_search import KBSearch
from .agent_rerank import AgentRerank


class Agent:
    """智能体类"""
    
    def __init__(self, agent_id: int, security: int):
        """
        初始化智能体
        
        Args:
            agent_id: 智能体ID
            security: 安全级别
        """
        self.agent_id = agent_id
        self.security = security
        self.agent_params = AgentParams()

    async def chat(self, question: str) -> list[KBResult] | None:
        """
        智能体对话处理
        
        Args:
            question: 用户问题
            
        Returns:
            list[KBResult] | None: 知识库结果列表或None
        """
        # 1. 获取智能体的默认配置信息
        agent = await KbotMdAgentRepository().get_by_id(self.agent_id)
        if not agent:
            logger.warning("未找到智能体")
            return None
            
        if agent.reranker_model_id is None:
            logger.warning("智能体重排模型ID为空")
            model_unique_name = None
        else:
            model_repo = KbotMdModelsRepository()
            model_unique_name = await model_repo.get_unique_name_by_id(agent.reranker_model_id)

        # 设置智能体参数
        self.agent_params.domain_id = agent.domain_id
        self.agent_params.prompt_id = agent.prompt_id
        self.agent_params.llm_id = agent.llm_id
        self.agent_params.llm_params = agent.llm_params
        self.agent_params.feedback_similarity_flag = True if agent.feedback_similarity_flag == 1 else False
        self.agent_params.synonym_similarity_flag = True if agent.synonym_similarity_flag == 1 else False
        self.agent_params.reranker_model_id = agent.reranker_model_id
        self.agent_params.reranker_top_k = agent.reranker_topk
        self.agent_params.reranker_score_threshold = agent.reranker_score_threshold
        self.agent_params.reranker_model_name = model_unique_name

        # 2. 获取智能体包含的知识库或工具配置信息
        agent_conf_repo = KbotMdAgentConfRepository()
        confs = await agent_conf_repo.get_by_agent_id(self.agent_id)
        if not confs:
            logger.warning("未找到智能体配置")
            return None
        
        kb_results_rerank: list[KBResult] = []
        kb_results_non_rerank: list[KBResult] = []
        kb_results: list[KBResult] = []
        
        logger.debug(f"找到 {len(confs)} 个工具")
        for conf in confs:
            # 生成工具参数，用于不同工具的调用
            logger.debug(f"工具ID: {conf.tool_id}")
            tool_params = ToolParams()
            tool_params.conf_id = conf.conf_id
            tool_params.tool_id = conf.tool_id
            tool_params.tool_type = conf.tool_type
            tool_params.tool_weight = conf.tool_weight or 0.0
            tool_params.reranker_flag = conf.reranker_flag or 0
            tool_params.search_type = conf.search_type
            tool_params.top_k = conf.search_topk or 10
            tool_params.threshold = conf.search_score_threshold or 0.7
            
            # 3. 根据配置的工具类型调用不同的工具
            # 知识库工具
            if tool_params.tool_type == ToolType.KB.value:
                logger.debug("工具类型: 知识库")
                kb = KBSearch(tool_params)
                result = await kb.search(question, self.security, self.agent_params.synonym_similarity_flag)
                if result:
                    # 如果开启了重排，则将结果添加到重排列表中
                    if tool_params.reranker_flag == YesNoEnum.YES.value:
                        kb_results_rerank += result
                    # 如果没有开启重排，则将结果添加到非重排列表中
                    else:
                        kb_results_non_rerank += result
                else:
                    logger.warning("知识库搜索结果为空")
                    continue
            # 函数调用工具
            elif tool_params.tool_type == ToolType.FUNCTIONCALL.value:
                logger.debug("工具类型: 函数调用")
                pass
            # 网络搜索工具
            elif tool_params.tool_type == ToolType.INTERNET.value:
                logger.debug("工具类型: 网络搜索")
                pass
            # 代理智能体工具
            elif tool_params.tool_type == ToolType.AGENT.value:
                logger.debug("工具类型: 代理智能体")
                pass
            # ChatAI工具
            elif tool_params.tool_type == ToolType.CHATAI.value:
                logger.debug("工具类型: ChatAI")
                pass
            # 其他类型暂不支持
            else:
                logger.warning("不支持的工具类型")
                continue
        
        # 4. 智能体范围内所有知识库查询和工具调用的结果合并后，进行重排（配置决定是否需要重排）
        # 如果重排结果大于1个，则进行重排，否则不进行重排
        if len(kb_results_rerank) > 1:
            reranker = AgentRerank(self.agent_params)
            reranked = await reranker.rerank_kb(question, kb_results_rerank)
            if reranked:
                # 根据重排阈值，提取出大于等于阈值的重排结果
                kb_results += [item for item in reranked if item.reranker_score >= self.agent_params.reranker_score_threshold]  # type: ignore

            # 计算重排结果的权重值，权重值等于数组中每个KBResult对象的weight值的加权平均值
            for index, result in enumerate(kb_results):
                result.weight = sum([item.weight for item in kb_results]) / len(kb_results)

        # 如果重排结果等于1个，则直接返回结果
        elif len(kb_results_rerank) == 1:
            logger.debug("只有1个结果，无需重排")
            kb_results += kb_results_rerank
        # 没有需要重排的结果
        else:
            logger.debug("向量搜索未返回结果，无需重排")
            pass

        # 添加非重排结果
        if len(kb_results_non_rerank) > 0:
            kb_results += kb_results_non_rerank

        # 5. 最后根据权重进行排序，权重值最大的排在前面
        kb_results.sort(key=lambda x: x.weight, reverse=True)  # type: ignore

        # 6. 对内容进行去重
        seen = set()
        unique_kb_results = []
        for item in kb_results:
            content = item.content.strip()  # 去除首尾空格
            if content not in seen:
                seen.add(content)
                unique_kb_results.append(item)
        kb_results = unique_kb_results

        # 7. 返回最终结果
        return kb_results