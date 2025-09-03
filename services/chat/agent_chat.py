
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
        self.agent_id = agent_id
        self.security = security
        self.agent_params = AgentParams()

    async def chat(self, question: str) -> list[KBResult] | None:
        """Agent chat"""

        # 1. 获取 agent 的默认配置信息
        agent = await KbotMdAgentRepository().get_by_id(self.agent_id)
        if not agent:
            logger.warning("Agent not found")
            return None
        if agent.reranker_model_id is None:
            logger.warning("Agent reranker model ID is None")
            model_unique_name = None
        else:
            model_repo = KbotMdModelsRepository()
            model_unique_name = await model_repo.get_unique_name_by_id(agent.reranker_model_id)

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

        # 2. 获取 agent 包含的知识库或工具配置信息
        agent_conf_repo = KbotMdAgentConfRepository()
        confs = await agent_conf_repo.get_by_agent_id(self.agent_id)
        if not confs:
            logger.warning("Agent config not found")
            return None
        
        kb_results_rerank: list[KBResult] = []
        kb_results_non_rerank: list[KBResult] = []
        kb_results: list[KBResult]  = []
        logger.debug(f"Found {len(confs)} tools.")
        for conf in confs:
            # 生成 toolparams，用于不同工具的调用
            logger.debug(f"Tool ID: {conf.tool_id}")
            tool_params = ToolParams()
            tool_params.conf_id = conf.conf_id
            tool_params.tool_id = conf.tool_id
            tool_params.tool_type = conf.tool_type
            tool_params.tool_weight = conf.tool_weight or 0.0
            tool_params.reranker_flag = conf.reranker_flag or 0
            tool_params.search_type = conf.search_type
            tool_params.top_k = conf.search_topk or 10
            tool_params.threshold = conf.search_score_threshold or 0.7
            
            # 3. 根据配置的 tool_type 调用不同的工具：
            # 知识库
            if tool_params.tool_type == ToolType.KB.value:
                logger.debug("ToolType: knowledge base")
                kb = KBSearch(tool_params)
                result = await kb.search(question, self.security, self.agent_params.synonym_similarity_flag)
                if result:
                    # 如果开启了reranker，则将结果添加到rerank列表中
                    if tool_params.reranker_flag == YesNoEnum.YES.value:
                        kb_results_rerank += result
                    # 如果没有开启reranker，则将结果添加到非rerank列表中
                    else:
                        kb_results_non_rerank += result
                else:
                    logger.warning("KB search result is None")
                    continue
            # 函数调用
            elif tool_params.tool_type == ToolType.FUNCTIONCALL.value:
                pass
            # 网络搜索
            elif tool_params.tool_type == ToolType.INTERNET.value:
                pass
            # 代理智能体
            elif tool_params.tool_type == ToolType.AGENT.value:
                pass
            # ChatAI
            elif tool_params.tool_type == ToolType.CHATAI.value:
                pass
            # 其他类型暂不支持
            else:
                logger.warning("Unsupported tool type.")
                continue
        
        # 4. Agent 范围内所有KB查询和工具调用的结果合并后，进行rerank(配置决定是否需要rerank)
        # 如果 reranker 结果大于1个，则进行rerank，否则不进行rerank
        if len(kb_results_rerank) > 1:
            reranker = AgentRerank(self.agent_params)
            reranked = await reranker.rerank_kb(question, kb_results_rerank)
            if reranked:
                # 根据reranker阈值，提取出大于等于阈值的reranker结果
                kb_results += [item for item in reranked if item.reranker_score >= self.agent_params.reranker_score_threshold] # type: ignore

            # 计算 reranker 结果的权重值，权重值等于数组中每个kbresult对象的weight值的加权平均值
            for index, result in enumerate(kb_results):
                result.weight = sum([item.weight for item in kb_results]) / len(kb_results)

        # 如果 reranker 结果等于1个，则直接返回结果
        elif len(kb_results_rerank) == 1:
            logger.debug("Only 1 result, reranker is not needed.")
            kb_results += kb_results_rerank
        # 没有需要 reranker 的结果
        else:
            logger.debug("Vector Search returned no result, reranker is not needed.")
            pass

        if len(kb_results_non_rerank) > 0:
            kb_results += kb_results_non_rerank

        # 5. 最后根据权重进行排序，权重值最大的排在前面
        kb_results.sort(key=lambda x: x.weight, reverse=True) # type: ignore

        # 6. 对 content 进行去重
        seen = set()
        unique_kb_results = []
        for item in kb_results:
            content = item.content.strip()  # 去除首尾空格
            if content not in seen:
                seen.add(content)
                unique_kb_results.append(item)
        kb_results = unique_kb_results

        # 7. 返回结果
        return kb_results

