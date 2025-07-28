
from loguru import logger
from dao.repositories.kbot_md_agent_conf_repo import KbotMdAgentConfRepository
from dao.data_dict import ToolType, YesNoEnum
from .agent_params import AgentParams
from .kb_search import KBSearch


class Agent:
    """智能体类"""
    def __init__(self):
        self.agent_params = AgentParams()

    async def chat(self, agent_id: int, chat: str):
        """Agent chat"""

        # Get agent conf
        confs = await KbotMdAgentConfRepository().get_by_agent_id(agent_id)
        if not confs:
            logger.error("AgentConf not found")
        
        kb_results_rerank = []
        kb_results_non_rerank = []

        for conf in confs:
            self.agent_params.conf_id = conf.conf_id
            self.agent_params.agent_id = conf.agent_id
            self.agent_params.tool_id = conf.tool_id
            self.agent_params.tool_type = conf.tool_type
            self.agent_params.tool_weight = conf.tool_weight or 0.0
            self.agent_params.reranker_flag = conf.reranker_flag or 0
            self.agent_params.search_type = conf.search_type
            self.agent_params.top_k = conf.search_topk or 10
            self.agent_params.threshold = conf.search_score_threshold or 0.7

            # Call tools
            if self.agent_params.tool_type == ToolType.KB.value:
                kb = KBSearch(self.agent_params)
                result = await kb.search(chat)
                if result:
                    # Rerank if configured
                    if self.agent_params.reranker_flag == YesNoEnum.YES.value:
                        kb_results_rerank += result
                    else:
                        kb_results_non_rerank += result
            elif self.agent_params.tool_type == ToolType.FUNCTIONCALL.value:
                pass
            elif self.agent_params.tool_type == ToolType.INTERNET.value:
                pass
            elif self.agent_params.tool_type == ToolType.AGENT.value:
                pass
            elif self.agent_params.tool_type == ToolType.CHATAI.value:
                pass
            else:
                logger.error("Invalid tool type")
                raise ValueError("Invalid tool type")
        
            # Rerank if configured
            if self.agent_params.reranker_flag == YesNoEnum.YES.value:
                pass
                
        
            else:
                # kb_results = sorted(kb_results, key=lambda x: x.similarity, reverse=True)
                pass


        return kb_results_rerank + kb_results_non_rerank

