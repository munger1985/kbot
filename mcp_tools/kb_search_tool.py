from loguru import logger
from pydantic import BaseModel
from typing import Any
from mcp_tools import MCPTool, ToolResult
from core.dictionary import MCPToolType
from services.search.kb_search_for_mcp import KBSearch
from services.chat.agent_rerank import AgentRerank, AgentParams
from dao.repositories.kbot_md_agent_conf_repo import KbotMdAgentConfRepository
from dao.repositories.kbot_md_agent_repo import KbotMdAgentRepository


class KBSearchToolParams(BaseModel):
    """工具参数类"""
    conf_id: int = 0
    tool_id: int = 0
    tool_type: int = 0
    tool_weight: float = 0.0
    reranker_flag: int = 0
    search_type: int = 0
    search_top_k: int = 20
    threshold: float = 0.7
    kb_catogory: int | None = None
    img2txt_model: int | None = None
    img_embed_model: int | None = None
    txt_embed_model: int | None = None

    class Config:
        # 允许任意类型，避免序列化问题
        arbitrary_types_allowed = True
        from_attributes = True

    @classmethod
    def from_orm(cls, obj: Any) -> 'KBSearchToolParams':
        """从ORM对象创建KBSearchToolParams"""
        return cls(
            conf_id=getattr(obj, 'conf_id', 0),
            tool_id=getattr(obj, 'tool_id', 0),
            tool_type=getattr(obj, 'tool_type', 0),
            tool_weight=getattr(obj, 'tool_weight', 0.0) or 0.0,
            reranker_flag=getattr(obj, 'reranker_flag', 0) or 0,
            search_type=getattr(obj, 'search_type', 0),
            search_top_k=getattr(obj, 'search_top_k', 10) or 10,
            threshold=getattr(obj, 'search_score_threshold', 0.7) or 0.7,
            kb_catogory=None,
            img2txt_model=None,
            img_embed_model=None,
            txt_embed_model=None
        )


class KBSearchTool(MCPTool):
    """知识库搜索工具"""
    
    def __init__(self, agent_id: int):
        super().__init__(
            tool_type=MCPToolType.KB_SEARCH,
            tool_name="knowledge_base_search",
            description="搜索知识库获取相关信息"
        )
        self.agent_id = agent_id
    
    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            query = parameters.get("query", "")

            search_type = parameters.get("search_type", "hybrid")
            limit = parameters.get("limit", 100)
            
            results = []
            metadata_list = []
            # 获取智能体的工具配置
            agent_conf_repo = KbotMdAgentConfRepository()
            kb_ids = await agent_conf_repo.get_unique_kb_id(self.agent_id)
            for kb_id in kb_ids:
                
                kb = KBSearch(agent_id=self.agent_id, kb_id=kb_id)
            
                result = await kb.search(
                        questions=query,
                        search_type=search_type,
                        security=self.security,
                        tags=self.tags
                    )
                
                # 构造元数据
                metadata = {
                    "kb_id": kb_id,
                    "result_count": len(results),
                    "limit": limit
                }

                # 限制结果数量
                limited_result = result[:limit] if result else []
                
                # 合并结果
                results.extend(limited_result)
                metadata_list.append(metadata)
            
            # 重排序结果
            agent = await KbotMdAgentRepository().get_by_id(self.agent_id)
            if not agent:
                logger.warning("未找到智能体配置，无法进行重排序")
                return ToolResult(
                    tool_type=self.tool_type,
                    kb_results=results,
                    confidence=0.0,
                    metadata=metadata_list
                )

            # 设置智能体参数
            agent_params = AgentParams(
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
            
            agent = AgentRerank(agent_params=agent_params)
            kb_results = await agent.rerank_kb(
                question=query,
                kb_results=results
            )

            return ToolResult(
                tool_type=self.tool_type,
                kb_results=kb_results,
                confidence=0.9,
                metadata=metadata_list
            )
            
        except Exception as e:
            logger.error(f"知识库搜索失败: {e}")
            return ToolResult(
                tool_type=self.tool_type,
                kb_results=[],
                confidence=0.0,
                metadata=[{"error": str(e)}]
            )
    

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询语句"
                },
                "search_type": {
                    "type": "string",
                    "enum": ["vector", "fulltext", "summary", "hybrid"],
                    "description": "搜索类型：向量搜索、全文搜索、摘要搜索、混合搜索"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量限制",
                    "default": 100
                }
            },
            "required": ["query"]
        }