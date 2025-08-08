class AgentParams:
    def __init__(self):
        #self.agent_id: int
        self.domain_id: int | None
        self.prompt_id: int | None
        self.llm_id: int | None
        self.llm_params: dict | None
        self.feedback_similarity_flag: bool = False
        self.synonym_similarity_flag: bool = False
        self.reranker_model_id: int | None
        self.reranker_model_name: str | None
        self.reranker_top_k: int | None
        self.reranker_score_threshold: float | None = 0


class ToolParams:
    def __init__(self):
        self.conf_id: int
        self.tool_id: int
        self.tool_type: int
        self.tool_weight: float | None
        self.reranker_flag: int | None
        self.search_type: int | None
        self.top_k: int | None
        self.threshold: float | None
        self.kb_catogory: int | None
        self.img2txt_model: int | None
        self.img_embed_model: int | None
        self.txt_embed_model: int | None

class KBResult:
    def __init__(self):
        self.file_id: str
        self.chunk_type: int
        self.page_num: int = 0
        self.content: str = ""
        self.similarity: float = 0.0
        self.weight: float = 0.0
        self.reranker_score: float = 0.0