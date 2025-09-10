

class KbotBizTxtEmbedding():
    """文本向量嵌入表"""
    def __init__(self, embed_id: str, kb_id: int, file_id: str, chunk_doc: str, chunk_metadata: dict, embedding: list, security_level: int):
        self.embed_id = embed_id
        self.kb_id = kb_id
        self.file_id = file_id
        self.chunk_doc = chunk_doc
        self.chunk_metadata = chunk_metadata
        self.embedding = embedding
        self.security_level = security_level
    