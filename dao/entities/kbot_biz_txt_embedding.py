import json
from utils.common import safe_read_content

class KbotBizTxtEmbedding():
    """文本向量嵌入表"""
    def __init__(self, embed_id: str, 
                 kb_id: int, 
                 file_id: str, 
                 chunk_doc: str, 
                 chunk_metadata: dict, 
                 biz_metadata: dict,
                 embedding: list, 
                 security_level: int,
                 status: int = 1):
        self.embed_id = embed_id
        self.kb_id = kb_id
        self.file_id = file_id
        self.chunk_doc = chunk_doc
        self.chunk_metadata = chunk_metadata
        self.biz_metadata = biz_metadata
        self.embedding = embedding
        self.security_level = security_level
        self.status = status
         
    def to_dict(self) -> dict:
        return {
            "embed_id": self.embed_id,
            "kb_id": self.kb_id,
            "file_id": self.file_id,
            "chunk_doc": safe_read_content(self.chunk_doc),
            "chunk_metadata": self.chunk_metadata,
            "biz_metadata": self.biz_metadata,
            "embedding": list(self.embedding),
            "security_level": self.security_level,
            "status": self.status
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    