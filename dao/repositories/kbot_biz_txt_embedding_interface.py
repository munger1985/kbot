from abc import ABC, abstractmethod
from typing import Sequence
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding

class IEmbeddingRepository(ABC):
    """嵌入存储库的统一接口"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化连接"""
        pass
    
    @abstractmethod
    async def create(self, kb_id: int, embeddings: list[KbotBizTxtEmbedding]) -> bool:
        """批量创建嵌入记录"""
        pass
    
    @abstractmethod
    async def delete_by_file_ids(self, kb_id: int, file_ids: list[str]) -> int:
        """根据文件ID删除嵌入记录"""
        pass
    
    @abstractmethod
    async def get_similar_embeddings(self,
                                   kb_id: int,
                                   query_vec: str,
                                   security: int,
                                   similarity_threshold: float | None = 0.8,
                                   search_top_k: int | None = 10,
                                   is_summary_search: bool = False,
                                   tags: list[str] | None = None) -> Sequence:
        """向量相似度搜索"""
        pass
    
    @abstractmethod
    async def full_text_search(self,
                             kb_id: int,
                             keyword: str,
                             security: int,
                             search_top_k: int | None = 10,
                             similarity_threshold: float | None = 0.8,
                             tags: list[str] | None = None) -> Sequence:
        """全文检索"""
        pass
    
    @abstractmethod
    async def update_chunk(self,
                         embed_id: str,
                         new_chunk: str,
                         new_embedding: list[float]) -> bool:
        """更新块内容和嵌入向量"""
        pass
    
    @abstractmethod
    async def get_summary_id_by_chunk_id(self, file_id: str, chunk_id: str) -> str | None:
        """根据块ID获取摘要ID"""
        pass
    
    @abstractmethod
    async def delete_by_embed_ids(self, embed_ids: list[str]) -> int:
        """根据嵌入ID删除记录"""
        pass
    
    @abstractmethod
    async def update_status_by_chunk_id(self, chunk_id: str, status: int) -> int:
        """更新块状态"""
        pass

    @abstractmethod
    async def get_chunks_by_file_id(self, file_id: str) -> list[KbotBizTxtEmbedding] | None:
        """根据文件ID获取块"""
        pass
    
    @abstractmethod
    async def get_chunk_doc_by_id(self, embed_id: str) -> str | None:
        """根据ID获取chunk文档"""
        pass

    @abstractmethod
    async def update_chunk_description(self, embed_id: str, description: str) -> int:
        """更新块描述"""
        pass

    @abstractmethod
    async def update_tags(self, embed_id: str, tags: list[str]) -> bool:
        """更新块标签"""
        pass