import json
from utils.common import safe_read_content


class Reference:
    """参考文献数据类"""
    def __init__(self, 
                 chunk_type: int,
                 chunk_file_path: str,
                 page_num: int,
                 content: str,
                 download_link: str,
                 preview_link: str,
                 similarity_score: float,
                 reranker_score: float | None = None):
        self.chunk_type = chunk_type
        self.chunk_file_path = chunk_file_path
        self.page_num = page_num
        self.content = content
        self.download_link = download_link
        self.preview_link = preview_link
        self.similarity_score = similarity_score
        self.reranker_score = reranker_score
    
    def to_dict(self) -> dict:
        return {
            "chunk_type": self.chunk_type,
            "chunk_file_path": self.chunk_file_path,
            "page_num": self.page_num,
            "content": safe_read_content(self.content),
            "download_link": self.download_link,
            "preview_link": self.preview_link,
            "similarity_score": self.similarity_score,
            "reranker_score": self.reranker_score
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

class QAData:
    """问答数据类"""
    def __init__(self,
                 question: str,
                 answer: str,
                 qa_embedding: list,
                 references: list[Reference],
                 feedback: int,
                 by: str,
                 request_time: str | None = None,
                 response_time: str | None = None):
        self.question = question
        self.answer = answer
        self.qa_embedding = qa_embedding
        self.references = references
        self.feedback = feedback
        self.by = by
        self.request_time = request_time
        self.response_time = response_time
    
    def to_dict(self) -> dict:
        # 确保qa_embedding是Python列表，而不是array.array
        qa_embedding_list = self.qa_embedding
        if hasattr(qa_embedding_list, 'tolist'):
            qa_embedding_list = qa_embedding_list.tolist() # type: ignore
        elif isinstance(qa_embedding_list, list):
            qa_embedding_list = list(qa_embedding_list)
        
        return {
            "question": self.question,
            "answer": self.answer,
            "qa_embedding": qa_embedding_list,
            "references": [ref.to_dict() for ref in self.references],
            "feedback": self.feedback,
            "by": self.by,
            "request_time": self.request_time,
            "response_time": self.response_time
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

class KbotBizChatSession:
    """问答会话类"""
    def __init__(self,
                 session_id: str,
                 agent_id: int,
                 qa_data: list[QAData]):
        self.session_id = session_id
        self.agent_id = agent_id
        self.qa_data = qa_data
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "qa_data": [qa.to_dict() for qa in self.qa_data]
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    