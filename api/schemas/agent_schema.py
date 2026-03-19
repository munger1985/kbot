from pydantic import BaseModel, Field

class AgentChatForm(BaseModel):
    """Agent chat form model.
    
    This model defines the data structure for sending chat requests to an AI agent,
    including session information, user identity, agent configuration and query content.
    """
    session_id: str = Field(..., description="Session ID (unique identifier for the chat session)")
    by: str = Field(..., description="Request user ID (identifier of the user initiating the request)")
    agent_id: int = Field(..., description="Agent ID (unique identifier for the AI agent)")
    security_level: int = Field(0, description="Security level (default: 0)")
    request_time: str = Field(..., description="Request time (timestamp string, e.g. ISO 8601 format)")
    question: str = Field(..., description="User's question/query content")
    tags: list[str] = Field([], description="Tags associated with the query (default: empty list)")
    deep_mind: int = Field(0, description="Whether to use deep thinking mode (0: disabled, 1: enabled)")

class AgentChatFeedbackForm(BaseModel):
    """Agent chat feedback form model.
    
    This model defines the data structure for submitting feedback on AI agent chat responses.
    """
    chat_record_id: int = Field(..., description="Chat record ID (unique identifier for the chat history entry)")
    feedback: int = Field(..., description="Feedback value (0: no feedback, 1: approve, -1: disapprove)")

class DifySearchForm(BaseModel):
    """Dify knowledge base retrieval form model.
    
    This model defines the data structure for performing retrieval operations 
    against Dify knowledge base.
    """
    knowledge_id: str = Field(..., description="Knowledge base ID (unique identifier for the target knowledge base)")
    query: str = Field(..., description="Query text (the content to search for in the knowledge base)")
    retrieval_setting: dict = Field(..., description="Retrieval settings (e.g. top_k, score_threshold, etc.)")
    metadata_condition: dict | None = Field(None, description="Metadata filter conditions (optional)")