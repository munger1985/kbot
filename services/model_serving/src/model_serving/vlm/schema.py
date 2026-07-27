from pydantic import BaseModel, Field
from typing import Any
import uuid
import time


# Define request models
class VLMRequest(BaseModel):
    """VLM inference request model
    
    Request schema for Vision-Language Model inference requests, supporting both
    streaming and non-streaming responses with configurable sampling parameters.
    """

    served_model_name: str = Field(..., description="Name of the model to use for inference")
    messages: list[dict[str, Any]] = Field(..., description="List of messages containing text/image content")
    max_tokens: int | None = Field(None, description="Maximum number of tokens to generate")
    temperature: float | None = Field(None, description="Sampling temperature (0.0-1.0, lower = more deterministic)")
    stream: bool = Field(False, description="Whether to return response as a stream")
    timeout: int | None = Field(None, description="Request timeout in seconds")
    top_p: float | None = Field(None, description="Top-p sampling parameter")
    frequency_penalty: float | None = Field(None, description="Frequency penalty (reduces repetition)")
    presence_penalty: float | None = Field(None, description="Presence penalty (reduces topic repetition)")


class OpenAIVLMRequest(BaseModel):
    """OpenAI 兼容的多模态对话请求。"""

    model: str = Field(min_length=1, max_length=128)
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    timeout: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    def to_internal(self) -> VLMRequest:
        payload = self.model_dump(exclude={"model"})
        return VLMRequest(served_model_name=self.model, **payload)

# Define response models
class VLMResponse(BaseModel):
    """VLM inference response model (OpenAI-compatible)
    
    Response schema matching OpenAI's chat completion format with additional
    custom fields for processing metrics.
    """

    id: str = Field(default_factory=lambda: f"sse-{uuid.uuid4()}", description="Unique identifier for the response stream")
    object: str = Field("chat.completion", description="Object type, always 'chat.completion'")
    created: int = Field(default_factory=lambda: int(time.time()), description="Unix timestamp when the response was created")
    model: str = Field(..., description="Name of the model that generated the response")
    choices: list[dict[str, Any]] = Field(..., description="List containing the response message(s)")
    usage: dict[str, int] = Field(..., description="Token usage statistics including prompt_tokens, completion_tokens, and total_tokens")
