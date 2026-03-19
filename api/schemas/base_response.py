from pydantic import BaseModel, Field
from typing import Generic, TypeVar

T = TypeVar('T')

class SuccessResponse(BaseModel, Generic[T]):
    """KBOT API success response model.
    
    A generic response model for standardizing successful API responses in the KBOT system,
    encapsulating user-friendly messages and business data.
    """
    message: str = Field("Success", description="Response message (displayed to end users on the frontend)")
    data: T | None = Field(default=None, description="Business data returned in the response (generic type)")
    
    # Explicitly set model configuration
    model_config = {
        "arbitrary_types_allowed": True,
    }