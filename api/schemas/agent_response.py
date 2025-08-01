from pydantic import BaseModel, Field

class BaseResponse(BaseModel):
    code: int = Field(..., description="The response code")
    message: str = Field(..., description="The response from the agent")
    success: bool = Field(..., description="The response status")

class SuccessResponse(BaseResponse):
    code: int = Field(200, description="The response code")
    message: str = Field("Success", description="The response from the agent")
    success: bool = Field(True, description="The response status")

class ErrorResponse(BaseResponse):
    code: int = Field(400, description="The response code")
    message: str = Field("Error", description="The response from the agent")
    success: bool = Field(False, description="The response status")

class SuccessQueryResponse(SuccessResponse):
    data: dict = Field(..., description="The response data")