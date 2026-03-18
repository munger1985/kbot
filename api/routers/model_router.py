from fastapi import APIRouter, status, HTTPException
from loguru import logger
from api.controllers.model_controller import model_controller as controller
from api.schemas.model_schema import *
from api.schemas.base_response import *
from core.auth.shortcuts import *

router = APIRouter(prefix="/model", tags=["Model Management"])
   
@router.post(
        "/test",
        summary="Test if the specified model is available"
)
async def handle_test_model(form: ModelForm, auth: AnyAuth) -> SuccessResponse:
    """Tests if the specified model is available.

    Args:
        form: Test model request form with the following fields:
            - model_id: int = Field(..., description="Model ID")
            - model_category: int = Field(..., description="Model category")

    Returns:
        SuccessResponse: Success response with the following structure:
            - message: str = Field("Success", description="Response message")
    """
    
    if await controller.verify_model(form.model_id, form.model_category):
        return SuccessResponse(message="Model is available.")
    else:
        msg = f"Model {form.model_id} is not available."
        logger.error(msg)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg
        )