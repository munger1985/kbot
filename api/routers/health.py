from fastapi import APIRouter, status, HTTPException
from loguru import logger
from api.controllers.model_controller import model_controller as controller
from api.schemas.model_schema import *
from api.schemas.base_response import *
from core.auth.shortcuts import *

router = APIRouter(tags=["Health Check"])

@router.get("/health", summary="Health Check")
async def health_check():
    """
    ### Description
    The primary heartbeat endpoint for the service. Used by orchestrators (like Kubernetes or Docker) to verify if the container is running.

    ---
    ### Returns
    - `status`: Returns `"ok"` if the service is reachable.
    """
    return {"status": "ok"}


@router.post(
        "/model/test",
        summary="Test if the specified model is available"
)
async def handle_test_model(form: ModelForm, auth: AnyAuth) -> SuccessResponse:
    """
    ### Description
    Checks the connectivity and availability of a specific model within the system.

    ---
    ### Parameters
    - **model_id** (`int`): The unique identifier of the model to test.
    - **model_category** (`int`): The category code (e.g., LLM, Embedding, Rerank).

    ### Returns
    - **SuccessResponse**: If the model is online and responding.

    ### Error Handling
    - Returns **400 Bad Request** if the model is offline, misconfigured, or the ID is invalid.
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