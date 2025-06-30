
import json
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from backend.api.controllers.kb_upload_controller import upload_knowledge_base_files
from backend.api.schemas.kb_upload_schema import KBUploadRequest, KBUploadForm
from backend.api.schemas.kb_response import KBUploadResponse

router = APIRouter(
    prefix="/api/knowledge-base",
    tags=["Knowledge Base"]
)

@router.post(
    "/upload",
    summary="Upload files to the knowledge base. 上传文件到知识库",
    description="Upload one or more files to the specified knowledge base. 上传一个或多个文件到指定的知识库",
    response_model=KBUploadResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": "Files uploaded successfully",
                        "code": 200,
                        "data": {
                            "id": "123",
                            "name": "sample_kb",
                            "description": "Knowledge base for sample_kb",
                            "created_at": "2023-01-01T00:00:00",
                            "updated_at": "2023-01-01T00:00:00"
                        }
                    }
                }
            }
        },
        404: {
            "description": "Knowledge base not found",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "message": "Knowledge base sample_kb does not exist",
                        "code": 404,
                        "error_type": "knowledge_base_not_found",
                        "details": None
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "message": "Upload records save failed",
                        "code": 500,
                        "error_type": "database_error",
                        "details": None
                    }
                }
            }
        },
        422: {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "metadata"],
                                "msg": "field required",
                                "type": "value_error.missing"
                            }
                        ]
                    }
                }
            }
        }
    }
)

async def upload_files_to_knowledge_base(
    files: List[UploadFile] = File(...),
    metadata: str = Form(...)
):
    try:
        # Parse JSON metadata
        metadata_dict = json.loads(metadata)
        upload_request = KBUploadRequest(**metadata_dict)
        
        # Create form object
        form = KBUploadForm(files=files, metadata=upload_request)
        
        # Call the controller
        return await upload_knowledge_base_files(form)
        
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON format for metadata: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )