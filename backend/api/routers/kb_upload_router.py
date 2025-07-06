
import json
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from api.controllers.kb_controller import upload_knowledge_base_files
from api.schemas.kb_upload_schema import KBUploadRequest, KBUploadForm
from api.schemas.kb_response import SuccessResponse, ErrorResponse

router = APIRouter(
    prefix="/api/knowledge-base",
    tags=["Knowledge Base"]
)

@router.post(
    "/upload",
    summary="Upload files to the knowledge base. 上传文件到知识库",
    description="Upload one or more files to the specified knowledge base. 上传一个或多个文件到指定的知识库",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK
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
        result = await upload_knowledge_base_files(form)
        
        if result:
            return SuccessResponse(
                code=200,
                success=True,
                message="Upload files successfully."
            )
        else:
            return ErrorResponse(
                code=400,
                success=False,
                message="Upload files failed.",
                error_type="File Upload Error",
                details=None
            )
        
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