
import json
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from api.controllers.kb_controller import upload_kb_files, delete_kb_files, get_kb_files
from api.schemas.kb_schema import KBUploadForm, KBDeleteForm
from api.schemas.kb_response import SuccessResponse, ErrorResponse

router = APIRouter(
    prefix="/kb",
    tags=["Knowledge Base"]
)

@router.post(
    "/upload",
    summary="Upload files to the knowledge base. 上传文件到知识库",
    description="Upload one or more files to the specified knowledge base. 上传一个或多个文件到指定的知识库",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK
)
async def upload_files(
    files: list[UploadFile] = File(...),
    metadata: str = Form(...)
):
    try:
        # Parse and validate as form model
        metadata_dict = json.loads(metadata)
        form = KBUploadForm(files=files, **metadata_dict)
        
        # Call the controller
        result = await upload_kb_files(form)
        
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
                message="Upload files failed."
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
    
@router.post(
    "/delete",
    summary="Delete files from the knowledge base. 删除知识库中的文件",
    description="Delete files or all files from the specified knowledge base along with the knowledge base or batch. 从指定的知识库中删除文件或所有文件以及其知识库或批次",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK
)
async def delete_files(
    metadata: str = Form(...)
):
    try:
        # Parse and validate as form model
        metadata_dict = json.loads(metadata)
        form = KBDeleteForm(**metadata_dict)
        
        # Call the controller
        result = await delete_kb_files(form)


        if result["failed_file_cnt"] == 0 and result["meta_cnt"] > 0:
            return SuccessResponse(
                code=200,
                success=True,
                message="Delete files successfully."
            )
        else:
            return ErrorResponse(
                code=400,
                success=False,
                message=f"Delete files failed. Result: {result}"
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
    
@router.get(
    "/download",
    summary="Download a file from the knowledge base. 从知识库中下载文件",
    description="Download a file from the knowledge base. 从知识库中下载文件",
    response_model=bytes | list[str],
    status_code=status.HTTP_200_OK
)
async def download_file(
    file_id: str
):
    try:
        result = await get_kb_files(file_id, download=True)
        
        if result:
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found."
            )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )