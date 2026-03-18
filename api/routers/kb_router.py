import json
from loguru import logger
from fastapi import APIRouter, UploadFile, File, Form, status, HTTPException, Body
from fastapi.responses import FileResponse
from api.controllers.kb_controller import kb_controller as controller
from api.schemas.kb_schema import *
from api.schemas.base_response import *
from core.auth.shortcuts import *

router = APIRouter(prefix="/kb", tags=["Knowledge Base"])

@router.post(
    "/upload",
    summary="Upload one or multiple files to the specified knowledge base"
)
async def handle_upload_files(
    auth: UserAuth,
    files: list[UploadFile] = File(...),
    metadata: str = Form(...)
) -> SuccessResponse:
    """
    Uploads one or multiple files to the specified knowledge base.

    Args:
        files: List of uploaded files.
        metadata: Metadata of the uploaded files in JSON format. Contains the following fields:
            - app_id: int
            - domain_id: int
            - kb_id: int
            - overwrite: bool
            - batch_name: str
            - batch_id: int | None = None
            - biz_metadata: dict | None = None
            - created_by: str | None = None

    Returns:
        SuccessResponse: Success response with the following structure:
            - code: int = Field(status.HTTP_200_OK, description="Response status code")
            - message: str = Field("Success", description="Response message")
            - success: bool = Field(True, description="Request response status")
        ErrorResponse: Error response with the following structure:
            - code: int = Field(status.HTTP_400_BAD_REQUEST, description="Response status code")
            - message: str = Field("Error", description="Response message")
            - success: bool = Field(False, description="Request response status")
    """
    # Parse and validate as form model.
    metadata_dict = json.loads(metadata)
    form = KBUploadForm(files=files, **metadata_dict)
    return await controller.upload_kb_files(form)
        
    
@router.post(
    "/attach",
    summary="Attach a folder to the specified knowledge base"
)
async def attach_folder_to_kb(
    auth: UserAuth,
    kb_attach_form: KBAttachForm = Body(...)
) -> SuccessResponse:
    """
    Attaches a folder to the specified knowledge base.
    """
    return await controller.attach_folder(kb_attach_form)
    
@router.post(
    "/delete",
    summary="Delete files, all files and the knowledge base, or a batch from the specified knowledge base"
)
async def handle_delete_files(
    auth: UserAuth,
    form: KBDeleteForm = Body(...)
) -> SuccessResponse:
    """
    Deletes files, all files and the knowledge base, or a batch from the specified knowledge base.

    Args:
        form: Deletion file metadata in JSON format. Contains the following fields:
            - app_id: int
            - domain_id: int
            - kb_id: int
            - batch_id: int | None = None
            - file_id: str | None = None
            - delete_batch: bool = False
            - delete_kb: bool = False

    Returns:
        SuccessResponse: Success response with the following structure:
            - code: int = Field(status.HTTP_200_OK, description="Response status code")
            - message: str = Field("Success", description="Response message")
            - success: bool = Field(True, description="Request response status")
        ErrorResponse: Error response with the following structure:
            - code: int = Field(status.HTTP_400_BAD_REQUEST, description="Response status code")
            - message: str = Field("Error", description="Response message")
            - success: bool = Field(False, description="Request response status")
    """
    return await controller.delete_kb_files(form)
    
@router.get(
    "/download",
    summary="Download a file from the knowledge base",
    response_model=None,
    status_code=status.HTTP_200_OK
)
async def handle_download_file(
    auth: UserAuth,
    file_id: str
) -> FileResponse:
    """
    Downloads a file from the knowledge base.

    Args:
        file_id: File ID.

    Returns:
        FileResponse: File response.
        ErrorResponse: Error response with the following structure:
            - code: int = Field(status.HTTP_400_BAD_REQUEST, description="Response status code")
            - message: str = Field("Error", description="Response message")
            - success: bool = Field(False, description="Request response status")
    """
    return await controller.get_kb_file(file_id)

@router.post(
    "/file/reparse",
    summary="Reparse files",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK
)
async def handle_reparse_files(
    auth: UserAuth,
    form: KBReparseForm = Body(...)
) -> SuccessResponse:
    """
    Reparses files.

    Args:
        form: Reparse file metadata with the following fields:
            - kb_id: int
            - file_ids: list[str]

    Returns:
        SuccessResponse: Success response with the following structure:
            - code: int = Field(status.HTTP_200_OK, description="Response status code")
            - message: str = Field("Success", description="Response message")
            - success: bool = Field(True, description="Request response status")
        ErrorResponse: Error response with the following structure:
            - code: int = Field(status.HTTP_400_BAD_REQUEST, description="Response status code")
            - message: str = Field("Error", description="Response message")
            - success: bool = Field(False, description="Request response status")
    """
    return await controller.reparse_kb_files(form)

    
@router.post(
    "/file/chunk",
    summary="Modify or delete chunk content of a knowledge base file",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK
)
async def handle_edit_file_chunk(
    auth: UserAuth,
    form: KBFileChunkEditForm
) -> SuccessResponse:
    """
    Modifies or deletes the chunk content of a knowledge base file.

    Args:
        form: File chunk edit metadata with the following fields:
            - kb_id: int = Field(..., description="Knowledge base ID")
            - file_id: str = Field(..., description="File ID")
            - embed_id: str = Field(..., description="Chunk ID")
            - new_chunk: str | None = Field(None, description="New chunk content")
            - action: str = Field(..., description="Operation type")

    Returns:
        SuccessResponse: Success response with the following structure:
            - code: int = Field(status.HTTP_200_OK, description="Response status code")
            - message: str = Field("Success", description="Response message")
            - success: bool = Field(True, description="Request response status")
        ErrorResponse: Error response with the following structure:
            - code: int = Field(status.HTTP_400_BAD_REQUEST, description="Response status code")
            - message: str = Field("Error", description="Response message")
            - success: bool = Field(False, description="Request response status")
    """
    if form.action == "update":
        if form.new_chunk is None or form.new_chunk.strip() == "":
            msg = "New chunk content is required for the update operation."
            logger.error(msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=msg
            )
        return await controller.edit_kb_file_chunk(
            kb_id=form.kb_id,
            file_id=form.file_id,
            embed_id=form.embed_id,
            new_chunk=form.new_chunk
        )
    elif form.action == "delete":
        return await controller.delete_kb_file_chunk(
            kb_id=form.kb_id,
            file_id=form.file_id,
            embed_id=form.embed_id
        )
        
    elif form.action in ["enable", "disable"]:
        return await controller.toggle_kb_file_chunk_status(
            kb_id=form.kb_id,
            chunk_id=form.embed_id,
            is_active=True if form.action == "enable" else False
        )
    else:
        msg = "Invalid action type. Only 'update', 'delete', 'enable', and 'disable' are supported."
        logger.error(msg)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg
        )
    
@router.get(
    "/file/get_chunks",
    summary="Retrieve chunk content of a file by file ID",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK
)
async def handle_get_file_chunks(
    auth: UserAuth,
    kb_id: int,
    file_id: str
) -> SuccessResponse:
    """
    Retrieves chunk content of a file by file ID.

    Args:
        kb_id: int = Field(..., description="Knowledge base ID")
        file_id: str = Field(..., description="File ID")

    Returns:
        SuccessQueryResponse: Success query response with the following structure:
            - code: int = Field(status.HTTP_200_OK, description="Response status code")
            - message: str = Field("Success", description="Response message")
            - success: bool = Field(True, description="Request response status")
            - data: dict | list[dict] = Field(..., description="Response data")
        - data: Model parameters with the following structure:
            {
                embed_id: str = Field(..., description="Chunk ID")
                kb_id: int = Field(..., description="Knowledge base ID")
                file_id: str = Field(..., description="File ID")
                chunk_doc: str = Field(..., description="Chunk content")
                chunk_metadata: str = Field(..., description="Chunk metadata")
                biz_metadata: str = Field(..., description="Business metadata")
                embedding = [],  # Embedding is not returned to prevent excessive interface data.
                security_level: int = Field(..., description="Security level")
                status: int = Field(..., description="Status")
            }
        ErrorResponse: Error response with the following structure:
            - code: int = Field(status.HTTP_400_BAD_REQUEST, description="Response status code")
            - message: str = Field("Error", description="Response message")
            - success: bool = Field(False, description="Request response status")
    """
    return await controller.get_kb_file_chunk_by_id(file_id=file_id)
    
@router.post(
    "/file/chunk/update_description",
    summary="Update the description of a knowledge base file chunk",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK
)
async def handle_update_chunk_description(
    auth: UserAuth,
    form: KBFileChunkUpdateDescriptionForm
) -> SuccessResponse:
    """
    Updates the description of a knowledge base file chunk.

    Args:
        kb_id: int = Field(..., description="Knowledge base ID")
        embed_id: str = Field(..., description="Chunk ID")
        description: str = Field(..., description="Chunk description")

    Returns:
        SuccessResponse: Success response with the following structure:
            - code: int = Field(status.HTTP_200_OK, description="Response status code")
            - message: str = Field("Success", description="Response message")
            - success: bool = Field(True, description="Request response status")
        ErrorResponse: Error response with the following structure:
            - code: int = Field(status.HTTP_400_BAD_REQUEST, description="Response status code")
            - message: str = Field("Error", description="Response message")
            - success: bool = Field(False, description="Request response status")
    """
    return await controller.update_kb_file_chunk_description(
            kb_id=form.kb_id,
            embed_id=form.embed_id,
            description=form.description
        )

@router.post(
    "/file/chunk/update_tags",
    summary="Update tags of a knowledge base file",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK
)
async def handle_update_chunk_tags(
    auth: UserAuth,
    form: KBFileChunkUpdateTagsForm
) -> SuccessResponse:
    """
    Updates the tags of a knowledge base file.

    Args:
        kb_id: int = Field(..., description="Knowledge base ID")
        file_id: str = Field(..., description="File ID")
        tags: list[str] = Field(..., description="File chunk tags")

    Returns:
        SuccessResponse: Success response with the following structure:
            - code: int = Field(status.HTTP_200_OK, description="Response status code")
            - message: str = Field("Success", description="Response message")
            - success: bool = Field(True, description="Request response status")
        ErrorResponse: Error response with the following structure:
            - code: int = Field(status.HTTP_400_BAD_REQUEST, description="Response status code")
            - message: str = Field("Error", description="Response message")
            - success: bool = Field(False, description="Request response status")
    """
    return await controller.update_file_tags(
            kb_id=form.kb_id,
            file_id=form.file_id,
            tags=form.tags
        )


@router.post(
    "/preview/extracted-img", 
    response_class=FileResponse, 
    status_code=status.HTTP_200_OK, 
    summary="Preview extracted image from file"
)
async def handle_preview_extracted_image(auth: UserAuth, params: PreviewImageParams):
    """
    Preview a specific image extracted from a document within the knowledge base.
    
    Access: **User**

    Args:
        params (PreviewImageParams): Request parameters including:
            - **kb_id** (int): Knowledge Base ID.
            - **file_id** (str): The unique ID of the source file.
            - **image_name** (str): The specific filename of the extracted image.

    Returns:
        FileResponse: The binary image file with cache headers.

    Raises:
        HTTPException (404): If the source file or the specific image cannot be found.
    """
    return await controller.preview_extracted_image(params)