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
    ### Description
    Uploads one or multiple files to a specific knowledge base and associates them with provided metadata.

    ---
    ### Parameters (Metadata JSON fields)
    - **app_id** (`int`): Unique identifier of the application.
    - **domain_id** (`int`): Domain identifier.
    - **kb_id** (`int`): Target Knowledge Base ID.
    - **overwrite** (`bool`): Whether to overwrite existing files with the same name.
    - **batch_name** (`str`): Name for this upload batch.
    - **biz_metadata** (`dict`, optional): Custom business-level metadata.

    > **Note:** The `metadata` field must be passed as a JSON string within the Form data.
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
    ### Description
    Attaches an existing local folder or directory path to the specified knowledge base.
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
    ### Description
    Deletes specific files, an entire batch, or the whole knowledge base content.

    ---
    ### Deletion Modes
    - **Single File:** Provide the `file_id`.
    - **Batch Delete:** Set `delete_batch=True` and provide the `batch_id`.
    - **Clear Knowledge Base:** Set `delete_kb=True`.

    ### Requirements
    - `app_id`, `domain_id`, and `kb_id` are mandatory for locating the resources.
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
    ### Description
    Retrieves the original binary file stream from the knowledge base.

    - **file_id**: The unique identifier of the file to download.
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
    ### Description
    Triggers the parsing pipeline again for specific files. Useful if parsing configurations or models have been updated.

    - **kb_id**: Knowledge Base ID.
    - **file_ids**: A list of unique file identifiers.
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
    ### Description
    Provides granular management of document segments (chunks).

    ---
    ### Supported Actions (`action`)
    1. `update`: Updates the text content. Requires `new_chunk`.
    2. `delete`: Permanently removes the vector chunk from the index.
    3. `enable`: Activates the chunk for retrieval.
    4. `disable`: Deactivates the chunk (hides it from search results).

    > **Warning:** Ensure the `embed_id` belongs to the specified `kb_id`.
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
    ### Description
    Retrieves detailed information for all chunks associated with a specific file.

    ### Response Data
    Returns a list of objects containing:
    - `chunk_doc`: The raw text content.
    - `chunk_metadata`: Technical metadata generated during parsing.
    - `status`: Current status of the chunk.
    - `embedding`: **Note:** The vector array is omitted by default to optimize performance.
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
    ### Description
    Manually updates the description or summary of a specific chunk.
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
    ### Description
    Updates the list of tags associated with a knowledge base file.
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
    ### Description
    Displays a specific image that was extracted from a PDF/Word document during the parsing process.

    ### Access Level
    - **User** or higher.

    ### Error Handling
    Returns **404 Not Found** if the image path is invalid or the file has been purged.
    """
    return await controller.preview_extracted_image(params)