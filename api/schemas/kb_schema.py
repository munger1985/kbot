from pydantic import BaseModel, Field
from fastapi import UploadFile


class KBUploadForm(BaseModel):
    """Knowledge base file upload form model.
    
    This model defines the data structure for uploading files to a knowledge base,
    including file list and associated business metadata.
    """
    files: list[UploadFile] = Field(..., description="List of files to upload (FastAPI UploadFile objects)")
    app_id: int = Field(..., description="Application ID (unique identifier of the associated application)")
    domain_id: int = Field(..., description="Domain ID (unique identifier of the business domain)")
    kb_id: int = Field(..., description="Knowledge base ID (unique identifier of the target knowledge base)")
    overwrite: bool = Field(..., description="Whether to overwrite existing files with the same name")
    skip_approval: bool = Field(..., description="Whether to skip the approval process for upload")
    batch_name: str = Field(..., description="Batch name (human-readable label for the upload batch)")
    batch_id: int|None = Field(None, description="Batch ID (unique identifier of the upload batch, optional)")
    biz_metadata: dict|None = Field(None, description="Business metadata (custom key-value pairs for business logic)")
    created_by: str|None = Field(None, description="Creator (username of the user who initiated the upload)")

class KBAttachForm(BaseModel):
    """Knowledge base folder attachment form model.
    
    This model defines the data structure for attaching a local folder to a knowledge base.
    """
    folder_path: str = Field(..., description="Folder path (absolute/relative path to the target folder)")
    app_id: int = Field(..., description="Application ID (unique identifier of the associated application)")
    domain_id: int = Field(..., description="Domain ID (unique identifier of the business domain)")
    kb_id: int = Field(..., description="Knowledge base ID (unique identifier of the target knowledge base)")
    batch_name: str = Field(..., description="Batch name (human-readable label for the attachment batch)")
    biz_metadata: dict|None = Field(None, description="Business metadata (custom key-value pairs for business logic)")
    created_by: str|None = Field(None, description="Creator (username of the user who initiated the attachment)")

class KBDeleteForm(BaseModel):
    """Knowledge base deletion form model.
    
    This model defines the data structure for deleting files/batches from a knowledge base.
    """
    app_id: int = Field(..., description="Application ID (unique identifier of the associated application)")
    domain_id: int = Field(..., description="Domain ID (unique identifier of the business domain)")
    kb_id: int = Field(..., description="Knowledge base ID (unique identifier of the target knowledge base)")
    batch_id: int|None = Field(None, description="Batch ID (delete entire batch if specified, optional)")
    batch_name: str|None = Field(None, description="Batch name (alternative to batch ID for batch deletion, optional)")
    file_ids: list[str]|None = Field(None, description="List of file IDs (specific files to delete, optional)")
    file_paths: list[str]|None = Field(None, description="List of file paths (specific files to delete by path, optional)")

class KBReparseForm(BaseModel):
    """Knowledge base reparse form model.
    
    This model defines the data structure for re-parsing specific files in a knowledge base.
    """
    kb_id: int = Field(..., description="Knowledge base ID (unique identifier of the target knowledge base)")
    file_ids: list[str] = Field(..., description="List of file IDs (files to re-parse)")

class KBFilePreviewForm(BaseModel):
    """Knowledge base file preview form model.
    
    This model defines the data structure for previewing content of files in a knowledge base,
    supporting different file types (PDF, Excel, PowerPoint, etc.).
    """
    file_id: str = Field(..., description="File ID (unique identifier of the target file)")
    max_length: int|None = Field(None, description="Maximum content length to preview (optional)")
    pages: int|list[int] | None = Field(None, description="Pages to preview (single page number or list of pages, optional)")
    sheet_index: int|None = Field(None, description="Sheet index (for Excel files, optional)")
    preview_rows: int|None = Field(None, description="Number of rows to preview (for Excel/CSV files, optional)")
    slide: int|None = Field(None, description="Slide number (for PowerPoint files, optional)")

class KBFileChunkEditForm(BaseModel):
    """Knowledge base file chunk edit form model.
    
    This model defines the data structure for editing individual chunks (embeddings) of a file in a knowledge base.
    """
    kb_id: int = Field(..., description="Knowledge base ID (unique identifier of the target knowledge base)")
    file_id: str = Field(..., description="File ID (unique identifier of the target file)")
    chunk_id: str = Field(..., description="Chunk ID (unique identifier of the target embedding chunk)")
    new_chunk: str|None = Field(None, description="New chunk content (for update operations, optional)")
    action: str = Field(..., description="Action type (e.g., 'update', 'delete', 'archive')")
    
class KBFileChunkUpdateDescriptionForm(BaseModel):
    """Knowledge base file chunk update description form model.
    
    This model defines the data structure for updating the description of a file chunk in a knowledge base.
    """
    kb_id: int = Field(..., description="Knowledge base ID (unique identifier of the target knowledge base)")
    chunk_id: str = Field(..., description="Chunk ID (unique identifier of the target embedding chunk)")
    description: str = Field(..., description="New description for the chunk")

class KBFileChunkUpdateTagsForm(BaseModel):
    """Knowledge base file chunk update tags form model.
    
    This model defines the data structure for updating tags of file chunks in a knowledge base.
    """
    kb_id: int = Field(..., description="Knowledge base ID (unique identifier of the target knowledge base)")
    file_id: str = Field(..., description="File ID (unique identifier of the target file)")
    tags: list[str] = Field(..., description="Tags for the file chunks (list of string labels)")

class PreviewImageParams(BaseModel):
    """Preview extracted image parameters model.
    
    This model defines the data structure for previewing images extracted from knowledge base files.
    """
    file_id: str = Field(..., description="File ID (unique identifier of the source file)")
    image_name: str = Field(..., description="Image name (name of the extracted image to preview)")