from loguru import logger
import os
from pathlib import Path
from fastapi import HTTPException
import urllib.parse
from pathlib import Path
from fastapi.responses import FileResponse
from services.kb.chunk_service import ChunkService
from services.kb.file_service import FileService
# from services.kb.kb_file_preview import FilePreview
from api.schemas.kb_schema import *
from api.schemas.base_response import SuccessResponse
from core.config.settings import get_app_config
# from utils.file_converter import FileToImage
from utils.sanitize import sanitize_text_for_oracle_json


class KBController:
    """Knowledge Base Controller"""
    def __init__(self):
        self.file_service = FileService()
        self.chunk_service = ChunkService()
        # self.file_preview = FilePreview()
    
    async def upload_kb_files(
            self,
            upload_form: KBUploadForm
        ) -> SuccessResponse:
        """Upload files to knowledge base"""
        logger.debug(f"batch_id: {upload_form.batch_id}")

        await self.file_service.upload_file_service(
            files=upload_form.files,
            app_id=upload_form.app_id,
            domain_id=upload_form.domain_id,
            kb_id=upload_form.kb_id,
            overwrite=upload_form.overwrite,
            skip_approval=upload_form.skip_approval,
            batch=upload_form.batch,
            biz_metadata=upload_form.biz_metadata,
            created_by=upload_form.created_by
        )
        return SuccessResponse(message=f"Successfully uploaded files to knowledge base {upload_form.kb_id}")
        
    async def attach_folder(
            self,
            attach_form: KBAttachForm
        ) -> SuccessResponse:
        """Attach existing folder to knowledge base"""
        await self.file_service.attach_folder(
            folder_path=attach_form.folder_path,
            app_id=attach_form.app_id,
            domain_id=attach_form.domain_id,
            kb_id=attach_form.kb_id,
            batch=attach_form.batch,
            skip_approval=attach_form.skip_approval,
            biz_metadata=attach_form.biz_metadata,
            created_by=attach_form.created_by
        )
        return SuccessResponse(message=f"Successfully attached folder to knowledge base {attach_form.kb_id}")

    async def get_kb_file(self, file_id: str) -> FileResponse:
        """Get knowledge base file"""
        file_path = await self.file_service.get_path_by_id(file_id=file_id)
        return FileResponse(
                path=file_path,
                filename=urllib.parse.quote(Path(file_path).name, encoding='utf-8'),
                media_type="multipart/form-data",
                headers={
                    "Content-Disposition": "attachment; filename*=UTF-8''{}".format(urllib.parse.quote(Path(file_path).name, encoding='utf-8'))
                },
                content_disposition_type=None # type: ignore
            )

    async def delete_kb_files(
            self,
            form: KBDeleteForm
        ) -> SuccessResponse:
        """Delete files from knowledge base"""
        await self.file_service.delete_file_service(
                kb_id=form.kb_id,
                batch=form.batch,
                file_ids=form.file_ids,
            )
        return SuccessResponse(message=f"Successfully deleted files {form.file_ids}")
        
    async def reparse_kb_files(self, form: KBReparseForm) -> SuccessResponse:
        """Reparse files in knowledge base"""
        await self.file_service.reparse_file(kb_id=form.kb_id, file_ids=form.file_ids)
        return SuccessResponse(message=f"Successfully triggered reparse for files {form.file_ids}")

    async def edit_kb_file_chunk(
            self,
            kb_id: int,
            file_id: str,
            chunk_id: str,
            new_chunk: str,
        ) -> SuccessResponse:
        """Edit chunk content of knowledge base file and update chunk embedding vector"""
        await self.chunk_service.edit_file_chunk(
            kb_id=kb_id,
            file_id=file_id,
            chunk_id=chunk_id,
            new_chunk=new_chunk
        )
        return SuccessResponse(message=f"Successfully edited chunk {chunk_id} of file {file_id}")
        
    async def delete_kb_file_chunk(
            self,
            kb_id: int,
            file_id: str,
            chunk_id: str,
        ) -> SuccessResponse:
        """Delete chunk content of knowledge base file and update chunk embedding vector"""
        await self.chunk_service.delete_file_chunk(
                kb_id=kb_id,
                file_id=file_id,
                chunk_id=chunk_id
            )
        return SuccessResponse(message=f"Successfully deleted chunk {chunk_id} of file {file_id}")
        
    async def toggle_kb_file_chunk_status(
            self,
            kb_id: int,
            chunk_id: str,
            is_active: bool
        ) -> SuccessResponse:
        """Toggle chunk status of knowledge base file"""
        await self.chunk_service.toggle_chunk_active_status(
                chunk_id=chunk_id,
                is_active=is_active
            )
        return SuccessResponse(message=f"Successfully toggled status of chunk {chunk_id}")
        
    async def get_kb_file_chunk_by_id(
            self,
            file_id: str
        ) -> SuccessResponse:
        """Get chunks of knowledge base file"""
        result = await self.chunk_service.get_chunks_by_file_id(file_id=file_id)
        return SuccessResponse(data=result, message=f"Successfully retrieved chunks of file {file_id}")
        
    async def update_kb_file_chunk_description(
            self,
            kb_id: int,
            chunk_id: str,
            description: str
        ) -> SuccessResponse:
        """Update chunk description of knowledge base file"""
        # 1. Sanitize special characters in description for Oracle JSON compatibility
        description = sanitize_text_for_oracle_json(description, max_length=4000)
        await self.chunk_service.update_chunk_description(
                kb_id=kb_id,
                chunk_id=chunk_id,
                description=description
            )
        return SuccessResponse(message=f"Successfully updated description of chunk {chunk_id}")
        
    async def update_file_tags(
            self,
            kb_id: int,
            file_id: str,
            tags: list[str]
        ) -> SuccessResponse:
        """Update tags of knowledge base file, auto-sync tags to associated file chunks"""
        await self.file_service.update_file_tag(
            file_id=file_id,
            kb_id=kb_id,
            tags=tags
        )
        return SuccessResponse(message=f"Successfully updated tags of file {file_id}")
        

    async def preview_extracted_image(self, params: PreviewImageParams) -> FileResponse:
        """Preview images extracted from a knowledge base file"""
        # 1. Get the base file path to locate the associated image directory
        file_path = await self.file_service.get_path_by_id(file_id=params.file_id)
        
        # 2. Construct the path to the extracted images folder
        # Images are typically stored in a subfolder named after the file_id
        dir_name = os.path.dirname(file_path)
        image_dir = Path(dir_name) / params.file_id
        img_path = image_dir / params.image_name
        
        # 3. Verify existence and return the file
        if not img_path.exists():
            logger.error(f"Image not found at path: {img_path}")
            raise HTTPException(status_code=404, detail="Requested image does not exist")
            
        logger.info(f"Serving extracted image: {params.image_name} for file: {params.file_id}")
        return FileResponse(
            path=img_path, 
            headers={"Cache-Control": "max-age=3600"}
        )
    
    async def get_pdf_by_id(self, file_id: str) -> FileResponse:
        """Get PDF file by file_id"""
        file_path = await self.file_service.get_path_by_id(file_id)
        return FileResponse(file_path, media_type="application/pdf")
    
kb_controller = KBController()