import os
import uuid
import json
import shutil
import re
from pathlib import Path
from typing import Any
from fastapi import UploadFile
from loguru import logger
from core.config.settings import get_app_config, get_prompt_config
from dao.entities import BatchEntity, FileEntity
from core.dictionary import FileStatus, YesNoEnum
from dao.repositories import (KBRepository, FileRepository, BatchRepository,
                              TxtChunkRepository, PromptRepository)
from utils.common import run_in_thread_pool
from services.ai_model import AIModelService
from core.database.oracle import get_session
from core.exceptions import *

class FileService:
    """Knowledge Base File Operation Service.
    
    Manages file upload/download, metadata persistence, and physical file deletion
    for knowledge base systems. Supports batch operations, version control, and
    multi-threaded file saving for performance optimization.
    """
    
    def __init__(self) -> None:
        """Initialize KB file operation service with storage configuration."""
        config = get_app_config()
        self.file_storage = config.file_storage
        self.upload_workers = config.upload_workers
        self.model_service = AIModelService()

    @property
    def oracle_session(self):
        """Get async Oracle database session context manager.
        
        Returns:
            AsyncContextManager[AsyncSession]: Async database session manager
        """
        return get_session()

    def save_file(self, file: UploadFile, domain_id: int, kb_id: int, batch_name: str, overwrite: bool) -> dict[str, Any]:
        """Save single file to disk with version control and return file metadata.
        
        Handles filename conflicts by appending version numbers or overwriting existing files
        based on the overwrite flag.
        
        Args:
            file: Uploaded file object from FastAPI
            domain_id: Business domain ID
            kb_id: Target knowledge base ID
            batch_name: Batch name for this upload
            overwrite: Whether to overwrite existing files (True/False)
            
        Returns:
            dict[str, Any]: File metadata dict with:
                - file_path: Absolute file path (str)
                - file_name: Final saved filename (str)
                - file_ext: File extension (str)
                - is_overwrite: 1=overwritten, 0=not overwritten (int)
                - file_version: File version number (int)
                - file_size: File size in bytes (int)
                
        Raises:
            ValueError: If filename is empty
            OSError: If file write operation fails
        """
        filename = file.filename
        if not filename:
            raise ValueError("Filename cannot be empty")
        
        try:
            logger.debug(f"Starting file save: {filename} to KB {kb_id}")
            file_content = file.file.read()

            # Build target path structure: storage/domain_id/kb_id/source/batch_name
            root_path = Path(self.file_storage).resolve()
            target_path = root_path / str(domain_id) / str(kb_id) / "source" / batch_name
            target_path.mkdir(parents=True, exist_ok=True)
            file_path = target_path / filename

            # Extract base file parameters
            name, ext = os.path.splitext(filename)
            file_params = {
                "file_path": str(file_path),
                "file_name": filename,
                "file_ext": ext,
                "is_overwrite": YesNoEnum.YES.value if overwrite else YesNoEnum.NO.value,
                "file_version": 1,
                "file_size": len(file_content)
            }

            # Handle filename conflicts with versioning
            if file_path.exists():
                logger.debug(f"File {filename} already exists in KB {kb_id}")
                counter = 1
                new_filename = ""
                
                if overwrite:
                    logger.debug(f"Overwrite enabled - will replace {filename}")
                    # Find next version number (for tracking)
                    new_path = file_path
                    while new_path.exists():
                        new_filename = f"{name}({counter}){ext}"
                        new_path = target_path / new_filename
                        counter += 1
                    file_params["file_version"] = counter
                else:
                    logger.debug(f"Overwrite disabled - generating new filename")
                    # Generate unique filename with version suffix
                    while file_path.exists():
                        new_filename = f"{name}({counter}){ext}"
                        file_path = target_path / new_filename
                        counter += 1
                    file_params["file_name"] = new_filename
                    file_params["file_path"] = str(file_path)
                    file_params["file_version"] = counter

            # Write file content to disk
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            logger.info(f"File saved successfully: {filename} -> {file_path}")
            return file_params

        except Exception as e:
            error_filename = filename if 'filename' in locals() else "unknown file"
            logger.error(f"Failed to save file {error_filename}: {str(e)}")
            raise InternalServerError(message=f"Failed to save file {error_filename}: {str(e)}")

    async def _save_files_in_thread(self, 
                                   files: list[UploadFile],
                                   domain_id: int,
                                   kb_id: int,
                                   batch_name: str,
                                   overwrite: bool) -> list[dict[str, Any]]:
        """Save multiple files to disk using multi-threading for performance.
        
        Args:
            files: list of UploadFile objects
            domain_id: Business domain ID
            kb_id: Target knowledge base ID
            batch_name: Batch name for this upload
            overwrite: Whether to overwrite existing files (True/False)
            
        Returns:
            list[dict[str, Any]]: list of file metadata dicts (same structure as save_file return)
        """
        file_params = [
            {
                "file": file, 
                "domain_id": domain_id, 
                "kb_id": kb_id,
                "batch_name": batch_name, 
                "overwrite": overwrite
            } 
            for file in files
        ]
        
        # Run file save operations in thread pool
        results = [
            result async for result in run_in_thread_pool(
                func=self.save_file, 
                params=file_params, 
                workers=self.upload_workers
            )
        ]

        logger.debug(f"File save results: {len(results)} files processed")
        return results

    async def _save_file_metadata(self, 
                                 fileparams: list[dict[str, Any]], 
                                 app_id: int,
                                 kb_id: int,
                                 batch_name: str,
                                 skip_approval: bool,
                                 batch_id: int | None = None,
                                 biz_metadata: dict[str, Any] | None = None,
                                 created_by: str | None = None):
        """Persist file and batch metadata to Oracle database.
        
        Args:
            fileparams: list of file metadata dicts from save_file
            app_id: Application ID
            kb_id: Target knowledge base ID
            batch_name: Batch name for this upload
            skip_approval: Whether to skip approval procedure(True/False)
            batch_id: Optional existing batch ID
            biz_metadata: Optional business metadata (JSON-serializable)
            created_by: Optional creator identifier

        """
        async with self.oracle_session as session:
            # Validate knowledge base exists
            kb_repo = KBRepository(session)
            kb_entity = await kb_repo.get_by_id(kb_id)
            
            if not kb_entity:
                error_msg = f"Knowledge base {kb_id} does not exist"
                logger.error(error_msg)
                raise NotFoundError(message=error_msg)

            # Get default VLM prompt configuration
            prompt_repo = PromptRepository(session)
            vlm_prompt_unique_name = get_prompt_config().image2text
            try:
                prompt = await prompt_repo.get_prompt_by_unique_name(vlm_prompt_unique_name)
            except DataNotFoundException as e:
                logger.warning(f"Prompt not found: {e}, will use default prompt")
                prompt = None
            except Exception as e:
                logger.warning(f"Failed to get prompt: {e}, will use default prompt")

            # Clean up prompt text: remove leading/trailing whitespace and normalize newlines
            if prompt:
                prompt = prompt.strip().replace('\n', ' ')
                # Remove multiple consecutive spaces
                prompt = re.sub(r'\s+', ' ', prompt)

            if not prompt:
                prompt = "Please describe the image in detail."
                
            vlm_model_id = kb_entity.img2txt_model_id
            llm_model_id = kb_entity.llm_model_id
            txt_embed_model_id = kb_entity.txt_embed_model_id
            # get model display name
            if vlm_model_id:
                vlm_model = await self.model_service.get_display_name_by_id(vlm_model_id)
            else:
                vlm_model = None
            
            if txt_embed_model_id:
                txt_embed_model = await self.model_service.get_display_name_by_id(txt_embed_model_id)
            else:
                txt_embed_model = None
            
            llm_model = await self.model_service.get_display_name_by_id(llm_model_id)

            # Build chunk parser configuration (JSON-serialized)
            # Create chunk parser configuration as dict
            chunk_parser_config = {
                "do_ocr": True,
                "overlap": 50,
                "use_vlm": bool(vlm_model),
                "vlm_model": vlm_model,
                "txt_embedding_model": txt_embed_model,
                "llm_model": llm_model,
                "chunk_size": 1000,
                "ocr_engine": "tesseract",
                "vlm_prompt": prompt,
                "images_scale": 2.0,
                "min_chunk_len": 200,
                "generate_picture_images": True
            }

            # Handle batch: use existing if batch_id provided, otherwise create new with batch_name
            # Note: Unique constraint is (app_id, kb_id, batch_name)
            logger.debug(f"Batch handling - batch_id: {batch_id}, batch_name: {batch_name}, app_id: {app_id}, kb_id: {kb_id}")
            
            if batch_id:
                logger.info(f"Using existing batch {batch_id} for KB {kb_id}")
            elif batch_name:
                # Create new batch with the given name
                batch_repo = BatchRepository(session)
                batch_entity = BatchEntity(
                    app_id=app_id,
                    batch_name=batch_name,
                    kb_id=kb_id,
                    created_by=created_by,
                    updated_by=created_by
                )
                try:
                    batch_id = await batch_repo.create(batch_entity)
                    logger.info(f"Created new batch {batch_id} for KB {kb_id}")
                except Exception as e:
                    error_msg = f"Failed to create batch: {str(e)}"
                    handle_exception(e, error_msg)
                    raise

            # Create file entities for batch persistence
            file_entities = []
            for fileparam in fileparams:
                file_entity = FileEntity(
                    file_id=str(uuid.uuid4()),
                    app_id=app_id,
                    kb_id=kb_id,
                    batch_id=batch_id,
                    file_path=fileparam.get("file_path"),
                    file_name=fileparam.get("file_name"),
                    file_ext=fileparam.get("file_ext"),
                    status=FileStatus.UPLOADED.value if not skip_approval else FileStatus.APPROVED.value,
                    file_version=fileparam.get("file_version"),
                    is_overwrite=fileparam.get("is_overwrite"),
                    security_level=kb_entity.security_level or 1,
                    chunk_parser=chunk_parser_config,
                    process_priority=kb_entity.process_priority,
                    file_size=fileparam.get("file_size"),
                    biz_metadata=biz_metadata,
                    created_by=created_by,
                    updated_by=created_by
                )
                file_entities.append(file_entity)
            # Persist to database
            file_repo = FileRepository(session)
            
            try:
                await file_repo.create(file_entities)
                logger.info(f"Successfully persisted {len(file_entities)} files to database (KB: {kb_id})")
            except Exception as e:
                error_msg = f"Failed to persist file metadata: {str(e)}"
                handle_exception(e, error_msg)

    async def upload_file_service(self, 
                                 files: list[UploadFile], 
                                 app_id: int,
                                 domain_id: int,
                                 kb_id: int,
                                 batch_name: str,
                                 overwrite: bool,
                                 skip_approval: bool,
                                 batch_id: int | None = None,
                                 biz_metadata: dict[str, Any] | None = None,
                                 created_by: str | None = None):
        """End-to-end file upload service (disk + database persistence).
        
        Args:
            files: list of UploadFile objects
            app_id: Application ID
            domain_id: Business domain ID
            kb_id: Target knowledge base ID
            batch_name: Batch name for this upload
            overwrite: Whether to overwrite existing files (True/False)
            skip_approval: Whether to skip approval procedure(True/False)
            batch_id: Optional existing batch ID
            biz_metadata: Optional business metadata (JSON-serializable)
            created_by: Optional creator identifier
            
        """
        logger.info(f"Starting upload of {len(files)} files to KB {kb_id}")
        # Step 1: Save files to disk
        fileparams = await self._save_files_in_thread(
            files=files,
            domain_id=domain_id,
            kb_id=kb_id,
            batch_name=batch_name,
            overwrite=overwrite
        )
        logger.debug(f"Files saved to disk: {[fp['file_name'] for fp in fileparams]}")
        # Step 2: Persist metadata to database
        logger.info(f"Persisting file metadata to database (KB: {kb_id})")
        await self._save_file_metadata(
            fileparams=fileparams,
            app_id=app_id,
            kb_id=kb_id,
            batch_name=batch_name,
            skip_approval=skip_approval,
            batch_id=batch_id,
            biz_metadata=biz_metadata,
            created_by=created_by
        )
        logger.info(f"Successfully uploaded {len(files)} files to KB {kb_id}")


    async def _delete_files(self, 
                           domain_id: int, 
                           kb_id: int | None = None, 
                           batch_name: str | None = None,
                           file_paths: list[str] | None = None):
        """Delete physical files from disk (supports multiple deletion modes).
        
        Modes:
            1. Explicit file paths: Delete specific files
            2. Batch name + KB ID: Delete entire batch directory
            3. KB ID only: Delete entire KB directory
            
        Args:
            domain_id: Business domain ID
            kb_id: Optional knowledge base ID
            batch_name: Optional batch name
            file_paths: Optional list of file paths to delete
            
        Returns:
            tuple[int, int]: (Number of successfully deleted files, Number of failed deletions)
        """
        success_cnt = 0
        failed_cnt = 0

        # Mode 1: Delete specific files by path
        if file_paths:
            for file_path in file_paths:
                logger.info(f"Deleting file: {file_path}")
                path = Path(file_path)
                
                if path.exists():
                    try:
                        os.remove(path)
                        logger.info(f"Successfully deleted file: {file_path}")
                        success_cnt += 1
                    except Exception as e:
                        logger.error(f"Failed to delete file {file_path}: {str(e)}")
                        failed_cnt += 1
                else:
                    logger.error(f"File not found: {file_path}")
                    failed_cnt += 1
            logger.debug(f"Deleted {success_cnt} files, {failed_cnt} failed")

        # Mode 2: Delete entire batch directory
        elif batch_name and kb_id:
            root_path = Path(self.file_storage).resolve()
            target_path = root_path / str(domain_id) / str(kb_id) / "source" / batch_name

            if not target_path.exists():
                logger.warning(f"Batch directory not found: {target_path} (KB: {kb_id})")
                return success_cnt, failed_cnt

            # Count files (fix: correct file counting logic)
            file_count = sum(1 for _ in target_path.rglob("*") if _.is_file())
            
            try:
                logger.info(f"Deleting batch directory: {target_path}")
                shutil.rmtree(target_path)
                logger.info(f"Deleted {file_count} files from batch {batch_name} (KB: {kb_id})")
            except Exception as e:
                logger.error(f"Failed to delete batch directory {target_path}: {str(e)}")

        # Mode 3: Delete entire KB directory
        elif kb_id:
            root_path = Path(self.file_storage).resolve()
            target_path = root_path / str(domain_id) / str(kb_id)

            if not target_path.exists():
                logger.warning(f"KB directory not found: {target_path}")
                return

            # Count files (fix: correct file counting logic)
            file_count = sum(1 for _ in target_path.rglob("*") if _.is_file())
            
            try:
                logger.info(f"Deleting entire KB directory: {target_path}")
                shutil.rmtree(target_path)
                logger.info(f"Deleted {file_count} files from KB {kb_id}")
            except Exception as e:
                logger.error(f"Failed to delete KB directory {target_path}: {str(e)}")

        # Invalid parameters
        else:
            msg = "Invalid deletion parameters - must provide file_paths, (batch_name + kb_id), or kb_id"
            logger.error(msg)
            raise ParamValueError(msg)

    async def _delete_metadata(self, 
                              kb_id: int | None = None, 
                              batch_id: int | None = None, 
                              file_ids: list[str] | None = None):
        """Delete file metadata from database (supports multiple deletion modes).
        
        Modes:
            1. KB ID only: Delete all files/chunks for the KB
            2. Batch ID only: Delete all files/chunks for the batch
            3. File IDs only: Delete specific files/chunks
            
        Args:
            kb_id: Optional knowledge base ID
            batch_id: Optional batch ID
            file_ids: Optional list of file IDs
            
        Raises:
            ParamValueError: If no valid deletion parameters provided
            Exception: If database deletion fails (handled by handle_exception)
        """
        async with self.oracle_session as session:
            file_repo = FileRepository(session)
            chunk_repo = TxtChunkRepository(session)

            # Mode 1: Delete entire KB
            if kb_id and not batch_id and not file_ids:
                try:
                    await file_repo.delete(kb_id=kb_id, batch_id=None, file_ids=None)
                    await chunk_repo.delete_by_kb_id(kb_id)
                    logger.info(f"Deleted all metadata for KB {kb_id}")
                except Exception as e:
                    error_msg = f"Failed to delete metadata for KB {kb_id}: {str(e)}"
                    handle_exception(e, error_msg)

            # Mode 2: Delete entire batch
            elif batch_id:
                try:
                    files = await file_repo.get_by_batch_id(batch_id)
                    file_ids = [file.file_id for file in files]
                    await file_repo.delete(kb_id=None, batch_id=batch_id, file_ids=None)
                    await chunk_repo.delete_by_file_ids(file_ids)
                    logger.info(f"Deleted all metadata for batch {batch_id} ({len(file_ids)} files)")
                except Exception as e:
                    error_msg = f"Failed to delete metadata for batch {batch_id}: {str(e)}"
                    handle_exception(e, error_msg)

            # Mode 3: Delete specific files
            elif file_ids:
                try:
                    await file_repo.delete(kb_id=None, batch_id=None, file_ids=file_ids)
                    await chunk_repo.delete_by_file_ids(file_ids)
                    logger.info(f"Deleted metadata for {len(file_ids)} files")
                except Exception as e:
                    error_msg = f"Failed to delete metadata for files {file_ids}: {str(e)}"
                    handle_exception(e, error_msg)

            # Invalid parameters
            else:
                error_msg = "Invalid deletion parameters - must provide kb_id, batch_id, or file_ids"
                logger.error(error_msg)
                raise ParamValueError(error_msg)

    async def delete_file_service(
            self,
            app_id: int,
            domain_id: int,
            kb_id: int, 
            batch_id: int | None = None, 
            batch_name: str | None = None,
            file_ids: list[str] | None = None,
            file_paths: list[str] | None = None):
        """Unified file deletion service (metadata + physical file deletion).
        
        Args:
            app_id: Application ID
            domain_id: Business domain ID
            kb_id: Knowledge base ID
            batch_id: Optional batch ID (for batch deletion)
            batch_name: Optional batch name (for physical file deletion)
            file_ids: Optional list of file IDs (for specific file deletion)
            file_paths: Optional list of file paths (for physical file deletion)
            
        Returns:
            dict[str, Any]: Deletion result with:
                - success_count: Number of successfully deleted files
                - failed_count: Number of failed deletions
                - status: "success" or "partial_failure"
                - message: Result description
        """
        # Mode 1: Delete specific files
        if file_paths and file_ids:
            logger.info(f"Deleting {len(file_paths)} specific files (KB: {kb_id})")
            await self._delete_metadata(kb_id=None, batch_id=None, file_ids=file_ids)
            await self._delete_files(domain_id, None, None, file_paths)

        # Mode 2: Delete entire batch
        elif batch_name and batch_id:
            logger.info(f"Deleting batch {batch_name} (ID: {batch_id}) from KB {kb_id}")
            await self._delete_metadata(kb_id=None, batch_id=batch_id, file_ids=None)
            await self._delete_files(domain_id, kb_id, batch_name, None)

        # Mode 3: Delete entire KB
        elif kb_id and not batch_id and not file_ids:
            logger.info(f"Deleting all files from KB {kb_id}")
            await self._delete_metadata(kb_id=kb_id, batch_id=None, file_ids=None)
            await self._delete_files(domain_id, kb_id, None, None)

        else:
            error_msg = "Invalid deletion parameters - must provide (file_paths+file_ids), (batch_name+batch_id), or kb_id only"
            logger.error(error_msg)
            raise ParamValueError(error_msg)


    async def update_file_tags(self, file_id: str, tags: list[str]):
        """Update tags for a specific knowledge base file.
        
        Args:
            file_id: Target file ID
            tags: list of new tags to apply
            
        Raises:
            Exception: If tag update fails (handled by handle_exception)
        """
        async with self.oracle_session as session:
            file_repo = FileRepository(session)
            chunk_repo = TxtChunkRepository(session)
            try:
                # Update tags for all chunks of the file
                await chunk_repo.update_tag(file_id=file_id, tags=tags)
                logger.info(f"Successfully updated tags for all chunks of file {file_id}: {tags}")
                # Update tags for file
                await file_repo.update_tags(file_id=file_id, tags=tags)
                logger.info(f"Updated tags for file {file_id}: {tags}")

            except Exception as e:
                error_msg = f"Failed to update tags for file {file_id}: {str(e)}"
                handle_exception(e, error_msg)

    async def attach_folder(self, 
                           folder_path: str,
                           app_id: int,
                           domain_id: int,
                           kb_id: int,
                           batch_name: str,
                           biz_metadata: dict[str, Any] | None = None,
                           created_by: str | None = None):
        """Attach existing folder to knowledge base (metadata only, no file copy).
        
        Scans a local folder and persists file metadata to the database without
        copying files - useful for linking pre-existing file directories.
        
        Args:
            folder_path: Absolute path to existing folder
            app_id: Application ID
            domain_id: Business domain ID
            kb_id: Target knowledge base ID
            batch_name: Batch name for this attachment
            biz_metadata: Optional business metadata (JSON-serializable)
            created_by: Optional creator identifier
            
        Returns:
            None
        """
        try:
            root_folder = Path(folder_path).resolve()
            
            # Validate folder exists and is a directory
            if not root_folder.exists() or not root_folder.is_dir():
                error_msg = f"Folder path is invalid or does not exist: {folder_path}"
                logger.error(error_msg)
                raise ParamValueError(error_msg)

            # Scan folder for files (recursive)
            fileparams = []
            for file_path in root_folder.rglob("*"):
                if file_path.is_file():
                    # Build file metadata (matches save_file output format)
                    file_params = {
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "file_ext": file_path.suffix,
                        "is_overwrite": 0,
                        "file_version": 1,
                        "file_size": file_path.stat().st_size
                    }
                    fileparams.append(file_params)

            # Handle empty folder
            if not fileparams:
                error_msg = f"No files found in folder: {folder_path}"
                logger.warning(error_msg)
                raise NotFoundError(error_msg)

            logger.info(f"Found {len(fileparams)} files in folder {folder_path} - attaching to KB {kb_id}")

            # Persist metadata (no file copy)
            await self._save_file_metadata(
                fileparams=fileparams,
                app_id=app_id,
                kb_id=kb_id,
                batch_name=batch_name,
                skip_approval=True,
                biz_metadata=biz_metadata,
                created_by=created_by
            )
            logger.info(f"Successfully attached folder {folder_path} to KB {kb_id}")

        except Exception as e:
            error_msg = f"Failed to attach folder {folder_path}: {str(e)}"
            handle_exception(e, error_msg)

    async def reparse_files(self, file_ids: list[str]) -> None:
        """Reparse specified files in the knowledge base.
        
        Deletes existing text chunks for target files and resets file status to 
        trigger re-parsing workflow.
        
        Args:
            file_ids: List of file IDs to be re-parsed
            
        Raises:
            RuntimeError: If chunk deletion or status update fails (wrapped by handle_exception)
        """
        try:
            # Use session within context manager (FIX: moved inside try block)
            async with self.oracle_session as session:
                file_repo = FileRepository(session)
                chunk_repo = TxtChunkRepository(session)

                # Step 1: Delete existing text chunks for target files
                await chunk_repo.delete_by_file_ids(file_ids=file_ids)
                logger.debug(f"Deleted text chunks for files {file_ids}")
                
                # Step 2: Reset file status to trigger re-parsing (APPROVED = pending parse)
                await file_repo.batch_update_file_status(
                    file_ids=file_ids, 
                    status=FileStatus.APPROVED, 
                    log_msg="Reparse file trigger"
                )
                logger.info(f"Marked files {file_ids} as pending re-parsing (status: APPROVED)")
                
        except Exception as e:
            error_msg = f"Failed to reparse files: {str(e)}"
            handle_exception(e, error_msg)

    async def get_file_path_by_id(self, file_id: str) -> str:
        """Get file by ID.
        
        Args:
            file_id: Target file ID
            
        Returns:
            File path: File path
            
        Raises:
            NotFoundError: If file not found
        """
        async with self.oracle_session as session:
            file_repo = FileRepository(session)
            try:
                file = await file_repo.get_by_id(file_id=file_id)
                if not file:
                    error_msg = f"File not found: {file_id}"
                    logger.error(error_msg)
                    raise NotFoundError(error_msg)
                return file.file_path
            except Exception as e:
                error_msg = f"Failed to get file {file_id}: {str(e)}"
                handle_exception(e, error_msg)