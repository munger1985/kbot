import os
import uuid
import json
import shutil
from pathlib import Path
from fastapi import UploadFile
from loguru import logger
from configuration import ConfigManager
from dao.entities.kbot_md_kb_batch import KbotMdKbBatch
from dao.entities.kbot_md_kb_files import KbotMdKbFiles
from core.dictionary import FileStatus, YesNoEnum
from dao.repositories.kbot_md_kb_repo import KbotMdKbRepository
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_biz_txt_embedding_repo import KbotBizTxtEmbeddingRepository
from utils.common_methods import run_in_thread_pool
from utils.decimal_encoder import DecimalEncoder


class KBFileOperator:
    '''
    File upload and download service. 
    '''
    def __init__(self) -> None:
        '''Initialize the file upload/delete service. '''

        config = ConfigManager.get_app_config()
        self.file_storage = config.kbot.file_storage
        self.upload_workers = config.kbot.upload_workers


    def save_file(self, file: UploadFile, domain_id: int, kb_id: int, batch_name:str, overwrite: bool) -> dict:
            '''
            Save single file to disk and return the file path // 保存单个文件到磁盘并返回文件路径
            Args:
                file: File to upload // 要上传的文件
                domain_id: Target domain id // 业务域id
                kb_id: Target knowledge base id // 目标知识库id
                batch_name: Batch name for this upload // 本次上传的批次名称
                overwrite: Whether to overwrite existing files // 是否覆盖已存在的文件
                
            Returns:
                dict: File saving result containing: // 文件保存结果，包含:
                    {
                        "file_path": str,  // 文件保存路径
                        "file_name": str,  // 文件名
                        "file_ext": str,   // 文件扩展名
                        "is_overwrite": int,  // 是否覆盖(1是 0-否)
                        "file_version": int,  // 文件版本号
                        "file_size": int     // 文件大小
                    }
                or empty dict on error // 或出错时返回空字典
            '''
            filename = file.filename
            if filename is None:
                    raise ValueError("Filename cannot be None")
            try:
                logger.debug(f"Start saving file: {filename} to knowledge base: {kb_id}")
                file_content = file.file.read()

                root_path = Path(self.file_storage).resolve()  # 转换为绝对路径
                target_path = root_path / Path(str(domain_id)) / Path(str(kb_id)) / Path("source") / Path(batch_name)
                target_path.mkdir(parents=True, exist_ok=True)
                file_path = target_path / Path(filename)

                # Get file parameters. // 获取文件相关参数
                name, ext = os.path.splitext(filename)

                fileparams = {"file_path": str(file_path), 
                            "file_name": filename, 
                            "file_ext": ext, 
                            "is_overwrite": YesNoEnum.YES.value if overwrite else YesNoEnum.NO.value,
                            "file_version": 1, 
                            "file_size": len(file_content)}          
                
                # Handle filename conflicts. // 处理文件名冲突
                if file_path.exists():
                    logger.debug(f"File {filename} already exists.")
                    counter = 1
                    new_filename = ""
                    if overwrite:
                        logger.debug(f"File {filename} already exists, will overwrite it.")
                        # After retrieving the maximum version number, the original file must still be overwritten, 
                        # and subsequent saves will continue using the same file_path.
                        # 在获取最大版本号之后仍然需要覆盖最初的文件，后续保存文件仍然使用 file_path
                        new_path = file_path
                        while new_path.exists():
                            new_filename = f"{name}({counter}){ext}"
                            new_path = target_path / new_filename
                            counter += 1
                        fileparams["file_version"] = counter
                    else:
                        logger.debug(f"File {filename} already exists, will not overwrite it.")
                        # Append a numeric suffix to the filename until the conflict is resolved. // 添加数字后缀直到文件名不冲突
                        while file_path.exists():
                            new_filename = f"{name}({counter}){ext}"
                            file_path = target_path / new_filename
                            counter += 1
                        fileparams["file_name"] = new_filename
                        fileparams["file_path"] = str(file_path)
                        fileparams["file_version"] = counter
                    

                # Save the file. // 保存文件
                with open(file_path, "wb") as f:
                    f.write(file_content)
                
                logger.info(f"File saved successfully: {filename} -> {file_path}")
                return fileparams

            except Exception as e:
                logger.error(f"Failed to save file {filename if 'filename' in locals() else 'unknown'}: {str(e)}")
                raise e
            
    async def _save_files_in_thread(self, 
                            files: list[UploadFile],
                            domain_id: int,
                            kb_id: int,
                            batch_name: str,
                            overwrite: bool) -> list[dict]:
        '''
        Save uploaded files to corresponding knowledge base directory using multi-threading. // 通过多线程将上传的文件保存到对应知识库目录内
        Args:
            files: List of files to upload // 要上传的文件列表
            domain_id: Target domain id // 业务域id
            kb_id: Target knowledge base id // 目标知识库id
            batch_name: Batch name for this upload // 本次上传的批次名称
            overwrite: Whether to overwrite existing files // 是否覆盖已存在的文件
        
        Returns:
            List[dict]: List of file saving results, each contains: // 文件保存结果列表，每个结果包含:
                {
                    "file_path": str,  // 文件保存路径
                    "file_name": str,  // 文件名
                    "file_ext": str,   // 文件扩展名
                    "is_overwrite": str,  // 是否覆盖(Y/N)
                    "file_version": int,  // 文件版本号
                    "file_size": int     // 文件大小
                }
        '''
        file_params = [{"file": file, "domain_id": domain_id, "kb_id": kb_id,
                "batch_name": batch_name, "overwrite": overwrite} for file in files]
        results = [result async for result in run_in_thread_pool(func=self.save_file, params=file_params, workers=self.upload_workers)]

        logger.debug(f"file save result: {results}")
        return results

    async def upload_file_service(self, 
                                files: list[UploadFile], 
                                app_id: int,
                                domain_id: int,
                                kb_id: int,
                                batch_name:str,
                                overwrite: bool,
                                batch_id: int | None = None,
                                biz_metadata: dict | None = None,
                                created_by: str | None = None,
                                ) -> bool:
        '''
        Upload files to knowledge base and save records to database. // 上传文件到知识库并保存记录到数据库
        Args:
            files: List of files to upload // 要上传的文件列表
            app_id: Application ID // 应用ID
            domain_id: Target domain id // 业务域id
            kb_id: Target knowledge base id // 目标知识库id
            batch_name: Batch name for this upload // 本次上传的批次名称
            overwrite: Whether to overwrite existing files // 是否覆盖已存在的文件
            batch_id: Optional batch ID // 可选的批次ID
            biz_metadata: Business metadata in JSON format // 业务元数据(JSON格式)
            created_by: Creator identifier // 创建者标识
        
        Returns:
            KBUploadResponse: On successful upload // 上传成功时返回
            KBErrorResponse: On error // 出错时返回
        '''
        
        # Get default configuration from KB table. //从KB表获取默认配置
        kb_repo = KbotMdKbRepository()
        kb_entity = await kb_repo.get_by_id(kb_id)
        if kb_entity is None:
            logger.error(f"Knowledge base {kb_id} does not exist.")
            return False
        
        # Save the file. // 保存文件
        logger.info(f"Start uploading {len(files)} files to knowledge base: {kb_id}")
        fileparams = await self._save_files_in_thread(files=files, domain_id=domain_id, kb_id=kb_id, batch_name=batch_name, overwrite=overwrite)
        logger.debug(f"Files saved to disk: {[fp['file_name'] for fp in fileparams]}")

        # Construct the batch entities for saving to the database. //构造 batch 的实体用于保存到数据库
        batch_entity = KbotMdKbBatch(
            batch_id=batch_id,
            app_id=app_id,
            batch_name=batch_name,
            kb_id=kb_id,
            created_by=created_by,
            updated_by=created_by
        )
        
        # Construct the file entities for batch saving to the database. //构造 file 的实体列表用于批量保存到数据库
        file_entitities = []
        for fileparam in fileparams:
            file_entitity = KbotMdKbFiles(
                file_id = str(uuid.uuid4()),
                app_id = app_id,
                kb_id = kb_id,
                batch_id = batch_id,
                file_path = fileparam["file_path"],
                file_name = fileparam["file_name"],
                file_ext = fileparam["file_ext"],
                status=FileStatus.UPLOADED.value,
                file_version = fileparam["file_version"],
                is_overwrite = fileparam["is_overwrite"],
                security_level = kb_entity.security_level or 1,
                chunk_parser = json.dumps(kb_entity.chunk_parser, cls=DecimalEncoder) if kb_entity.chunk_parser is not None else None,
                enable_summary = kb_entity.enable_summary,
                is_img2txt = kb_entity.is_img2txt,
                is_table_head_fill = kb_entity.is_table_head_fill,
                process_priority = kb_entity.process_priority,
                file_size = fileparam["file_size"],
                biz_metadata = json.dumps(biz_metadata, cls=DecimalEncoder) if biz_metadata is not None else None,
                created_by=created_by,
                updated_by=created_by
            )
            file_entitities = file_entitities + [file_entitity]
        
        # Save upload records to database. // 保存上传记录到数据库
        kb_files_repo = KbotMdKbFilesRepository()
        try:
            logger.debug(f"Start saving {len(file_entitities)} files to database for knowledge base: {kb_id}")
            r = await kb_files_repo.create(batch_entity, file_entitities)
            logger.info(f"Successfully saved {len(file_entitities)} files to database")
            return r
        except Exception as e:
            logger.error(f"Failed to save files to database for knowledge base: {kb_id}")
            logger.error(e)
            raise e
        
    async def _delete_files(self, 
                        domain_id: int, 
                        kb_id: int | None, 
                        batch_name: str | None,
                        file_paths: list[str] | None) -> tuple[int, int]:
        '''
        Delete files from disk by file IDs or batch ID or knowledge base ID.
        根据文件ID或批次ID或知识库ID从磁盘删除文件
        
        Args/参数:
            domain_id: ID of the domain where the files reside
            文件所在的业务域ID
            kb_id: ID of the knowledge base where the files reside (optional)
            文件所在的知识库ID(可选)
            batch_name: Name of the batch where the files reside (optional)
            文件所在的批次名称(可选)
            file_paths: List of file paths to delete (optional)
            要删除的文件路径列表(可选)
        
        Returns/返回:
            result: The number of successfully deleted files and the number of failed files
            包含成功删除的文件数和失败文件数的结果对象
        '''
        success_cnt = 0
        failed_cnt = 0
        # 模式1: 通过文件ID删除
        if file_paths is not None:
            for file in file_paths:
                logger.info("Deleting file: {}", file)
                if os.path.exists(Path(file)):
                    try:
                        os.remove(Path(file))
                        logger.info("Successfully deleted file: {}", file)
                        success_cnt += 1
                    except Exception as e:
                        logger.error(f"Failed to delete file {file}: {str(e)}")
                        failed_cnt += 1
                else:
                    logger.error(f"File {file} does not exist")
                    failed_cnt += 1
            return success_cnt, failed_cnt
        
        # 模式2: 通过批次ID删除
        elif batch_name is not None and kb_id is not None:
            # 使用知识库ID和批次名称构建完整目标路径
            root_path = Path(self.file_storage).resolve()  # 转换为绝对路径
            target_path = root_path / str(domain_id) / str(kb_id) / "source" / batch_name

            file_count = 0
            for files in os.walk(target_path):
                file_count += len(files)
            
            # 添加存在性检查
            if not target_path.exists():
                logger.warning(f"The batch {batch_name} in knowledge base {kb_id} has no files, skip deletion")
                return success_cnt, failed_cnt
        
            try:
                logger.info(f"Deleting batch files: {str(target_path)}")
                shutil.rmtree(target_path)
                success_cnt = file_count            
                return success_cnt, failed_cnt
            except Exception as e:
                logger.error(f"Failed to delete batch files: {str(target_path)}: {str(e)}")
                failed_cnt = file_count    
                return success_cnt, failed_cnt
        
        # 模式3: 通过知识库ID删除
        elif kb_id is not None:
            # 使用知识库ID构建完整目标路径
            root_path = Path(self.file_storage).resolve()  # 转换为绝对路径
            target_path = root_path / str(domain_id) / str(kb_id)

            file_count = 0
            for files in os.walk(target_path):
                file_count += len(files)
            
            # 添加存在性检查
            if not target_path.exists():
                logger.warning(f"Knowledge base {kb_id} has no files, skip deletion")
                return success_cnt, failed_cnt

            try:
                logger.info(f"Deleting files in knowlodge base {str(kb_id)}: {str(target_path)}")
                shutil.rmtree(target_path)
                success_cnt = file_count            
                return success_cnt, failed_cnt
            except Exception as e:
                logger.error(f"Failed to delete files in knowlodge base {str(kb_id)}: {str(target_path)}, error: {str(e)}")
                failed_cnt = file_count    
                return success_cnt, failed_cnt
        
        else:
            logger.error("Invalid parameters")
            return success_cnt, failed_cnt

    async def _delete_metadata(self, 
                            kb_id: int | None, 
                            batch_id: int | None, 
                            file_ids: list[str] | None) -> bool:
        """
        Delete file metadata either by individual file IDs or by batch ID or by kb ID.
        根据文件ID或批次ID或知识库ID删除文件元数据
        
        Args/参数:
            kb_id: ID of the knowledge base to delete all contained files (optional)
            知识库ID(用于整个知识库删除)(可选)
            batch_id: ID of the batch to delete all contained files (optional)
            要删除的批次ID(将删除该批次所有文件)(可选)
            file_ids: List of specific file IDs to delete (optional)
            要删除的特定文件ID列表(可选)
            
        Returns/返回:
            result: Object containing counts of successfully and failed deletions
            包含成功和失败删除计数的结果对象
        
        Note/注意:
            - Either file_ids or batch_id or kb_id must be provided (but only one parameter at a time)
            必须提供file_ids或batch_id或kb_id之一(但一次只能提供一个参数)
            - This is an async function and needs to be awaited
            这是一个异步函数，需要await调用
        
        Example/示例:
            >>> # Delete by kb_id
            >>> result = await delete_file_metadata(kb_id=123, batch_id=None, file_ids=None)
            >>> # Delete by batch
            >>> result = await delete_file_metadata(kb_id=None, batch_id=456, file_ids=None)
            >>> # Delete specific files
            >>> result = await delete_file_metadata(kb_id=None, batch_id=None, file_ids=[1,2,3])
        """
        file_repo = KbotMdKbFilesRepository()

        # Delete all files in the knowledge base
        # 删除整个知识库中的所有文件
        if kb_id is not None:
            try:
                rowcnt = await file_repo.delete(kb_id, None, None)
                logger.info(f"Successfully deleted {rowcnt} files in kb {kb_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete files in kb {kb_id}: {str(e)}")
                return False
            
        # Batch deletion logic
        # 批次删除逻辑
        elif batch_id is not None:
            try:
                rowcnt = await file_repo.delete(None, batch_id, None)
                logger.info(f"Successfully deleted {rowcnt} files in batch {batch_id}")           
                return True           
            except Exception as e:
                logger.error(f"Failed to delete files in batch {batch_id}: {str(e)}")
                return False
        
        # Individual file deletion logic
        # 单个文件删除逻辑
        elif file_ids is not None:
            try:
                rowcnt = await file_repo.delete(None, None, file_ids)
                logger.info(f"Successfully deleted {rowcnt} files")
                return True
            except Exception as e:
                logger.error(f"Failed to delete file(s): {str(file_ids)}: {str(e)}")
                return False
        else:
            logger.error("Invalid deletion parameters: must provide either kb_id, batch_id, or file_ids")
            return False

    async def _delete_vec_data(self, 
                            kb_id: int, 
                            batch_id: int | None, 
                            file_ids: list[str] | None) -> int:
        """
        Delete vector data from the database by file IDs and delete file metadata finally. 
        根据文件ID从数据库中删除向量数据，最后彻底删除文件元数据。
        
        Args/参数:
            kb_id: Knowledge base ID //知识库ID
            batch_id: Batch ID //批次ID
            file_ids: List of file IDs to delete //要删除的文件ID列表
        
        Returns/返回:
            result: Row count of deleted records //删除的记录行数
        
        Note/注意:
            - This is an async function and needs to be awaited //这是一个异步函数，需要await调用
        """

        embed_repo = KbotBizTxtEmbeddingRepository(kb_id=kb_id)
        await embed_repo.initialize()
        file_repo = KbotMdKbFilesRepository()
        vec_cnt = 0
        # Mode 1: Delete by file IDs //模式1：通过文件ID删除
        if file_ids is not None:
            try:
                logger.debug(f"Deleting vector data for {len(file_ids)} files in vector base")
                vec_cnt = await embed_repo.delete_by_file_ids(kb_id, file_ids)
                logger.debug(f"Successfully deleted {vec_cnt} records in vector base")
                return vec_cnt
            except Exception as e:
                logger.error(f"Failed to delete vector data: {str(e)}")
                return 0
        # Mode 2: Delete by batch ID //模式2：通过批次ID删除
        elif batch_id is not None:
            try:
                file_repo = KbotMdKbFilesRepository()
                files = await file_repo.get_by_batch_id(batch_id)
                file_ids = []
                if files is None:
                    logger.error(f"Files in batch {batch_id} not found")
                    return 0
                for file in files:
                    file_ids.append(file.file_id)
                logger.debug(f"Deleting vector data for {len(file_ids)} files in vector base")
                vec_cnt = await embed_repo.delete_by_file_ids(kb_id, file_ids)
                logger.debug(f"Successfully deleted {vec_cnt} records in vector base")
                return vec_cnt
            except Exception as e:
                logger.error(f"Failed to delete vector data: {str(e)}")
                return 0
        # Mode 3: Delete by knowledge base ID //模式3：通过知识库ID删除
        else:
            try:
                file_repo = KbotMdKbFilesRepository()
                files = await file_repo.get_by_kb_id(kb_id)
                file_ids = []
                if files is None:
                    logger.error(f"Files in knowledge base {kb_id} not found")
                    return 0
                for file in files:
                    file_ids.append(file.file_id)
                logger.debug(f"Deleting vector data for {len(file_ids)} files in vector base")
                vec_cnt = await embed_repo.delete_by_file_ids(kb_id, file_ids)
                logger.debug(f"Successfully deleted {vec_cnt} records in vector base")
                return vec_cnt
            except Exception as e:
                logger.error(f"Failed to delete vector data: {str(e)}")
                return 0


    async def delete_file_service(
        self,
        app_id: int,
        domain_id: int,
        kb_id: int, 
        batch_id: int | None, 
        batch_name: str | None,
        file_ids: list[str] | None,
        file_paths: list[str] | None
    ) -> dict:
        """
        Unified file deletion service that handles multiple deletion scenarios.
        统一文件删除服务，处理多种删除场景
        
        Args/参数:
            kb_id: Knowledge base ID (for full KB deletion) 
            知识库ID(用于整个知识库删除)
            batch_id: Batch ID (for batch deletion)
            批次ID(用于批次删除)
            batch_name: Batch name (for file path construction)
            批次名称(用于文件路径构建)
            file_ids: List of file IDs (for specific file deletion)
            文件ID列表(用于特定文件删除)
            file_paths: List of file paths (for physical file deletion)
            文件路径列表(用于物理文件删除)
        
        Returns/返回:
            SuccessWithErrorResponse: Standard response object with detailed deletion results
            标准响应对象，包含详细的删除结果
        
        Note/注意:
            - Supports three deletion modes: single files, batch, or entire knowledge base
            支持三种删除模式：单个文件、批次或整个知识库
            - Returns HTTP 207 (Multi-Status) if partial failures occur
            如果部分失败会返回HTTP 207(多状态)
        
        Example/示例:
            >>> # Delete specific files
            >>> await delete_file_service(None, "kb1", None, None, [1,2], ["/path1","/path2"])
            >>> # Delete entire batch
            >>> await delete_file_service(None, "kb1", 123, "batch1", None, None)
            >>> # Delete entire knowledge base
            >>> await delete_file_service(None, "kb1", 123, None, None, None)
        """
        # Initialize result objects
        # 初始化结果对象
        result = {"success_file_cnt": 0, "failed_file_cnt": 0, "meta_cnt": 0, "vec_cnt": 0}

        # Mode 1: Delete specific files by IDs and paths
        # 模式1：通过ID和路径删除特定文件
        if file_paths is not None and file_ids is not None:
            logger.info(f"Starting to delete files, total {len(file_paths)} files...")
            # 1. Delete vector data //删除向量数据
            result["vec_cnt"] = await self._delete_vec_data(kb_id, None, file_ids)
            # 2. Delete file metadata //删除文件元数据
            result["meta_cnt"] = await self._delete_metadata(None, None, file_ids)
            # 3. Delete files physically //物理删除文件
            result["success_file_cnt"], result["failed_file_cnt"] = await self._delete_files(domain_id, None, None, file_paths)
            return result
        # Mode 2: Delete entire batch
        # 模式2：删除整个批次
        elif batch_name is not None and batch_id is not None:
            logger.info(f"Starting to delete files in batch: {batch_name}")
            # 1. Delete vector data //删除向量数据
            result["vec_cnt"] = await self._delete_vec_data(kb_id, batch_id, None)
            # 2. Delete file metadata //删除文件元数据
            result["meta_cnt"] = await self._delete_metadata(None, batch_id, None)
            # 3. Delete files physically //物理删除文件
            result["success_file_cnt"], result["failed_file_cnt"] = await self._delete_files(domain_id, kb_id, batch_name, None)
            return result
        # Mode 3: Delete entire knowledge base
        # 模式3：删除整个知识库
        elif kb_id is not None and batch_id is None and file_ids is None:
            logger.info(f"Starting to delete knowledge base: {kb_id}")
            # 1. Delete vector data //删除向量数据
            result["vec_cnt"] = await self._delete_vec_data(kb_id, None, None)
            # 2. Delete file metadata //删除文件元数据
            result["meta_cnt"] = await self._delete_metadata(kb_id, None, None)
            # 3. Delete files physically //物理删除文件
            result["success_file_cnt"], result["failed_file_cnt"] = await self._delete_files(domain_id, kb_id, None, None)
            return result
        else:
            logger.error("Invalid deletion parameters: must provide either kb_id, batch_id, or file_ids")
            return result

    