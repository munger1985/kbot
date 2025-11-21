import os
import uuid
import json
import shutil
from pathlib import Path
from fastapi import UploadFile
from loguru import logger
from core.config.settings import get_app_config
from dao.entities.kbot_md_kb_batch import KbotMdKbBatch
from dao.entities.kbot_md_kb_files import KbotMdKbFiles
from core.dictionary import FileStatus, YesNoEnum
from dao.repositories.kbot_md_kb_repo import KbotMdKbRepository
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_biz_txt_embedding_factory import EmbeddingRepositoryFactory
from dao.repositories.kbot_md_parser_conf_repo import KbotMdParserConfRepository
from utils.common import run_in_thread_pool
from utils.decimal_encoder import DecimalEncoder



class KBFileOperator:
    '''
    文件上传和下载服务
    '''
    def __init__(self) -> None:
        '''初始化文件上传/删除服务'''

        config = get_app_config()
        self.file_storage = config.file_storage
        self.upload_workers = config.upload_workers


    def save_file(self, file: UploadFile, domain_id: int, kb_id: int, batch_name:str, overwrite: bool) -> dict:
            '''
            保存单个文件到磁盘并返回文件路径
            
            参数:
                file: 要上传的文件
                domain_id: 业务域ID
                kb_id: 目标知识库ID
                batch_name: 本次上传的批次名称
                overwrite: 是否覆盖已存在的文件
                
            返回:
                dict: 文件保存结果，包含:
                    {
                        "file_path": str,  // 文件保存路径
                        "file_name": str,  // 文件名
                        "file_ext": str,   // 文件扩展名
                        "is_overwrite": int,  // 是否覆盖(1是 0-否)
                        "file_version": int,  // 文件版本号
                        "file_size": int     // 文件大小
                    }
                或出错时返回空字典
            '''
            filename = file.filename
            if filename is None:
                    raise ValueError("文件名不能为空")
            try:
                logger.debug(f"开始保存文件: {filename} 到知识库: {kb_id}")
                file_content = file.file.read()

                root_path = Path(self.file_storage).resolve()  # 转换为绝对路径
                target_path = root_path / Path(str(domain_id)) / Path(str(kb_id)) / Path("source") / Path(batch_name)
                target_path.mkdir(parents=True, exist_ok=True)
                file_path = target_path / Path(filename)

                # 获取文件相关参数
                name, ext = os.path.splitext(filename)

                fileparams = {"file_path": str(file_path), 
                            "file_name": filename, 
                            "file_ext": ext, 
                            "is_overwrite": YesNoEnum.YES.value if overwrite else YesNoEnum.NO.value,
                            "file_version": 1, 
                            "file_size": len(file_content)}          
                
                # 处理文件名冲突
                if file_path.exists():
                    logger.debug(f"文件 {filename} 已存在")
                    counter = 1
                    new_filename = ""
                    if overwrite:
                        logger.debug(f"文件 {filename} 已存在，将覆盖该文件")
                        # 在获取最大版本号之后仍然需要覆盖最初的文件，后续保存文件仍然使用 file_path
                        new_path = file_path
                        while new_path.exists():
                            new_filename = f"{name}({counter}){ext}"
                            new_path = target_path / new_filename
                            counter += 1
                        fileparams["file_version"] = counter
                    else:
                        logger.debug(f"文件 {filename} 已存在，不进行覆盖")
                        # 添加数字后缀直到文件名不冲突
                        while file_path.exists():
                            new_filename = f"{name}({counter}){ext}"
                            file_path = target_path / new_filename
                            counter += 1
                        fileparams["file_name"] = new_filename
                        fileparams["file_path"] = str(file_path)
                        fileparams["file_version"] = counter
                    

                # 保存文件
                with open(file_path, "wb") as f:
                    f.write(file_content)
                
                logger.info(f"文件保存成功: {filename} -> {file_path}")
                return fileparams

            except Exception as e:
                logger.error(f"保存文件 {filename if 'filename' in locals() else '未知文件'} 失败: {str(e)}")
                raise e
            
    async def _save_files_in_thread(self, 
                            files: list[UploadFile],
                            domain_id: int,
                            kb_id: int,
                            batch_name: str,
                            overwrite: bool) -> list[dict]:
        '''
        通过多线程将上传的文件保存到对应知识库目录内
            
        参数:
            files: 要上传的文件列表
            domain_id: 业务域ID
            kb_id: 目标知识库ID
            batch_name: 本次上传的批次名称
            overwrite: 是否覆盖已存在的文件
        
        返回:
            List[dict]: 文件保存结果列表，每个结果包含:
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

        logger.debug(f"文件保存结果: {results}")
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
                                ) -> tuple[bool, str | None]:
        '''
        上传文件到知识库并保存记录到数据库
            
        参数:
            files: 要上传的文件列表
            app_id: 应用ID
            domain_id: 业务域ID
            kb_id: 目标知识库ID
            batch_name: 本次上传的批次名称
            overwrite: 是否覆盖已存在的文件
            batch_id: 可选的批次ID
            biz_metadata: 业务元数据(JSON格式)
            created_by: 创建者标识
        
        返回:
            tuple[bool, str | None]: (上传是否成功, 错误信息)
        '''
        
        # 从KB表获取默认配置
        kb_repo = KbotMdKbRepository()
        kb_entity = await kb_repo.get_by_id(kb_id)
        if kb_entity is None:
            error_msg = f"知识库 {kb_id} 不存在"
            logger.error(error_msg)
            return False, error_msg
        
        # 保存文件
        logger.info(f"开始上传 {len(files)} 个文件到知识库: {kb_id}")
        try:
            fileparams = await self._save_files_in_thread(files=files, domain_id=domain_id, kb_id=kb_id, batch_name=batch_name, overwrite=overwrite)
            logger.debug(f"文件已保存到磁盘: {[fp['file_name'] for fp in fileparams]}")
        except Exception as e:
            error_msg = f"保存文件到磁盘失败: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

        # 构造 batch 的实体用于保存到数据库
        batch_entity = KbotMdKbBatch(
            batch_id=batch_id,
            app_id=app_id,
            batch_name=batch_name,
            kb_id=kb_id,
            created_by=created_by,
            updated_by=created_by
        )
        
        # 构造 file 的实体列表用于批量保存到数据库
        parser_repo = KbotMdParserConfRepository()
        file_entitities = []
        for fileparam in fileparams:
             # 根据 kb id 从 PARSER_CONF 表获取默认配置
            parser_conf = await parser_repo.get_default_paser(file_ext=fileparam.get("file_ext", "").lower(), kb_id=kb_id)
            
            # 构造文件实体
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
                #chunk_parser = json.dumps(kb_entity.chunk_parser, cls=DecimalEncoder) if kb_entity.chunk_parser is not None else None,
                chunk_parser = json.dumps(parser_conf, cls=DecimalEncoder) if parser_conf is not None else None,
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
        
        # 保存上传记录到数据库
        kb_files_repo = KbotMdKbFilesRepository()
        try:
            logger.debug(f"开始将 {len(file_entitities)} 个文件保存到数据库，知识库: {kb_id}")
            r = await kb_files_repo.create(batch_entity, file_entitities)
            logger.info(f"成功将 {len(file_entitities)} 个文件保存到数据库")
            return True, None
        except Exception as e:
            error_msg = f"保存文件到数据库失败: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        
    async def _delete_files(self, 
                        domain_id: int, 
                        kb_id: int | None, 
                        batch_name: str | None,
                        file_paths: list[str] | None) -> tuple[int, int]:
        '''
        根据文件ID或批次ID或知识库ID从磁盘删除文件
        
        参数:
            domain_id: 文件所在的业务域ID
            kb_id: 文件所在的知识库ID(可选)
            batch_name: 文件所在的批次名称(可选)
            file_paths: 要删除的文件路径列表(可选)
        
        返回:
            tuple: 包含成功删除的文件数和失败文件数的元组
        '''
        success_cnt = 0
        failed_cnt = 0
        # 模式1: 通过文件路径删除
        if file_paths is not None:
            for file in file_paths:
                logger.info("正在删除文件: {}", file)
                if os.path.exists(Path(file)):
                    # 删除文件
                    try:
                        os.remove(Path(file))
                        logger.info("成功删除文件: {}", file)
                        success_cnt += 1
                    except Exception as e:
                        logger.error(f"删除文件 {file} 失败: {str(e)}")
                        failed_cnt += 1
                else:
                    logger.error(f"文件 {file} 不存在")
                    failed_cnt += 1
            return success_cnt, failed_cnt
        
        # 模式2: 通过批次名称删除
        elif batch_name is not None and kb_id is not None:
            # 使用知识库ID和批次名称构建完整目标路径
            root_path = Path(self.file_storage).resolve()  # 转换为绝对路径
            target_path = root_path / str(domain_id) / str(kb_id) / "source" / batch_name

            file_count = 0
            for files in os.walk(target_path):
                file_count += len(files)
            
            # 添加存在性检查
            if not target_path.exists():
                logger.warning(f"知识库 {kb_id} 中的批次 {batch_name} 没有文件，跳过删除")
                return success_cnt, failed_cnt
        
            try:
                logger.info(f"正在删除批次文件: {str(target_path)}")
                shutil.rmtree(target_path)
                success_cnt = file_count            
                return success_cnt, failed_cnt
            except Exception as e:
                logger.error(f"删除批次文件失败: {str(target_path)}: {str(e)}")
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
                logger.warning(f"知识库 {kb_id} 没有文件，跳过删除")
                return success_cnt, failed_cnt

            try:
                logger.info(f"正在删除知识库 {str(kb_id)} 中的文件: {str(target_path)}")
                shutil.rmtree(target_path)
                success_cnt = file_count            
                return success_cnt, failed_cnt
            except Exception as e:
                logger.error(f"删除知识库 {str(kb_id)} 中的文件失败: {str(target_path)}, 错误: {str(e)}")
                failed_cnt = file_count    
                return success_cnt, failed_cnt
        
        else:
            logger.error("无效的参数")
            return success_cnt, failed_cnt

    async def _delete_metadata(self, 
                            kb_id: int | None, 
                            batch_id: int | None, 
                            file_ids: list[str] | None) -> bool:
        """
        根据文件ID或批次ID或知识库ID删除文件元数据
        
        参数:
            kb_id: 知识库ID(用于整个知识库删除)(可选)
            batch_id: 要删除的批次ID(将删除该批次所有文件)(可选)
            file_ids: 要删除的特定文件ID列表(可选)
            
        返回:
            bool: 删除是否成功
        
        注意:
            - 必须提供file_ids或batch_id或kb_id之一(但一次只能提供一个参数)
        
        示例:
            >>> # 按知识库ID删除
            >>> result = await delete_file_metadata(kb_id=123, batch_id=None, file_ids=None)
            >>> # 按批次删除
            >>> result = await delete_file_metadata(kb_id=None, batch_id=456, file_ids=None)
            >>> # 删除特定文件
            >>> result = await delete_file_metadata(kb_id=None, batch_id=None, file_ids=[1,2,3])
        """
        file_repo = KbotMdKbFilesRepository()

        # 删除整个知识库中的所有文件
        if kb_id is not None and batch_id is None and file_ids is None:
            try:
                rowcnt = await file_repo.delete(kb_id, None, None)
                logger.info(f"成功删除知识库 {kb_id} 中的 {rowcnt} 个文件")
                # 删除知识库对应的解析默认配置
                parser_repo = KbotMdParserConfRepository()
                await parser_repo.delete_by_kb_id(kb_id)
                logger.info(f"成功删除知识库 {kb_id} 中的解析默认配置")
                return True
            except Exception as e:
                logger.error(f"删除知识库 {kb_id} 中的文件失败: {str(e)}")
                return False
            
        # 批次删除逻辑
        elif batch_id is not None:
            try:
                rowcnt = await file_repo.delete(None, batch_id, None)
                logger.info(f"成功删除批次 {batch_id} 中的 {rowcnt} 个文件")           
                return True           
            except Exception as e:
                logger.error(f"删除批次 {batch_id} 中的文件失败: {str(e)}")
                return False
        
        # 单个文件删除逻辑
        elif file_ids is not None:
            try:
                rowcnt = await file_repo.delete(None, None, file_ids)
                logger.info(f"成功删除 {rowcnt} 个文件")
                return True
            except Exception as e:
                logger.error(f"删除文件失败: {str(file_ids)}: {str(e)}")
                return False
        else:
            logger.error("无效的删除参数: 必须提供kb_id、batch_id或file_ids之一")
            return False

    async def _delete_vec_data(self, 
                            kb_id: int, 
                            batch_id: int | None, 
                            file_ids: list[str] | None) -> int:
        """
        根据文件ID从数据库中删除向量数据
        
        参数:
            kb_id: 知识库ID
            batch_id: 批次ID
            file_ids: 要删除的文件ID列表
        
        返回:
            int: 删除的记录行数

        """

        embed_repo = await EmbeddingRepositoryFactory.create_repository(kb_id=kb_id)
        if embed_repo is None:
            logger.error(f"知识库 {kb_id} 对应的向量库不存在")
            return 0

        file_repo = KbotMdKbFilesRepository()
        vec_cnt = 0
        # 模式1: 通过文件ID删除
        if file_ids is not None:
            try:
                logger.debug(f"正在删除向量库中 {len(file_ids)} 个文件的向量数据")
                vec_cnt = await embed_repo.delete_by_file_ids(kb_id, file_ids)
                logger.debug(f"成功删除向量库中的 {vec_cnt} 条记录")
                return vec_cnt
            except Exception as e:
                logger.error(f"删除向量数据失败: {str(e)}")
                return 0
        # 模式2: 通过批次ID删除
        elif batch_id is not None:
            try:
                file_repo = KbotMdKbFilesRepository()
                files = await file_repo.get_by_batch_id(batch_id)
                file_ids = []
                if files is None:
                    logger.error(f"未找到批次 {batch_id} 中的文件")
                    return 0
                for file in files:
                    file_ids.append(file.file_id)
                logger.debug(f"正在删除向量库中 {len(file_ids)} 个文件的向量数据")
                vec_cnt = await embed_repo.delete_by_file_ids(kb_id, file_ids)
                logger.debug(f"成功删除向量库中的 {vec_cnt} 条记录")
                return vec_cnt
            except Exception as e:
                logger.error(f"删除向量数据失败: {str(e)}")
                return 0
        # 模式3: 通过知识库ID删除
        else:
            try:
                file_repo = KbotMdKbFilesRepository()
                files = await file_repo.get_by_kb_id(kb_id)
                file_ids = []
                if files is None:
                    logger.error(f"未找到知识库 {kb_id} 中的文件")
                    return 0
                for file in files:
                    file_ids.append(file.file_id)
                logger.debug(f"正在删除向量库中 {len(file_ids)} 个文件的向量数据")
                vec_cnt = await embed_repo.delete_by_file_ids(kb_id, file_ids)
                logger.debug(f"成功删除向量库中的 {vec_cnt} 条记录")
                return vec_cnt
            except Exception as e:
                logger.error(f"删除向量数据失败: {str(e)}")
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
        统一文件删除服务，处理多种删除场景
        
        参数:
            app_id: 应用ID
            domain_id: 业务域ID
            kb_id: 知识库ID(用于整个知识库删除)
            batch_id: 批次ID(用于批次删除)
            batch_name: 批次名称(用于文件路径构建)
            file_ids: 文件ID列表(用于特定文件删除)
            file_paths: 文件路径列表(用于物理文件删除)
        
        返回:
            dict: 删除结果字典，包含成功和失败的文件数
        
        注意:
            - 支持三种删除模式：单个文件、批次或整个知识库
            - 如果部分失败会返回详细的结果统计
        
        示例:
            >>> # 删除特定文件
            >>> await delete_file_service(None, "kb1", None, None, [1,2], ["/path1","/path2"])
            >>> # 删除整个批次
            >>> await delete_file_service(None, "kb1", 123, "batch1", None, None)
            >>> # 删除整个知识库
            >>> await delete_file_service(None, "kb1", 123, None, None, None)
        """
        # 初始化结果对象
        result = {"success_file_cnt": 0, "failed_file_cnt": 0, "meta_cnt": 0, "vec_cnt": 0}

        # 模式1: 通过ID和路径删除特定文件
        if file_paths is not None and file_ids is not None:
            logger.info(f"开始删除文件，共 {len(file_paths)} 个文件...")
            # 1. 删除向量数据
            result["vec_cnt"] = await self._delete_vec_data(kb_id, None, file_ids)
            # 2. 删除文件元数据
            result["meta_cnt"] = await self._delete_metadata(None, None, file_ids)
            # 3. 物理删除文件
            result["success_file_cnt"], result["failed_file_cnt"] = await self._delete_files(domain_id, None, None, file_paths)
            return result
        # 模式2: 删除整个批次
        elif batch_name is not None and batch_id is not None:
            logger.info(f"开始删除批次中的文件: {batch_name}")
            # 1. 删除向量数据
            result["vec_cnt"] = await self._delete_vec_data(kb_id, batch_id, None)
            # 2. 删除文件元数据
            result["meta_cnt"] = await self._delete_metadata(None, batch_id, None)
            # 3. 物理删除文件
            result["success_file_cnt"], result["failed_file_cnt"] = await self._delete_files(domain_id, kb_id, batch_name, None)
            return result
        # 模式3: 删除整个知识库
        elif kb_id is not None and batch_id is None and file_ids is None:
            logger.info(f"开始删除知识库: {kb_id}")
            # 1. 删除向量数据
            result["vec_cnt"] = await self._delete_vec_data(kb_id, None, None)
            # 2. 删除文件元数据
            result["meta_cnt"] = await self._delete_metadata(kb_id, None, None)
            # 3. 物理删除文件
            result["success_file_cnt"], result["failed_file_cnt"] = await self._delete_files(domain_id, kb_id, None, None)
            return result
        else:
            logger.error("无效的删除参数: 必须提供kb_id、batch_id或file_ids之一")
            return result
        
    async def update_file_tags(
        self,
        file_id: str,
        tags: list[str]
    ) -> bool:
        """更新知识库文件的标签"""
        try:
            result = await KbotMdKbFilesRepository().update_tags(
                file_id=file_id,
                tags=tags
            )
            return result
        except Exception as e:
            logger.error(f"更新文件标签失败: {str(e)}")
            return False
