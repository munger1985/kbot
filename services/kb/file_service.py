import os
import shutil
from pathlib import Path
from typing import Any
from fastapi import UploadFile
from loguru import logger
from core.config.settings import get_app_config, get_prompt_config
from core.dictionary import FileStatus, ParserEngine
from core.exceptions import *
from core.database.oracle import get_session
from dao.entities import FileEntity
from dao.repositories import KBRepository, FileRepository, TxtChunkRepository
from utils.common import run_in_thread_pool


class FileService:
    '''
    文件上传/删除等服务
    '''
    def __init__(self) -> None:
        '''初始化文件上传/删除服务'''
        config = get_app_config()
        self.file_storage = config.file_storage
        self.upload_workers = config.upload_workers

    @property
    def db_session(self):
        return get_session()

###############################################################################
# 保存文件
###############################################################################

    def _save_file(self, file: UploadFile, kb_id: int, batch: str, overwrite: bool) -> dict:
            '''
            保存单个文件到磁盘并返回文件路径
            
            参数:
                file: 要上传的文件
                kb_id: 目标知识库ID
                batch: 本次上传的批次
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
                raise ParamValueError("文件名不能为空")
            
            
            logger.debug(f"开始保存文件: {filename} 到知识库")
            try:
                # 读取文件内容
                file_content = file.file.read()

                root_path = Path(self.file_storage).resolve()  # 转换为绝对路径
                target_path = root_path / Path(str(kb_id)) / Path(batch)
                target_path.mkdir(parents=True, exist_ok=True)
                file_path = target_path / Path(filename)

                # 获取文件相关参数
                name, ext = os.path.splitext(filename)

                fileparams = {"file_path": str(file_path), 
                              "file_name": filename, 
                              "file_ext": ext, 
                              "is_overwrite": overwrite,
                              "file_version": 1, 
                              "file_size": len(file_content)}          
                
                # 处理文件名冲突
                if file_path.exists():
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
                msg = f"保存文件 {filename if 'filename' in locals() else '未知文件'} 失败: {e}"
                logger.exception(msg)
                raise InternalServerError(msg)
            
    async def _save_file_in_thread(self, 
                            files: list[UploadFile],
                            kb_id: int,
                            batch: str,
                            overwrite: bool) -> list[dict]:
        '''
        通过多线程将上传的文件保存到对应知识库目录内
            
        Args:
            files: 要上传的文件列表
            kb_id: 知识库ID
            batch: 批次路径
            overwrite: 是否覆盖已存在的文件
        
        Returns:
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
        file_params = [{"file": file, "kb_id": kb_id, "batch": batch, "overwrite": overwrite} for file in files]
        
        # 收集异步生成器的结果
        results = []
        async for result in run_in_thread_pool(func=self._save_file, params=file_params, workers=self.upload_workers):
            results.append(result)

        return results

    async def _save_metadata(self, 
                             fileparams: list[dict],
                             app_id: int,
                             domain_id: int,
                             kb_id: int,
                             batch: str,
                             skip_approval: bool,
                             security_level: int | None = 0,
                             process_priority: int | None = 1,
                             biz_metadata: dict | None = None,
                             created_by: str | None = None):
        """
        保存文件元数据到数据库
        """
        async with self.db_session as session:
            file_repo = FileRepository(session)
            kb_repo = KBRepository(session)
            # 从知识库获取模型配置
            models = await kb_repo.get_model_by_id(kb_id)
            model_config = models.get("model_config", {})
            if not model_config:
                msg = f"知识库 {kb_id} 未配置模型，跳过处理"
                logger.warning(msg)
                return
            if model_config and isinstance(model_config, dict):
                txt_embedding_model = model_config.get("txt_embedding_model", None) # 文本嵌入模型
                llm_model = model_config.get("llm_model", None) # LLM模型
                vlm_model = model_config.get("vlm_model", None) # 视觉语言模型
            else:
                txt_embedding_model = None
                llm_model = None
                vlm_model = None

            logger.debug(f"txt_embedding_model: {txt_embedding_model}, llm_model: {llm_model}, vlm_model: {vlm_model}")
            # 延迟导入避免循环依赖
            from agent.prompt import default_prompt
            prompt = await default_prompt.generate(get_prompt_config().image2text)

            # 获取默认的解析引擎配置
            from services.basic import ParserConfService
            
            parser_conf_service = ParserConfService()
            try:
                default_parser_conf = await parser_conf_service.get_parser_params_by_engine(domain_id=domain_id, engine=ParserEngine.TEXT.value)
            except Exception as e:
                logger.warning(f"获取默认解析引擎配置失败: {e}")
                # 定义默认的解析引擎配置
                default_parser_conf = {
                    "do_ocr": False, 
                    "overlap": 50, 
                    "use_vlm": bool(vlm_model), 
                    "llm_model": llm_model,
                    "vlm_model": vlm_model, 
                    "txt_embedding_model": txt_embedding_model,
                    "chunk_size": 1000, 
                    "ocr_engine": "tesseract", 
                    "img2txt_prompt": prompt, 
                    "image_scale": 2.0,
                    "min_chunk_len": 200, 
                    "generate_picture_images": True,
                    "extract_graph": False
                }
            
            # 构造 file 的实体列表用于批量保存到数据库
            file_entitities = []
            for fileparam in fileparams:
                
                # 构造文件实体
                file_entitity = FileEntity(
                    app_id = app_id,
                    kb_id = kb_id,
                    batch = batch,
                    file_path = fileparam["file_path"],
                    file_name = fileparam["file_name"],
                    file_ext = fileparam["file_ext"],
                    status=FileStatus.UPLOADED.value if not skip_approval else FileStatus.APPROVED.value,
                    file_version = fileparam["file_version"],
                    is_overwrite = fileparam["is_overwrite"],
                    security_level = security_level,
                    parser_params = default_parser_conf,
                    process_priority = process_priority,
                    file_size = fileparam["file_size"],
                    biz_metadata = biz_metadata,
                    created_by=created_by,
                    updated_by=created_by
                )
                file_entitities = file_entitities + [file_entitity]
            
            # 保存文件元数据到数据库
            try:
                logger.debug(f"开始将 {len(file_entitities)} 个文件保存到数据库，知识库: {kb_id}")
                await file_repo.create(file_entitities)
                logger.info(f"成功将 {len(file_entitities)} 个文件保存到数据库")
            except Exception as e:
                handle_exception(e, "保存文件元数据到数据库失败")


    async def upload_file_service(self, 
                                files: list[UploadFile], 
                                app_id: int,
                                domain_id: int,
                                kb_id: int,
                                batch:str,
                                overwrite: bool,
                                skip_approval: bool,
                                biz_metadata: dict | None = None,
                                created_by: str | None = None,
                                ):
        '''
        上传文件到知识库并保存记录到数据库
            
        Args:
            app_id: 应用ID
            files: 要上传的文件列表
            domain_id: 业务域ID
            kb_id: 目标知识库ID
            batch: 本次上传的批次名称
            overwrite: 是否覆盖已存在的文件
            skip_approval: 是否跳过审批
            biz_metadata: 业务元数据(JSON格式)
            created_by: 创建者标识
        '''
        async with self.db_session as session:
            kb_repo = KBRepository(session)
            try:
                # 获取知识库默认配置
                kb = await kb_repo.get_by_id(kb_id)
            except Exception as e:
                handle_exception(e, "获取知识库默认配置失败")
            
            # 保存文件
            logger.info(f"开始上传 {len(files)} 个文件到知识库: {kb.kb_name}")
            try:
                # 保存文件到磁盘
                fileparams = await self._save_file_in_thread(files=files, kb_id=kb_id, batch=batch, overwrite=overwrite)
                logger.debug(f"文件已保存到磁盘: {[fp['file_name'] for fp in fileparams]}")
            except Exception as e:
                error_msg = f"保存文件到磁盘失败: {e}"
                logger.error(error_msg)
                raise InternalServerError(error_msg)
            
            # 保存文件元数据到数据库
            await self._save_metadata(fileparams=fileparams, 
                                        app_id=app_id,
                                        domain_id=domain_id,
                                        kb_id=kb_id,
                                        batch=batch,
                                        skip_approval=skip_approval,
                                        biz_metadata=biz_metadata,
                                        security_level=kb.security_level,
                                        process_priority=kb.process_priority,
                                        created_by=created_by)

            
    async def attach_folder(self, 
                            folder_path: str,
                            app_id: int,
                            domain_id: int,
                            kb_id: int,
                            batch:str,
                            skip_approval: bool,
                            biz_metadata: dict | None = None,
                            created_by: str | None = None,
                        ):
        """
        直接将现有文件夹中的文件信息同步到知识库数据库中
           
        Args:
            folder_path: 要同步的文件夹路径
            app_id: 应用ID
            domain_id: 业务域ID
            kb_id: 目标知识库ID
            batch: 本次上传的批次名称
            skip_approval: 是否跳过审批
            biz_metadata: 业务元数据(JSON格式)
            created_by: 创建者标识
        """
        root_folder = Path(folder_path).resolve()
        if not root_folder.exists() or not root_folder.is_dir():
            msg = f"提供的路径不存在或不是目录: {folder_path}"
            logger.error(msg)
            raise ParamValueError(msg)

        fileparams = []
        
        try:
            # 1. 递归遍历文件夹下的所有文件
            # 使用 rglob("*") 匹配所有子目录下的文件
            for p in root_folder.rglob("*"):
                if p.is_file():
                    # 提取文件基础信息
                    filename = p.name
                    ext = p.suffix
                    file_size = p.stat().st_size
                    
                    # 构造与 save_file 返回结构一致的字典
                    param = {
                        "file_path": str(p),         # 文件的绝对物理路径
                        "file_name": filename,       # 文件名
                        "file_ext": ext,             # 后缀名
                        "is_overwrite": 0,           # 附加文件夹模式通常默认为不覆盖
                        "file_version": 1,           # 初始版本
                        "file_size": file_size       # 字节大小
                    }
                    fileparams.append(param)

        except Exception as e:
            error_msg = f"遍历文件夹失败: {e}"
            logger.error(error_msg)
            raise InternalServerError(error_msg)

        if not fileparams:
            msg = f"文件夹 {folder_path} 中未找到任何文件"
            logger.warning(msg)
            raise NotFoundError(msg)

        logger.info(f"扫描到 {len(fileparams)} 个文件，准备写入数据库。")

        async with self.db_session as session:
            kb_repo = KBRepository(session)
            try:
                # 获取知识库默认配置
                kb = await kb_repo.get_by_id(kb_id)
            except Exception as e:
                handle_exception(e, "获取知识库默认配置失败")
        
            # 2. 直接调用元数据保存方法
            await self._save_metadata(
                fileparams=fileparams,
                app_id=app_id,
                domain_id=domain_id,
                kb_id=kb_id,
                batch=batch,
                skip_approval=skip_approval,
                security_level=kb.security_level,
                process_priority=kb.process_priority,
                biz_metadata=biz_metadata,
                created_by=created_by
            )

            logger.info(f"文件夹 {folder_path} 的元数据已成功关联至知识库 {kb_id}")


###############################################################################
# 删除文件
###############################################################################
        
    def _delete_file(self, file_path: str, is_batch: bool):
        '''根据完整文件名从磁盘删除文件，如果batch文件夹为空则自动删除
        
        Args:
            file_path: 要删除的完整文件名
        '''
        
        # 用于记录需要检查的batch文件夹
        batch_folders_to_check = set()
        
        # 如果是批次删除，直接删除整个文件夹
        if is_batch:
            # 如果是批次文件夹，直接删除整个文件夹
            batch_folder = Path(file_path).parent
            try:
                # 检查文件夹是否仍然存在
                if not batch_folder.exists():
                    logger.info(f"批次文件夹已被删除，跳过删除操作: {batch_folder}")
                    return
                    
                # 检查是否是文件夹
                if not batch_folder.is_dir():
                    logger.info(f"路径不是文件夹，跳过删除操作: {batch_folder}")
                    return
                
                # 执行删除
                shutil.rmtree(batch_folder)
                logger.debug(f"成功删除批次文件夹: {batch_folder}")
                
            except FileNotFoundError:
                # 文件在删除过程中被其他线程删除
                logger.info(f"批次文件夹在删除过程中已被其他线程删除: {batch_folder}")
            except PermissionError as e:
                logger.error(f"权限不足，无法删除文件夹: {batch_folder}, 错误: {e}")
            except OSError as e:
                logger.error(f"操作系统错误，删除文件夹失败: {batch_folder}, 错误: {e}")
            except Exception as e:
                logger.error(f"未知错误，删除文件夹失败: {batch_folder}, 错误: {e}")
            
            return
            
        # 如果不是批次删除，只删除文件，只有当文件夹为空才会删除
        logger.debug(f"正在删除文件: {file_path}")
        file_obj = Path(file_path)
        
        if not file_obj.exists():
            logger.info(f"文件 {file_path} 不存在, 跳过删除")
            return
        
        # 记录batch文件夹路径
        batch_folder = file_obj.parent
        batch_folders_to_check.add(batch_folder)
        
        try:
            file_obj.unlink()  # 删除文件
            logger.debug(f"成功删除文件: {file_path}")
        except Exception as e:
            logger.error(f"删除文件 {file_path} 失败: {e}")
        
        # 检查并删除空的batch文件夹
        for batch_folder in batch_folders_to_check:
            try:
                if batch_folder.exists() and batch_folder.is_dir():
                    # 检查文件夹是否为空
                    if not any(batch_folder.iterdir()):
                        batch_folder.rmdir()
                        logger.debug(f"成功删除空的batch文件夹: {batch_folder}")
            except FileNotFoundError:
                # 在多线程环境下，文件夹可能已被其他线程删除，这是正常情况
                logger.debug(f"batch文件夹 {batch_folder} 已被其他线程删除")
            except Exception as e:
                # 其他异常情况需要记录警告
                logger.warning(f"删除batch文件夹 {batch_folder} 失败: {e}")
      
    async def _delete_file_in_thread(self, file_paths: list[str], is_batch: bool):
        """异步线程池方式并行删除文件
        
        Args:
            file_paths: 要删除的完整文件名列表
            is_batch: 是否是批次文件夹，如果是则删除整个批次文件夹，否则只删除文件
        """
        file_params = [{"file_path": file_path, "is_batch": is_batch} for file_path in file_paths]
        # 正确等待异步生成器
        async for _ in run_in_thread_pool(func=self._delete_file, params=file_params, workers=self.upload_workers):
            pass

    async def delete_file_service(
        self,
        kb_id: int, 
        batch: str | None = None,
        file_ids: list[str] | None = None
    ):
        """
        统一文件删除服务，处理多种删除场景
        
        Args:
            kb_id: 知识库ID
            batch: 批次名称(用于批次删除)
            file_ids: 文件ID列表(用于特定文件删除)
        """
        async with self.db_session as session:
            file_repo = FileRepository(session)
            kb_repo = KBRepository(session)
            chunk_repo = TxtChunkRepository(session)

            if batch or file_ids:
                logger.info(f"开始删除知识库 {kb_id} 中的文件")

                file_id_path_pairs = []
            
                # 1. 获取文件路径
                try:
                    file_id_path_pairs = await file_repo.get_file_id_path(kb_id, file_ids, [batch] if batch else None)
                
                    logger.info(f"开始删除文件，共 {len(file_id_path_pairs)} 个文件...")

                    # 2. 获取状态为已解析的文件ID
                    parsed_file_ids = [file_id for file_id, _, status in file_id_path_pairs 
                                    if status == FileStatus.PARSED.value or status == FileStatus.ARCHIVED.value]
                    
                    if len(parsed_file_ids) == 0:
                        logger.info("没有已解析的文件需要删除向量数据")
                    else:
                        # 3. 删除向量数据（仅删除已解析文件的向量数据）
                        await chunk_repo.delete_by_file_ids(parsed_file_ids)
                    # 获取所有需要删除的文件ID和路径
                    all_file_ids = []
                    all_file_paths = []

                    for file_id, path, _ in file_id_path_pairs:
                        all_file_ids.append(file_id)
                        all_file_paths.append(path)

                    # 4. 删除文件元数据
                    await file_repo.delete(kb_id, all_file_ids)

                except DataNotFoundException as e:
                    logger.info(e.message)
                except DatabaseException as e:
                    logger.error(e.original_error or e.message)
                    raise InternalServerError(e.message)
                except Exception as e:
                    msg = f"删除文件元数据失败: {e}"
                    logger.exception(msg)
                    raise InternalServerError(msg)
            
                # 4. 物理删除文件
                is_batch = batch is not None # 是否是批次删除
                try:
                    await self._delete_file_in_thread(all_file_paths, is_batch)
                except Exception as e:
                    logger.error(f"物理删除文件失败: {e}")
                    raise InternalServerError(f"物理删除文件失败: {e}")
            
            else:
                logger.info(f"开始删除知识库 {kb_id} 中的所有文件")
                # 1. 删除知识库中的所有文件的向量数据
                await chunk_repo.delete_by_kb_id(kb_id)
                # 2. 删除知识库中的所有文件的元数据
                await file_repo.delete(kb_id, None)
                # 3. 删除知识库中的所有的物理文件
                
                # 获取知识库和业务域的名称用于构造物理删除路径
                try:
                    kb_name = await kb_repo.get_name_by_id(kb_id)
                    
                except DataNotFoundException as e:
                    raise NotFoundError(e.message)
                except DatabaseException as e:
                    logger.error(e.original_error or e.message)
                    raise InternalServerError(e.message)
                
                root_path = Path(self.file_storage).resolve()  # 转换为绝对路径
                target_path = root_path / Path(str(kb_id))
                
                try:
                    shutil.rmtree(target_path)
                    logger.info(f"成功删除知识库 {kb_id} 的物理目录: {target_path}")
                except FileNotFoundError:
                    # 目录不存在是正常情况，继续执行
                    logger.info(f"知识库 {kb_id} 的物理目录不存在，跳过删除: {target_path}")
                except Exception as e:
                    logger.error(f"删除知识库 {kb_id} 的物理目录失败: {e}")
                    raise InternalServerError(f"删除知识库 {kb_id} 的物理目录失败: {e}")
                
                logger.info(f"知识库 {kb_id} 中的所有文件删除成功")


###############################################################################
# 其他文件相关操作
###############################################################################


    async def update_file_tag(self, file_id: str, kb_id: int, tags: list[str]):
        """更新知识库文件的标签"""
        async with self.db_session as session:
            file_repo = FileRepository(session)
            chunk_repo = TxtChunkRepository(session)
            try:
                # 1. 更新文件标签
                await file_repo.update_tags(file_id=file_id, tags=tags)
                logger.info(f"文件 {file_id} 标签已更新为 {tags}")
                # 2. 更新文件对应的chunk标签
                await chunk_repo.update_tags(file_id=file_id, tags=tags)
                logger.info(f"文件 {file_id} 对应的chunk标签已更新为 {tags}")

            except Exception as e:
                handle_exception(e, "更新文件标签失败")
        
    # async def approve_file(self, file_ids: list[str], username: str, is_approve: bool = True, comments: str | None = None):
    #     """
    #     审批知识库中的指定文件
        
    #     Args:
    #         file_ids: list[str] 需要审批的文件ID列表
    #         username: str 审批人用户名
    #         is_approve: bool 是否审批通过，默认True
    #         comments: str | None 审批评论，默认None
    #     """
    #     async with self.db_session as session:
    #         file_repo = FileRepository(session)
    #         try:
    #             await file_repo.approve_files(
    #                 username=username,
    #                 file_ids=file_ids, 
    #                 is_approve=is_approve,
    #                 comments=comments
    #             )
    #             logger.info(f"文件审批 {file_ids} 已{'通过' if is_approve else '拒绝'}，审批评论：{comments}")

    #         except Exception as e:
    #             handle_exception(e, "文件审批失败")

    async def reparse_file(self, kb_id: int, file_ids: list[str]):
        """
        重新解析知识库中的指定文件
        
        将指定文件标记为未解析状态，触发重新解析流程
        
        Args:
            kb_id: int 知识库ID
            file_ids: list[str] 需要重新解析的文件ID列表
        """
        async with self.db_session as session:
            file_repo = FileRepository(session)
            chunk_repo = TxtChunkRepository(session)
            try:
                # 1. 删除文件对应的文本片段数据
                try:
                    await chunk_repo.delete_by_file_ids(file_ids=file_ids)
                except DataNotFoundException as e:
                    logger.debug(f"文件 {file_ids} 对应的文本片段数据不存在，跳过删除")
                logger.info(f"文件 {file_ids} 对应的文本片段数据已删除")
                # 2. 重置文件状态为未解析
                await file_repo.update_file_status(
                    file_ids=file_ids, 
                    status=FileStatus.APPROVED, 
                    log_msg="重新解析文件"
                )
                logger.info(f"文件 {file_ids} 已标记为待重新解析状态")

            except Exception as e:
                handle_exception(e, "重新解析文件失败")

    async def get_file_in_kb(self, kb_id: int) -> list[dict]:
        """获取知识库文件的详细信息

        Args:
            kb_id: int 知识库ID
            file_id: str | None 文件ID，默认None表示获取知识库下所有文件
        """
        async with self.db_session as session:
            file_repo = FileRepository(session)
            try:
                files = await file_repo.get_all()
              
            except Exception as e:
                handle_exception(e, "获取知识库文件失败")
        
        return [file.to_dict() for file in files]
    
    async def modify_parser_config(self, file_id: str, parser_params: dict[str, Any]):
        """更新文件解析配置"""
        async with self.db_session as session:
            file_repo = FileRepository(session)
            try:
                await file_repo.update_parser_params(
                    file_id=file_id, 
                    parser_params=parser_params
                )
                logger.info(f"文件 {file_id} 解析配置已更新")
            except Exception as e:
                handle_exception(e, "更新文件解析配置失败")

    async def get_file_names_by_ids(self, file_ids: list[str]) -> list[tuple[str, str]]:
        """获取知识库文件的名称"""
        async with self.db_session as session:
            file_repo = FileRepository(session)
            try:
                files = await file_repo.get_file_ids(file_ids)
                return files
            except Exception as e:
                handle_exception(e, "获取知识库文件名称失败")

    async def get_batch_by_id(self, file_id: str) -> str:
        """获取文件所属的批次"""
        async with self.db_session as session:
            file_repo = FileRepository(session)
            try:
                file = await file_repo.get(file_id)
                return file.batch
            except Exception as e:
                handle_exception(e, "获取文件所属批次失败")

    async def get_path_by_id(self, file_id: str) -> str:
        """根据ID获取文件路径"""
        async with self.db_session as session:
            file_repo = FileRepository(session)
            try:
                file = await file_repo.get(file_id)
                return file.file_path
            except Exception as e:
                handle_exception(e, "根据ID获取文件路径出错：{e}")