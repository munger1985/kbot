from pathlib import Path
from loguru import logger
from services.kb.kb_chunk_operator import KBChunkOperator
from services.kb.kb_file_operator import KBFileOperator
from services.kb.kb_procedure import KBProcedure
from services.kb.kb_file_preview import FilePreview
from api.schemas.kb_schema import *
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
# from utils.file_converter import FileToImage
from utils.common import detect_file_encoding
from core.exceptions import ValidationException, InternalServerError, ResourceNotFoundException
from utils.sanitize import sanitize_text_for_oracle_json


class KBController:
    """知识库控制器"""
    
    async def upload_kb_files(
            self,
            upload_form: KBUploadForm
        ) -> tuple[bool, str | None]:
        """上传文件到知识库"""
        try:
            result, error_msg = await KBFileOperator().upload_file_service(
                files=upload_form.files,
                app_id=upload_form.app_id,
                domain_id=upload_form.domain_id,
                kb_id=upload_form.kb_id,
                overwrite=upload_form.overwrite,
                batch_name=upload_form.batch_name,
                batch_id=upload_form.batch_id,
                biz_metadata=upload_form.biz_metadata,
                created_by=upload_form.created_by
            )
            return result, error_msg
        except Exception as e:
            msg = f"上传文件到知识库 {upload_form.kb_id} 失败: {str(e)}"
            logger.error(msg)
            raise InternalServerError(message=msg)

    async def delete_kb_files(
            self,
            form: KBDeleteForm
        ) -> dict:
        """从知识库中删除文件"""
        try:
            result = await KBFileOperator().delete_file_service(
                app_id=form.app_id,
                domain_id=form.domain_id,
                kb_id=form.kb_id,
                batch_id=form.batch_id,
                batch_name=form.batch_name,
                file_ids=form.file_ids,
                file_paths=form.file_paths,
            )
            return result
        except Exception as e:
            msg = f"从知识库 {form.kb_id} 删除文件 {form.file_ids} 失败: {str(e)}"
            logger.error(msg)
            raise InternalServerError(message=msg)
        
    async def get_kb_files(
            self,
            file_id: str,
            download: bool = False,
            page_num: int = 0
        ) -> dict | None:
        """
        获取文件内容用于下载或预览

        params:
        - file_id (str): 文件id
        - download (bool): 是否下载原文件，而不是预览
        - page_num (int | None): 当预览时，该参数指定预览的页数

        return:
        - tuple[str, str | None] | str | None: 返回文件下载的路径和文件名，或者当预览文件时，返回文件的临时路径，或者如果没有找到文件，返回 None
        """
        file = await KbotMdKbFilesRepository().get_by_id(file_id=file_id)
        if file is None:
            raise ResourceNotFoundException(f"文件 {file_id} 不存在")
        
        file_path = file.file_path
        file_name = file.file_name
        file_ext = file.file_ext

        if file_path is None:
            raise ResourceNotFoundException(f"文件 {file_id} 不存在")

        if file_name is None:
            file_name = Path(file_path).name
        
        if file_ext is None:
            file_ext = Path(file_path).suffix
        
        encoding = detect_file_encoding(file_path)
        
        try:
            if download:
                # 下载文件
                try:
                    # 确保文件路径和文件名以 UTF-8 编码处理
                    file_path = str(Path(file_path).resolve())
                    file_name = file_name.encode('utf-8').decode('utf-8') if file_name else Path(file_path).name
                    return {"file_path": file_path, "file_name": file_name, "file_ext": file_ext, "encoding": encoding}
                except Exception as e:
                    msg = f"下载文件 {file_id} 失败: {str(e)}"
                    logger.error(msg)
                    raise InternalServerError(message=msg)
            else:
                if file_ext == ".txt":
                    # 预览文本文件
                    return {"file_path": file_path, "file_name": file_name, "file_ext": file_ext, "encoding": encoding}
                elif file_ext in [".png", ".jpg", ".jpeg"]:
                    # 预览图片文件
                    return {"file_path": file_path, "file_name": file_name, "file_ext": file_ext}
                else:
                    # TODO: 预览其他文件
                    pass

        except Exception as e:
            msg = f"获取文件 {file_id} 失败: {str(e)}"
            logger.error(msg)
            raise InternalServerError(message=msg)
        
    async def reparse_kb_files(self, form: KBReparseForm) -> bool:
        """重新解析知识库文件"""
        try:
            kbproc = KBProcedure()
            result = await kbproc.reparse_files(kb_id=form.kb_id, file_ids=form.file_ids)
            return result
        except Exception as e:
            msg = f"重新解析知识库 {form.kb_id} 文件 {form.file_ids} 失败: {str(e)}"
            logger.error(msg)
            raise InternalServerError(message=msg)
        
    async def preview_kb_file(self,
                            file_id: str, 
                            max_length: int = 10000, 
                            pages: int | list[int] | None = None, 
                            sheet_index: int = 0, 
                            preview_rows: int = 20,
                            slide: int | None = None) -> dict | None:
        """预览知识库文件"""
        try:
            preview_service = FilePreview()
            result = await preview_service.get_preview(
                file_id = file_id,
                max_length = max_length,
                pages = pages,
                sheet_index = sheet_index,
                preview_rows = preview_rows,
                slide = slide
            )
            return result
        except Exception as e:
            msg = f"预览文件 {file_id} 失败: {str(e)}"
            logger.error(msg)
            raise InternalServerError(message=msg)

    async def edit_kb_file_chunk(
            self,
            kb_id: int,
            file_id: str,
            embed_id: str,
            new_chunk: str,
        ) -> bool:
        """编辑知识库文件的分片内容，并更新分片的向量"""
        try:
            result = await KBChunkOperator().edit_file_chunk(
                kb_id=kb_id,
                file_id=file_id,
                embed_id=embed_id,
                new_chunk=new_chunk
            )
            return result
        except Exception as e:
            msg = f"编辑文件 {file_id} 分片 {embed_id} 失败: {str(e)}"
            logger.error(msg)
            raise InternalServerError(message=msg)
        
    async def delete_kb_file_chunk(
            self,
            kb_id: int,
            file_id: str,
            embed_id: str,
        ) -> bool:
        """删除知识库文件的分片内容，并更新分片的向量"""
        try:
            result = await KBChunkOperator().delete_file_chunk(
                kb_id=kb_id,
                file_id=file_id,
                embed_id=embed_id
            )
            return result
        except Exception as e:
            msg = f"删除文件 {file_id} 分片 {embed_id} 失败: {str(e)}"
            logger.error(msg)
            raise InternalServerError(message=msg)
        
    async def toogle_kb_file_chunk_status(
            self,
            kb_id: int,
            chunk_id: str,
            status: int
        ) -> bool:
        """切换知识库文件的分片状态"""
        try:
            result = await KBChunkOperator().toogle_file_chunk_status(
                kb_id=kb_id,
                chunk_id=chunk_id,
                status=status
            )
            return result
        except Exception as e:
            msg = f"切换知识库 {kb_id} 中的分片 {chunk_id} 状态失败: {str(e)}"
            logger.error(msg)
            raise InternalServerError(message=msg)
        
    async def get_kb_file_chunk_by_id(
            self,
            kb_id: int,
            file_id: str
        ) -> list[dict]:
        """获取知识库文件的分片"""
        try:
            result = await KBChunkOperator().get_chunks_by_file_id(kb_id=kb_id, file_id=file_id)
            return result
        except Exception as e:
            msg = f"获取知识库 {kb_id} 文件 {file_id} 分片失败: {str(e)}"
            logger.error(msg)
            raise InternalServerError(message=msg)
        
    async def update_kb_file_chunk_description(
            self,
            kb_id: int,
            embed_id: str,
            description: str
        ) -> bool:
        """更新知识库文件的分片描述"""
        # 1. 清理描述中的特殊字符
        description = sanitize_text_for_oracle_json(description, max_length=4000)
        
        try:
            result = await KBChunkOperator().update_chunk_description(
                kb_id=kb_id,
                embed_id=embed_id,
                description=description
            )
            return result
        except Exception as e:
            msg = f"更新知识库 {kb_id} 文件分片 {embed_id} 描述失败: {str(e)}"
            logger.error(msg)
            raise InternalServerError(message=msg)
        
    async def update_kb_file_chunk_tags(
            self,
            kb_id: int,
            file_id: str,
            tags: list[str]
        ) -> bool:
        """更新知识库文件的分片标签"""
        try:
            # 1. 更新文件的标签
            file_result = await KBFileOperator().update_file_tags(
                file_id=file_id,
                tags=tags
            )
            if not file_result:
                logger.warning(f"文件 {file_id} 文件标签更新失败")

            # 2. 更新分片标签
            chunk_result = await KBChunkOperator().update_chunk_tags(
                kb_id=kb_id,
                file_id=file_id,
                tags=tags
            )
            if not chunk_result:
                logger.warning(f"文件 {file_id} 分片标签更新失败")

            return file_result or chunk_result
        
        except Exception as e:
            msg = f"更新知识库 {kb_id} 文件 {file_id} 分片标签失败: {str(e)}"
            logger.error(msg)
            raise InternalServerError(message=msg)
        
kb_controller = KBController()