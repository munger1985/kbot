from pathlib import Path
from services.kb.kb_file_operator import KBFileOperator
from services.kb.kb_procedure import KBProcedure
from services.kb.kb_file_preview import FilePreview
from api.schemas.kb_schema import KBUploadForm, KBDeleteForm, KBReparseForm
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from utils.file_converter import FileToImage


async def upload_kb_files(
    form: KBUploadForm
) -> bool:
    """上传文件到知识库"""
    try:
        result = await KBFileOperator().upload_file_service(
            files=form.files,
            app_id=form.app_id,
            domain_id=form.domain_id,
            kb_id=form.kb_id,
            overwrite=form.overwrite,
            batch_name=form.batch_name,
            batch_id=form.batch_id,
            biz_metadata=form.biz_metadata,
            created_by=form.created_by
        )
        return result
    except Exception as e:
        raise e

async def delete_kb_files(
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
        raise e
    
async def get_kb_files(
        file_id: str,
        download: bool = False,
        page_num: int = 0
) -> dict[str, str] | None:
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
        return None
    
    file_path = file.file_path
    file_name = file.file_name
    file_ext = file.file_ext

    if file_path is None:
        return None
    
    if file_name is None:
        file_name = Path(file_path).name
    
    if file_ext is None:
        file_ext = Path(file_path).suffix
    
    try:
        if download:
            # 下载文件
            
            return {"file_path": file_path, "file_name": file_name, "file_ext": file_ext}
        else:
            if file_ext == ".txt":
                # 预览文本文件
                return {"file_path": file_path, "file_name": file_name, "file_ext": file_ext}
            else:
                # 预览其他文件
                img = FileToImage()
                try:
                    img_path = await img.convert_to_image(input_path=file_path, page_num=page_num)
                    return {"file_path": img_path, "file_name": file_name, "file_ext": ".png"}
                except Exception as e:
                    raise e

    except Exception as e:
        raise e
    
async def reparse_kb_files(
    form: KBReparseForm
) -> bool:
    """重新解析知识库文件"""
    try:
        kbproc = KBProcedure()
        result = await kbproc.reparse_files(kb_id=form.kb_id, file_ids=form.file_ids)
        return result
    except Exception as e:
        raise e
    
async def preview_kb_file(
    file_id: str,
    max_text_length: int = 500,
    max_pages: int = 2,
    max_sheets: int = 2,
    max_slides: int = 2,
    pdf_pages: int | list[int] | None = None,
    word_page: int | None = None,
    sheet_index: int = 0,
    start_index: int = 0,
    slide: int | None = None
) -> dict | None:
    """预览知识库文件"""
    try:
        preview_service = FilePreview()
        result = await preview_service.get_preview(
            file_id=file_id,
            max_text_length=max_text_length,
            max_pages=max_pages,
            max_sheets=max_sheets,
            max_slides=max_slides,
            pdf_pages=pdf_pages,
            word_page=word_page,
            sheet_index=sheet_index,
            start_index=start_index,
            slide=slide
        )
        return result
    except Exception as e:
        raise e

async def edit_kb_file_chunk(
    kb_id: int,
    file_id: str,
    embed_id: str,
    new_chunk: str,
) -> bool:
    """编辑知识库文件的分片内容，并更新分片的向量"""
    try:
        result = await KBFileOperator().edit_file_chunk(
            kb_id=kb_id,
            file_id=file_id,
            embed_id=embed_id,
            new_chunk=new_chunk
        )
        return result
    except Exception as e:
        raise e
    
async def delete_kb_file_chunk(
    kb_id: int,
    file_id: str,
    embed_id: str,
) -> bool:
    """删除知识库文件的分片内容，并更新分片的向量"""
    try:
        result = await KBFileOperator().delete_file_chunk(
            kb_id=kb_id,
            file_id=file_id,
            embed_id=embed_id
        )
        return result
    except Exception as e:
        raise e