from services.kb.kb_upload import upload_file_service
from services.kb.kb_delete import delete_file_service
from services.kb.kb_procedure import KBProcedure
from api.schemas.kb_schema import KBUploadForm, KBDeleteForm, KBReparseForm
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from utils.file_converter import FileToImage


async def upload_kb_files(
    form: KBUploadForm
) -> bool:
    """
    Upload files to the knowledge base.
    上传文件到知识库
    """
    try:
        result = await upload_file_service(
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
    """
    Delete files from the knowledge base.
    从知识库中删除文件
    """
    try:
        result = await delete_file_service(
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
) -> tuple[str, str | None] | str | None:
    """
    Get file content for download or preview.
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

    if file_path is None:
        return None
    
    try:
        if download:
            # 下载文件
            
            return file_path, file_name
        else:
            # 预览文件
            img = FileToImage()
            try:
                return await img.convert_to_image(input_path=file_path, page_num=page_num)
            except Exception as e:
                raise e

    except Exception as e:
        raise e
    
async def reparse_kb_files(
    form: KBReparseForm
) -> bool:
    """
    Re-parse files for the knowledge base.
    重新解析知识库文件
    """
    try:
        kbproc = KBProcedure()
        result = await kbproc.reparse_files(kb_id=form.kb_id, files=form.files)
        return result
    except Exception as e:
        raise e
    
