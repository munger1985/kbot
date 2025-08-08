
import os
import aiofiles
from docx import Document
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository

async def get_file(file_id: str):
    file = await KbotMdKbFilesRepository().get_by_id(file_id)
    if not file:
        return
    path = file.file_path
    if not path:
        return

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    async with aiofiles.open(path, 'rb') as file:
        content = await file.read()
    
    # Check file type and get page count
    # page_count = None
    # if path.lower().endswith('.pdf'):
    #     with open(path, 'rb') as f:
    #         reader = PdfReader(f)
    #         page_count = len(reader.pages)
    # elif path.lower().endswith('.docx'):
    #     doc = Document(path)
    #     page_count = len(doc.paragraphs)  # Note: This is a rough estimate for Word documents
    
    # return {
    #     'content': content,
    #     'page_count': page_count
    # }


