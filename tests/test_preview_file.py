from pathlib import Path
import sys
import asyncio
# 添加项目根目录到 Python 路径，确保可以导入项目模块
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from services.kb.kb_file_preview import FilePreview

class TestFilePreview:

    def __init__(self):
        self.preview_service = FilePreview()

    async def test_pdf(self):        
        # 测试 PDF 文件预览
        pdf_preview = await self.preview_service.get_preview(
        file_id="cc2d1104-1d4f-46ff-88d0-c374099920de",
        max_pages=2,
        pdf_pages=1
        )
        print("PDF Preview:", pdf_preview)

    async def test_excel(self):        
        # 测试 Excel 文件预览  
        excel_preview = await self.preview_service.get_preview(
            file_id="cdd43694-c209-4b61-bc53-fcff95627934",
            max_sheets=1,
            sheet_index=0
        )
        print("Excel Preview:", excel_preview)
    
    async def test_ppt(self):
        # 测试 PPT 文件预览
        ppt_preview = await self.preview_service.get_preview(
            file_id="61066b0e-38dd-466d-8c2a-b19cce54a928",
            max_slides=2,
            # slide=1
        )
        print("PPT Preview:", ppt_preview)
        
    async def test_word(self):
        # 测试 Word 文件预览
        word_preview = await self.preview_service.get_preview(
            file_id="65e7bf8e-7766-4154-87ba-fa41a8d2a8fb",
            max_pages=2,
            word_page=1
        )
        print("Word Preview:", word_preview)
        
        
    async def test_text(self):
        # 测试文本文件预览
        text_preview = await self.preview_service.get_preview(
            file_id="e954cf90-bdc0-485c-b6ff-251908d611ea",
            max_text_length=500
        )
        print("Text Preview:", text_preview)

if __name__ == "__main__":
    asyncio.run(TestFilePreview().test_excel())