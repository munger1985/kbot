import os
import sys
import asyncio
# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
from utils.file_converter import OfficeToPDF

async def test():
    """测试示例"""
    converter = OfficeToPDF()
    
    # 转换Word文档
    try:
        pdf_path = await converter.convert_to_pdf("/mnt/f/docs/resume.docx")
        print(f"转换成功: {pdf_path}")
    except Exception as e:
        print(f"转换失败: {e}")
    
    # # 转换PPT文档
    # try:
    #     pdf_path = await converter.convert_to_pdf("/mnt/f/docs/langchain_and_dify_diff.pptx", output_path="/mnt/f/")
    #     print(f"转换成功: {pdf_path}")
    # except Exception as e:
    #     print(f"转换失败: {e}")

if __name__ == "__main__":
    asyncio.run(test())