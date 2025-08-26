import os
import sys
import base64
import asyncio
# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
from utils.file_converter import FileToImage

async def test():
    """测试示例"""
    converter = FileToImage()
    
    # 转换Word文档
    try:
        word = await converter.convert_to_image("/mnt/f/docs/resume.docx")
        # 确保目标目录存在
        os.makedirs("/mnt/f/docs/temp", exist_ok=True)
        for page in word:
            # 解析页码
            page_number = await page['page'] if asyncio.iscoroutine(page['page']) else page['page']
            # 保存图片到文件
            image_path = os.path.join("/mnt/f/docs/temp", f"page_{page_number}.png")
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(page['image']))
            print(f"转换成功: {page_number} -> 图片已保存到 {image_path}")
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