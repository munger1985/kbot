import asyncio
import io
import hashlib
from PIL import Image
from loguru import logger
from docling_core.types.doc.document import PictureItem
from utils.clients import AIModelClient

class ParserToolLib:
    @staticmethod
    def get_image_hash(pic_item: PictureItem):
        # 1. 访问内部的 pil_image
        pic = pic_item.image
        if not pic:
            return None
        pil_img = pic.pil_image
        if not pil_img:
            return None
        
        # 2. 将 Pillow 对象转为 bytes 
        img_byte_arr = io.BytesIO()
        # 建议保存为特定格式以保证 Hash 稳定性，比如 PNG
        pil_img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # 3. 计算 MD5
        return hashlib.md5(img_bytes).hexdigest()

    @staticmethod
    def ensure_markdown_table_integrity(table_md: str) -> str:
        """强制修复 PDF 表格缺失分隔行的问题"""
        lines = [line.strip() for line in table_md.strip().split('\n') if line.strip()]

        if len(lines) < 1 or "|" not in lines[0]:
            return table_md
            
        # 如果第二行不是分隔行 (---|---|---)
        if len(lines) >= 1 and (len(lines) < 2 or "---" not in lines[1]):
            # 根据第一行的列数生成分隔行
            col_count = lines[0].count('|') - 1
            if col_count > 0:
                separator = "|" + " --- |" * col_count
                lines.insert(1, separator)
                
        return "\n" + "\n".join(lines) + "\n"

class ModelTask:
    """
    模型任务类
    1. 调用VLM解析图片
    2. 调用DS OCR识别文字
    3. 调用LLM提取全文摘要或文本语义
    """
    def __init__(self):
        self.llm_semaphore = asyncio.Semaphore(2)  # LLM API调用限流（最多2个并发）
        self.vlm_semaphore = asyncio.Semaphore(2)  # VLM API调用限流（最多2个并发）
        self.dsocr_semaphore = asyncio.Semaphore(2)  # DS OCR API调用限流（最多2个并发）

    async def vlm_task(self, client: AIModelClient, model_name: str, prompt: str, index: str, image_obj) -> tuple:
        """VLM 任务，用于处理图片逻辑"""
        async with self.vlm_semaphore:
            try:
                if image_obj:
                    # --- 智能压缩逻辑开始 ---
                    # 1. 设定最大长边限制（QwenVL 建议在 1000 左右平衡度最好）
                    max_size = 1024 
                    w, h = image_obj.size
                    
                    if max(w, h) > max_size:
                        scale = max_size / max(w, h)
                        new_size = (int(w * scale), int(h * scale))
                        # # 使用 LANCZOS 算法保证缩放后的边缘平滑，利于文字识别
                        # image_obj = image_obj.resize(new_size, Image.Resampling.LANCZOS)
                        # 换成 BICUBIC 兼顾速度与质量
                        image_obj = image_obj.resize(new_size, Image.Resampling.BICUBIC)
                        logger.debug(f"VLM 图片缩放: {w}x{h} -> {new_size}")

                    # 2. 如果是识别标题层级，可以转为 RGB 减少调色板干扰（Docling 有时返回 P 模式）
                    if image_obj.mode != "RGB":
                        image_obj = image_obj.convert("RGB")
                    
                    # 3. 内存压缩：转为低质量 JPEG 减少传输 Token 和带宽
                    # 这步能显著降低 API 端的解析延迟
                    from io import BytesIO
                    buffered = BytesIO()
                    image_obj.save(buffered, format="JPEG", quality=80, optimize=True)
                    # 这里的 image_obj 变成了压缩后的字节流或新的 Image 对象

                res = await client.get_vlm_answer(
                    model_name=model_name, 
                    image=image_obj, 
                    prompt=prompt, 
                    stream=True
                )
                return index, res
            except Exception as e:
                logger.error(f"VLM处理失败 (Index {index}): {e}")
                return index, None

    async def dsocr_task(self, client: AIModelClient, model_name: str, prompt: str, index: str, image_obj) -> tuple:
        """DS OCR 任务，用于高精度图片文字识别"""
        async with self.dsocr_semaphore:
            try:
                if image_obj:
                    max_size = 1024
                    w, h = image_obj.size
                    if max(w, h) > max_size:
                        scale = max_size / max(w, h)
                        new_size = (int(w * scale), int(h * scale))
                        image_obj = image_obj.resize(new_size, Image.Resampling.BICUBIC)
                    if image_obj.mode != "RGB":
                        image_obj = image_obj.convert("RGB")

                res = await client.get_dsocr_answer(
                    model_name=model_name,
                    image=image_obj,
                    prompt=prompt,
                    stream=True
                )
                return index, res
            except Exception as e:
                logger.error(f"DS OCR 处理失败 (Index {index}): {e}")
                return index, None

    async def llm_task(self, client: AIModelClient, model_name: str, prompt: str) -> str | None:
        """LLM 任务，用于处理全文摘要"""
        try:
            async with self.llm_semaphore:
                return await client.get_llm_answer(
                    model_name=model_name, 
                    prompt=prompt,
                    stream=True  # 即使是内部聚合，开启 stream 也能更早释放资源
                )
        except Exception as e:
            logger.error(f"LLM 任务执行失败: {str(e)}")
            return None
        
    async def llm_json_task(self, client: AIModelClient, model_name: str, prompt: str) -> dict | None:
        """LLM 任务，用于处理文本语义并返回 JSON"""
        try:
            async with self.llm_semaphore:
                return await client.get_llm_json(
                    model_name=model_name, 
                    prompt=prompt,
                    stream=True  # 即使是内部聚合，开启 stream 也能更早释放资源
                )
        except Exception as e:
            logger.error(f"LLM 任务执行失败: {str(e)}")
            return None