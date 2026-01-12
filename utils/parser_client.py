"""文档解析微服务调用类。

集成文档解析微服务的远程调用逻辑。
"""
import os
import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

from core.config.settings import get_parser_config
from core.exceptions import InternalServerError


class CallParser:
    """文档解析微服务调用类。"""

    def __init__(self):
        """初始化配置。"""
        self.parser_config = get_parser_config()

    async def call_doc_parser_service(
        self, 
        file_path: str,
        in_memory: bool = False,
        file_content: str | None = None,
        output_format: str = "chunks",
        do_ocr: bool = False,
        ocr_engine: str | None = None,
        generate_picture_images: bool = False,
        images_scale: float = 2.0,
        use_vlm: bool = False,
        vlm_model: str | None = None,
        vlm_prompt: str | None = None,
        chunk_size: int = 512,
        overlap: int = 50,
        min_chunk_len: int = 10
    ) -> str | list[str]:
        """调用文档解析微服务（平铺参数版）。

        Args:
            file_path: 待上传的本地文件路径。
            in_memory: 是否在内存中处理文件内容，默认 False。
            file_content: 待解析的文件内容，根据 in_memory 参数判断。
            output_format: 输出格式 (markdown, html, json, chunks)。
            do_ocr: 是否开启 OCR。
            ocr_engine: OCR 引擎 (easyocr, tesseract, paddle)。
            generate_picture_images: 是否生成图片副本。
            images_scale: 图片缩放比例。
            use_vlm: 是否开启 VLM 增强。
            vlm_model: 指定 VLM 模型。
            vlm_prompt: 自定义 VLM 提示词。
            chunk_size: 切分窗口大小。
            overlap: 切分窗口重叠部分。
            min_chunk_len: 最小切分长度。

        Returns:
            str | list[str]: 解析结果。
        """
        service_host = self.parser_config.service_host
        service_port = self.parser_config.service_port
        
        # 超时设置
        total_timeout = self.parser_config.timeout
        timeout = aiohttp.ClientTimeout(total=total_timeout)
        
        url = f"http://{service_host}:{service_port}/v1/parse/file"

        # 1. 构造 Multipart 报文
        data = aiohttp.FormData()
        
        # 填充解析控制参数
        kwargs = {
            "output_format": output_format,
            "do_ocr": do_ocr,
            "ocr_engine": ocr_engine,
            "generate_picture_images": generate_picture_images,
            "images_scale": images_scale,
            "use_vlm": use_vlm,
            "vlm_model": vlm_model,
            "vlm_prompt": vlm_prompt,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "min_chunk_len": min_chunk_len,
        }
        for k, v in kwargs.items():
            data.add_field(k, v)

        # 2. 读取并添加文件流
        try:
            filename = os.path.basename(file_path)
            # 采用这种方式 aiohttp 会自动管理文件关闭
            if not in_memory:
                data.add_field('file', 
                               open(file_path, 'rb'), 
                               filename=filename, 
                               content_type='application/octet-stream')
            else:
                data.add_field('file', 
                               file_content, 
                               filename=filename, 
                               content_type='application/octet-stream')
        except Exception as e:
            logger.error(f"读取文件失败: {filename}, error: {e}")
            raise InternalServerError(f"无法读取待解析文件: {e}")

        # 3. 发起请求
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, data=data) as response:
                    if response.status != 200:
                        err = await response.text()
                        raise InternalServerError(f"解析服务响应错误 {response.status}: {err}")
                    
                    res_json = await response.json()
                    if res_json.get("status") != "success":
                        raise InternalServerError(f"解析逻辑异常: {res_json.get('detail')}")
                    
                    return res_json.get("result")
                    
        except Exception as e:
            logger.exception("解析微服务调用异常")
            raise InternalServerError(f"解析微服务连接失败: {e}")