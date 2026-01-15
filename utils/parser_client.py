"""文档解析微服务调用类。

集成文档解析微服务的远程调用逻辑。
"""
import os
import aiohttp
from loguru import logger
from api.schemas.parser_schema import ParserParams

from core.config.settings import get_parser_config, get_prompt_config
from core.exceptions import InternalServerError
from services.dataparse.txt_to_md import TxtToMarkdownParser
from services.dataparse.parser_common import ParserCommonMethods


class CallParser:
    """文档解析微服务调用类。"""

    def __init__(self):
        """初始化配置。"""
        self.parser_config = get_parser_config()
        self.common = ParserCommonMethods()

    async def call_doc_parser_service(
        self, 
        file_path: str,
        parser_params: ParserParams,
        file_content: str | None = None,
        output_format: str = "chunks"
    ) -> str | list[dict]:
        """调用文档解析微服务。

        Args:
            file_path: 待上传的本地文件路径。
            parser_params: 解析参数对象。
            file_content: 待解析的文件内容，如果有则表示直接解析内容，否则从文件路径读取。
            output_format: 输出格式 (markdown, html, json, chunks)。

        Returns:
            str | list[dict]: 解析结果。
        """
        service_host = self.parser_config.service_host
        service_port = self.parser_config.service_port
        
        # 超时设置
        total_timeout = self.parser_config.timeout
        timeout = aiohttp.ClientTimeout(total=total_timeout)
        
        url = f"http://{service_host}:{service_port}/v1/parse/file"

        # 1. 构造 Multipart 报文
        data = aiohttp.FormData()

        # 处理文件内容
        if file_path.endswith(".txt"):
            # 因为docling不支持直接解析txt文件，所以先转换为md
            file_content = TxtToMarkdownParser().process(file_path)
            file_path = file_path.replace(".txt", ".md")
            in_memory = True
        else:
            in_memory = False

        # 获取 VLM 提示词
        prompt_content = None
        prompt_unique_name = parser_params.vlm_prompt
        if parser_params.use_vlm:
            if prompt_unique_name:
                prompt_content = await self.common.get_prompt_content(prompt_unique_name=prompt_unique_name)
            else:
                prompt_name = get_prompt_config().image2text
                prompt_content = await self.common.get_prompt_content(prompt_unique_name=prompt_name)


        # 填充解析控制参数
        kwargs = parser_params.model_dump()
        kwargs["output_format"] = output_format
        kwargs["vlm_prompt"] = prompt_content

        # 在循环中添加类型检查和转换
        for k, v in kwargs.items():
            if v is not None:
                # 根据类型进行适当的转换
                if isinstance(v, (int, float)):
                    data.add_field(k, str(v))
                elif isinstance(v, bool):
                    data.add_field(k, str(v).lower())  # 布尔值转为小写字符串
                else:
                    data.add_field(k, v)  # 字符串和其他类型直接添加

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