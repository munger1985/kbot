import os
import shutil
import aiohttp
from tempfile import mkdtemp
from pdf2image import convert_from_path
from pathlib import Path
from loguru import logger
from core.config.settings import get_libre_config


class OfficeToPDF:
    """Office文档转PDF工具类"""
    
    def __init__(self):
        
        # 支持的文件扩展名
        self.supported_extensions = ['.ppt', '.pptx', '.doc', '.docx', '.odt', '.ods']

        libre_config = get_libre_config()
        libre_host = libre_config.host
        libre_port = libre_config.port
        
        # 调用微服务接口
        self.url = f"http://{libre_host}:{libre_port}/convert"

    
    async def convert_to_pdf(self, input_path: str, output_path: str | None = None, page: int | None = None) -> str | None:
        """
        转换文档为PDF格式
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径（可选）
            page: 指定页码（可选）
            
        Returns:
            str: 转换后的PDF文件路径或None（失败时）
            
        Raises:
            FileNotFoundError: 输入文件不存在时抛出
            ValueError: 不支持的文件格式时抛出
            RuntimeError: 转换失败时抛出
        """
        input_file = Path(input_path)
        
        if not input_file.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
            
        if input_file.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"不支持的文件格式: {input_file.suffix}")
            
        # 处理输出路径
        if output_path is None:
            output_file = input_file.with_suffix('.pdf')
        else:
            output_file = Path(output_path)
            
            # 如果输出路径是目录，则在该目录下创建同名PDF文件
            if output_file.is_dir():
                output_file = output_file / input_file.with_suffix('.pdf').name
        
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 临时输出到目标目录，使用输入文件名
            temp_file = output_file.parent / input_file.with_suffix('.pdf').name

            
            # 准备表单数据
            try:
                file_obj = open(input_path, 'rb')

                data = aiohttp.FormData()
                data.add_field(
                    'file', 
                    file_obj,
                    filename=input_file.name,
                    content_type='application/octet-stream'
                )
                
                if page is not None:
                    data.add_field('page', str(page))
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(self.url, data=data) as response:
                            if response.status != 200:
                                error = await response.text()
                                logger.error(f"文档转换失败: HTTP {response.status}, {error}")
                                return None
                            
                            # 提取转换结果并写入临时文件
                            pdf_content  = await response.read()
                            
                            # 检查是否确实收到了PDF内容
                            if not pdf_content:
                                logger.error("接收到空的PDF内容")
                                return None
                                
                            # 检查内容类型是否为PDF
                            content_type = response.headers.get('Content-Type', '')
                            if 'application/pdf' not in content_type:
                                logger.error(f"意外的内容类型: {content_type}")
                                return None
                            
                            # 写入临时文件
                            with open(temp_file, 'wb') as f:
                                f.write(pdf_content)
                            
                            logger.info("成功获取转换后的文档")
                            
                except Exception as e:
                    logger.error(f"LibreOffice服务发生错误: {str(e)}")
                    return None
            finally:
                file_obj.close() # 确保文件最终被关闭

            # 重命名文件到用户指定的名称
            try:
                shutil.move(str(temp_file), str(output_file))
            except Exception as e:
                logger.error(f"文件重命名失败: {e}")
                raise

            # 验证输出文件
            if not output_file.exists():
                # 尝试处理可能的文件名编码问题
                alt_path = Path(output_file.parent) / (output_file.stem + '.pdf')
                if alt_path.exists():
                    return str(alt_path)
                raise RuntimeError("转换成功但输出文件未创建")
                
            return str(output_file)
            
        except Exception as e:
            raise RuntimeError(f"文档转换失败: {str(e)}")

        

class FileToImage:
    """文件转图片工具类"""

    def __init__(self):
    
        # 支持的文件扩展名
        self.supported_extensions = ['.ppt', '.pptx', '.doc', '.docx', '.txt', '.pdf']

    async def convert_to_image(self, input_path: str, page_num: int) -> str:
        """
        将文档转换为图片
        
        Args:
            input_path: 输入文件路径
            page_num: 页码
            
        Returns:
            str: 临时图片文件路径
            
        Raises:
            FileNotFoundError: 输入文件不存在时抛出
            ValueError: 不支持的文件格式时抛出
        """
        file_path = Path(input_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
            
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        # 根据不同的文件类型进行处理
        file_extension = file_path.suffix.lower()
        temp_dir = mkdtemp()

        if file_extension == '.pdf':
            # 使用 pdf2image 将 PDF 文件转换为图片
            try:
                images = []
                pdf_images = convert_from_path(file_path, first_page=page_num, last_page=page_num)
                image = pdf_images[0]
                
                img_path = os.path.join(temp_dir, "output.img")
                image.save(img_path, format="PNG")
                return img_path
                
            except Exception as e:
                logger.exception(f"PDF文件转换失败: {str(e)}")
                raise e
        
        elif file_extension in ['.ppt', '.pptx', '.doc', '.docx']:
            # 使用LibreOffice先将Office文档转换为PDF，再转换为图片
    
            try:
                # 第一步：将文件转为PDF
                pdf_path = os.path.join(temp_dir, "output.pdf")
                await OfficeToPDF().convert_to_pdf(input_path=str(file_path), output_path=pdf_path)
                
                # 第二步：将PDF转为图片
                return await self.convert_to_image(pdf_path, page_num)
            except Exception as e:
                logger.exception(f"Office文件转换失败: {str(e)}")
                raise e

        else:
            raise ValueError("不支持的文件类型进行图片转换")