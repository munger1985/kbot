import os
import shutil
import base64
import subprocess
from tempfile import mkdtemp
from io import BytesIO
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from loguru import logger


class OfficeToPDF:
    """Office 文档转 PDF 类"""
    
    def __init__(self):
        
        # 支持的文件扩展名
        self.supported_extensions = ['.ppt', '.pptx', '.doc', '.docx', '.odt', '.ods']
    
    async def convert_to_pdf(self, input_path: str, output_path: str | None = None) -> str:
        """转换文档为PDF，解决中文乱码问题"""
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

            # 添加额外的LibreOffice参数以确保中文处理
            cmd = [
                'libreoffice', 
                '--headless',
                '--language=zh',
                '--nolockcheck',
                '--nologo',
                '--norestore',
                '--convert-to', 'pdf:writer_pdf_Export:SelectPdfVersion=1:EmbedStandardFonts=true',
                input_path,
                '--outdir', str(output_file.parent)
            ]
            
            # 执行转换命令
            subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            # 重命名文件到用户指定的名称
            # 确保目标目录存在
            output_dir = os.path.dirname(str(output_file))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # 使用shutil.move替代rename，支持跨设备操作
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
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
            raise RuntimeError(f"文档转换失败: {error_msg}")

class FileToImage:
    """文件转图片类"""

    def __init__(self):
    
        # 支持的文件扩展名
        self.supported_extensions = ['.ppt', '.pptx', '.doc', '.docx', '.txt', '.pdf']

    async def convert_to_image(self, input_path: str) -> list[dict]:
        """
        Convert to images.
        生成一张图片       
        :param file_name: File name
        :param file_path: File path
        """
        file_path = Path(input_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
            
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        # 根据不同的文件类型进行处理
        file_extension = str(file_path).split('.')[-1].lower()

        if file_extension == 'pdf':
            # 使用 pdf2image 将 PDF 文件转换为Base64图片列表
            images = []
            pdf_images = convert_from_path(file_path)
            for page_num, image in enumerate(pdf_images, start=1):
                # Convert image to base64
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images.append({"page": page_num, "image": img_str})
            return images
        
        elif file_extension in ['ppt', 'pptx']:
            # 使用LibreOffice先将Word文档转换为PDF，再转换为Base64图片列表
            temp_dir = mkdtemp()
    
            try:
                # 第一步：将Word转为PDF
                pdf_path = os.path.join(temp_dir, "output.pdf")
                await OfficeToPDF().convert_to_pdf(input_path=str(file_path), output_path=pdf_path)
                
                # 第二步：将PDF转为图片
                return await self.convert_to_image(pdf_path)
                
            finally:
                # 清理临时文件
                shutil.rmtree(temp_dir)

        elif file_extension in ['doc', 'docx']:

            page_number = 2
            temp_dir = mkdtemp()
        
            try:
                cmd = [
                    "libreoffice",
                    "--headless",
                    "--convert-to", "png:writer_png_Export",
                    "--outdir", temp_dir,
                    file_path,
                    "--",
                    f"PageRange={page_number}-{page_number}"  # 只导出指定页
                ]
                
                # 添加优化参数
                cmd.extend([
                    "--nologo",
                    "--norestore",
                    "--nodefault",
                    "--nolockcheck",
                ])
                
                subprocess.run(cmd, check=True)
                
                # 查找生成的图片文件
                output_files = [
                    os.path.join(temp_dir, f) 
                    for f in os.listdir(temp_dir) 
                    if f.endswith('.png')
                ]
                
                if not output_files:
                    raise RuntimeError("未生成图片文件")

                images = []
                for image_file in output_files:
                    buffered = BytesIO()
                    Image.open(image_file).save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    images.append({"page": page_number, "image": img_str})
                return images
                
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"转换失败: {e.stderr.decode('utf-8', errors='ignore')}")
            finally:
                shutil.rmtree(temp_dir)

        elif file_extension == 'txt':
            # 分割文本文件并转换为图像
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 分割文本到合适的大小以便生成图像
            chunk_size = 500  # 每个文本块500字符
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            # 将每一个文本块渲染为图像
            images = []
            for page_num, chunk in enumerate(chunks, start=1):
                img = Image.new('RGB', (800, 600), color=(255, 255, 255))
                draw = ImageDraw.Draw(img)
                font = ImageFont.load_default()
                draw.text((10, 10), chunk, fill="black", font=font)
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images.append({"page": page_num, "image": img_str})
            return images
        else:
            raise ValueError("Unsupported file type for image conversion")
    
        