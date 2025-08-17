import os
import shutil
import subprocess
from pathlib import Path

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
                raise RuntimeError("转换失败: " + str(e))

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

    
        