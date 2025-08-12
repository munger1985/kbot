import sys
import os
import locale
import subprocess
from pathlib import Path


class OfficeToPDFConverter:
    """Office 文档转 PDF 类"""
    
    def __init__(self):
        # 设置系统区域设置和编码
        #self._set_system_locale()
        
        # 支持的文件扩展名
        self.supported_extensions = ['.ppt', '.pptx', '.doc', '.docx', '.odt', '.ods']
    
    # def _set_system_locale(self):
    #     """设置系统区域和编码以支持中文"""
    #     try:
    #         locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
    #     except locale.Error:
    #         try:
    #             locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    #         except locale.Error:
    #             locale.setlocale(locale.LC_ALL, '')
        
    #     # 设置Python默认编码
    #     if sys.version_info[0] < 3:
    #         reload(sys)
    #         sys.setdefaultencoding('utf-8')
        
    #     # 设置环境变量
    #     os.environ['LANG'] = 'zh_CN.UTF-8'
    #     os.environ['LC_ALL'] = 'zh_CN.UTF-8'
    
    def _encode_path(self, path: str | Path) -> str:
        """确保路径编码正确"""
        path = str(Path(path))
        return path
    
    async def convert_to_pdf(self, input_path: str | Path, output_path: str | Path | None = None) -> str:
        """转换文档为PDF，解决中文乱码问题"""
        input_path = Path(self._encode_path(input_path))
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
            
        if input_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"不支持的文件格式: {input_path.suffix}")
            
        # 处理输出路径
        if output_path is None:
            output_path = input_path.with_suffix('.pdf')
        else:
            output_path = Path(self._encode_path(output_path))
            
            # 如果输出路径是目录，则在该目录下创建同名PDF文件
            if output_path.is_dir():
                output_path = output_path / input_path.with_suffix('.pdf').name
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:

            # 临时输出到目标目录，使用输入文件名
            temp_output = output_path.parent / input_path.with_suffix('.pdf').name

            # 添加额外的LibreOffice参数以确保中文处理
            cmd = [
                'libreoffice', 
                '--headless',
                '--language=zh',
                '--nolockcheck',
                '--nologo',
                '--norestore',
                '--convert-to', 'pdf:writer_pdf_Export:SelectPdfVersion=1:EmbedStandardFonts=true',
                str(input_path),
                '--outdir', str(output_path.parent)
            ]
            
            # # 添加额外的环境变量
            # env = os.environ.copy()
            # env['LANGUAGE'] = 'zh_CN.UTF-8'
            # env['LC_PAPER'] = 'zh_CN.UTF-8'
            # env['LC_CTYPE'] = 'zh_CN.UTF-8'
            # env['LC_MESSAGES'] = 'zh_CN.UTF-8'
            # env['LC_COLLATE'] = 'zh_CN.UTF-8'
            # env['LC_MONETARY'] = 'zh_CN.UTF-8'
            # env['LC_NUMERIC'] = 'zh_CN.UTF-8'
            # env['LC_TIME'] = 'zh_CN.UTF-8'
            
            # 执行转换命令
            subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                #env=env
            )
            
            # 重命名文件到用户指定的名称
            if temp_output.exists() and temp_output != output_path:
                temp_output.rename(output_path)

            # 验证输出文件
            if not output_path.exists():
                # 尝试处理可能的文件名编码问题
                alt_path = Path(output_path.parent) / (input_path.stem + '.pdf')
                if alt_path.exists():
                    return str(alt_path)
                raise RuntimeError("转换成功但输出文件未创建")
                
            return str(output_path)
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
            raise RuntimeError(f"文档转换失败: {error_msg}")

