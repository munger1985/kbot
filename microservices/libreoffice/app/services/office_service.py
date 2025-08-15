import os
import uno
from com.sun.star.beans import PropertyValue
from com.sun.star.connection import NoConnectException
from app.config import Config
import time
from typing import Optional
from fastapi import HTTPException
import tempfile
from pdf2image import convert_from_path
from io import BytesIO
import base64

class OfficeService:
    def __init__(self):
        self.config = Config()
        self.context = None
        self.desktop = None
        self._connect()

    def _connect(self, retries=3, delay=5):
        local_context = uno.getComponentContext()
        resolver = local_context.ServiceManager.createInstanceWithContext(
            "com.sun.star.bridge.UnoUrlResolver", local_context)
        
        for attempt in range(retries):
            try:
                connection_str = f"socket,host={self.config.LIBREOFFICE_HOST},port={self.config.LIBREOFFICE_PORT};urp;StarOffice.ComponentContext"
                self.context = resolver.resolve(connection_str)
                smgr = self.context.ServiceManager
                self.desktop = smgr.createInstanceWithContext(
                    "com.sun.star.frame.Desktop", self.context)
                return
            except NoConnectException:
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
                raise HTTPException(status_code=500, detail="无法连接到LibreOffice服务")

    def _get_load_properties(self):
        return (
            PropertyValue("Hidden", 0, True, 0),
            PropertyValue("ReadOnly", 0, True, 0),
        )

    def convert_to_pdf(self, input_path: str, output_path: Optional[str] = None) -> str:
        input_url = uno.systemPathToFileUrl(os.path.abspath(input_path))
        
        if not output_path:
            output_path = os.path.join(self.config.TEMP_DIR, os.path.basename(input_path) + ".pdf")
        output_url = uno.systemPathToFileUrl(os.path.abspath(output_path))
        
        doc = self.desktop.loadComponentFromURL(
            input_url, "_blank", 0, self._get_load_properties())
        
        try:
            doc.storeToURL(output_url, (PropertyValue("FilterName", 0, "writer_pdf_Export", 0),))
            return output_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF转换失败: {str(e)}")
        finally:
            doc.close(True)

    def extract_page_as_image(self, input_path: str, page_number: int, dpi: int = None) -> dict:
        if not dpi:
            dpi = self.config.DEFAULT_DPI
        
        # 先转换为PDF
        pdf_path = self.convert_to_pdf(input_path)
        
        try:
            # 从PDF提取指定页
            images = convert_from_path(
                pdf_path,
                first_page=page_number,
                last_page=page_number,
                dpi=dpi,
                output_folder=self.config.TEMP_DIR
            )
            
            if not images:
                raise HTTPException(status_code=404, detail="未找到指定页面")
            
            # 转换为base64
            img_byte_arr = BytesIO()
            images[0].save(img_byte_arr, format='PNG')
            base64_str = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            return {
                "page": page_number,
                "image": base64_str,
                "format": "png",
                "dpi": dpi
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"页面提取失败: {str(e)}")
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)