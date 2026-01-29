import os
import mimetypes
import base64
# import mammoth
# from PyPDF2 import PdfReader, PdfWriter
# import docx
# import openpyxl
# from pptx import Presentation
from io import BytesIO
from typing import Any
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository



# 文件预览处理器
class FilePreview:

    async def get_preview(self, 
                    file_id: str, 
                    max_length: int = 10000, 
                    pages: int | list[int] | None = None, 
                    sheet_index: int = 0, 
                    preview_rows: int = 20,
                    slide: int | None = None
                    ) -> dict[str, Any]:
        """
        获取文件预览数据
        
        Args:
            file_id: 文件ID
            max_length: 最大文本/Word长度
            pages: 指定PDF页码（单页/多页/范围）
            sheet_index: 指定Excel工作表索引
            preview_rows: 指定Excel预览行数
            slide: 指定PPT幻灯片编号
            
        Returns:
            包含预览数据的字典
        """
        # 从数据库获取文件路径
        file_path = await KbotMdKbFilesRepository().get_path_by_id(file_id)

        if not file_path:
            return {"error": "文件不存在", "file_id": file_id}
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return {"error": "文件不存在", "file_path": file_path}
        
        # 获取文件MIME类型
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = 'application/octet-stream'
        
        # 根据文件类型调用不同的预览方法
        try:
            if mime_type == 'text/plain': 
                preview_data = await self._preview_text(file_path, max_length)
            elif mime_type in ['image/jpeg', 'image/png', 'image/gif']:
                preview_data = await self._preview_image(file_path)
            elif mime_type == 'application/pdf':
                preview_data = await self._preview_pdf(file_path, pages)
            elif mime_type in ['application/vnd.ms-excel', 
                              'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
                preview_data = await self._preview_excel(file_path, sheet_index, preview_rows)
            elif mime_type in ['application/vnd.ms-powerpoint',
                              'application/vnd.openxmlformats-officedocument.presentationml.presentation']:
                preview_data = await self._preview_ppt(file_path, slide)
            elif mime_type in ['application/msword', 
                              'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                preview_data = await self._preview_word(file_path, max_length)
                mime_type = 'text/html'  # Word预览为HTML格式
            elif mime_type == 'video/': 
                preview_data = await self._preview_video(file_path)
            elif mime_type == 'audio/': 
                preview_data = await self._preview_audio(file_path)
            else:
                pass
                
            preview_data.update({
                "file_id": file_id,
                "file_name": os.path.basename(file_path),
                "mime_type": mime_type,
                "file_size": os.path.getsize(file_path),
                "success": True
            })
            return preview_data
        except Exception as e:
            return {
                "error": f"处理文件时出错: {str(e)}",
                "file_id": file_id,
                "file_name": os.path.basename(file_path),
                "success": False
            }
    
    async def _preview_text(self, file_path: str, max_length: int) -> dict[str, Any]:
        """预览文本文件"""
        file_size = os.path.getsize(file_path)
        
        # 读取文件内容
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(max_length)
        except UnicodeDecodeError:
            # 如果UTF-8解码失败，尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read(max_length)
            except:
                content = "无法解码文本内容"
        
        truncated = file_size > max_length
        if truncated:
            content += "\n...(文件过大，只显示部分内容)"
        
        return {
            "preview_type": "text",
            "content": content
        }

    async def _preview_pdf(self, file_path: str, pages: int | list[int] | None = None) -> dict[str, Any]:
        """灵活版本 - 支持单页、页码列表或范围"""
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                total_pages = len(pdf_reader.pages)
                
                # 确定要提取的页码
                if pages is None:
                    # 默认提取所有页
                    target_pages = list(range(1, total_pages + 1))
                elif isinstance(pages, int):
                    # 单页
                    target_pages = [pages]
                else:
                    # 页码列表
                    target_pages = pages
                
                # 过滤有效页码
                valid_pages = [p for p in target_pages if 1 <= p <= total_pages]
                
                if not valid_pages:
                    return {
                        "preview_type": "pdf",
                        "error": f"没有有效的页码，总页数: {total_pages}"
                    }
                
                # 提取指定页
                writer = PdfWriter()
                for page_num in valid_pages:
                    writer.add_page(pdf_reader.pages[page_num - 1])
                
                # 生成PDF二进制数据
                buffer = BytesIO()
                writer.write(buffer)
                pdf_binary = buffer.getvalue()
                
                return {
                    "preview_type": "pdf",
                    "total_pages": total_pages,
                    "extracted_pages": valid_pages,
                    "page_count": len(valid_pages),
                    "content": base64.b64encode(pdf_binary).decode('utf-8'),
                    "message": f"成功提取 {len(valid_pages)} 页: {valid_pages}"
                }
                
        except Exception as e:
            return {
                "preview_type": "pdf",
                "error": f"处理PDF文件时出错: {str(e)}"
            }
    
    async def _preview_word(self, file_path: str, max_length: int = 10000) -> dict[str, Any]:
        """将Word转换为HTML保持原始结构进行预览"""
        try:
            # 使用mammoth库转换DOCX为HTML
            with open(file_path, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file)
                html_content = result.value
                
            # 限制HTML内容长度
            if len(html_content) > max_length:
                html_content = html_content[:max_length] + "...<p>[内容截断]</p>"
            
            return {
                "preview_type": "word",
                "content": html_content,
                "format": "html",
                "message": "Word文档HTML预览"
            }
            
        except Exception as e:
            return {
                "preview_type": "word",
                "error": f"处理Word文档时出错: {str(e)}"
            }
    
    async def _preview_excel(
        self, 
        file_path: str, 
        sheet_index: int = 0, 
        preview_rows: int = 10
    ) -> dict[str, Any]:
        """Excel预览 - 提取指定工作表的前几行数据"""
        workbook = None
        try:
            workbook = openpyxl.load_workbook(file_path, read_only=True)
            sheet_names = workbook.sheetnames
            total_sheets = len(sheet_names)
            
            # 验证工作表索引，如果超出范围，则设置为0或最后一个工作表
            if sheet_index < 0:
                sheet_index = 0
            if sheet_index >= total_sheets:
                sheet_index = total_sheets - 1
                
            
            # 提取指定工作表数据
            sheet_name = sheet_names[sheet_index]
            sheet = workbook[sheet_name]
            
            # 读取前preview_rows行数据
            data = []
            for i, row in enumerate(sheet.iter_rows(max_row=preview_rows, values_only=True)):
                # 转换单元格值为字符串，处理None值
                row_data = [str(cell) if cell is not None else "" for cell in row]
                data.append(row_data)
            
            workbook.close()
            
            return {
                "preview_type": "excel",
                "sheet_name": sheet_name,
                "sheet_index": sheet_index,
                "total_sheets": total_sheets,
                "preview_rows": len(data),
                "total_rows": sheet.max_row,
                "total_columns": sheet.max_column,
                "data": data,
                "message": f"工作表 '{sheet_name}' 的前{len(data)}行数据"
            }
            
        except Exception as e:
            if workbook:
                workbook.close()
            return {
                "preview_type": "excel",
                "error": f"处理Excel文档时出错: {str(e)}"
            }


    async def _preview_ppt(self, file_path: str, slide_number: int | None = None) -> dict[str, Any]:
        """预览PPT文档 - 简单返回文件信息和指定幻灯片的二进制内容"""
        try:
            # 读取PPT文件
            with open(file_path, 'rb') as f:
                ppt_content = f.read()
            
            # 获取总页数
            presentation = Presentation(file_path)
            total_slides = len(presentation.slides)
            content = base64.b64encode(ppt_content).decode('utf-8')
            
            result = {
                "preview_type": "ppt",
                "total_slides": total_slides,
                "content": content,
                "message": f"成功提取PPT文件内容"
            }
            
            # 如果指定了页码，添加到结果中
            if slide_number is not None:
                if 1 <= slide_number <= total_slides:
                    result["current_slide"] = slide_number
                    result["message"] = f"第{slide_number}张幻灯片预览"
                else:
                    result["error"] = f"幻灯片编号 {slide_number} 超出范围"
            
            return result
                
        except Exception as e:
            return {
                "preview_type": "ppt",
                "error": f"处理PPT文档时出错: {str(e)}"
            }
    
    async def _preview_image(self, file_path: str) -> dict[str, Any]:
        """预览图片文件"""
        # 将图片转换为base64编码
        with open(file_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        return {
            "preview_type": "image",
            "content": image_data
        }
    
    async def _preview_video(self, file_path: str) -> dict[str, Any]:
        """预览视频文件"""
        # 生成视频预览信息
        return {
            "preview_type": "video",
            "content": file_path,
            "message": "视频文件需要在HTML5 video标签中播放"
        }
    
    async def _preview_audio(self, file_path: str) -> dict[str, Any]:
        """预览音频文件"""
        # 生成音频预览信息
        return {
            "preview_type": "audio",
            "content": file_path,
            "message": "音频文件需要在HTML5 audio标签中播放"
        }
    