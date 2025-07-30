import os
import uuid
import json
import aiohttp
from loguru import logger

from .file_params import FileParams
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_biz_txt_embedding import KbotBizTxtEmbeddingRepository
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from dao.data_dict import FileStatus, ChunkType, SplitStrategy
from core.config import settings
from utils.chunk_text import chunk_text
from utils.call_models import call_embedding_model
from utils import check_text_file
import os
import uuid
import json
from pathlib import Path
import pdfplumber
from PIL import Image
import io
import pandas as pd
from typing import List, Dict, Any, Tuple
# 替换fitz为pdfminer.six
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTImage, LTFigure


class PDFPlumberParser:
    async def parse(self) -> Tuple[
        List[Dict], List[Dict], List[Dict] ]:
        pdf_path = self.file_params.file_path
        split_strategy = int(self. file_params.parser.get("split_strategy", 1))
        file_repo = KbotMdKbFilesRepository()

        if split_strategy == SplitStrategy.BY_PAGE.value:
            text_content, images_info, tables_info = self.extract_all_per_page()
            self.text_content = text_content
            self.images_info= images_info
            self.tables_info = tables_info
            ###### text
            embed_entities = []
            for each_content in text_content:
                text = each_content['text']
                page = each_content['page']
                if text != "":
                    embeddings_list =await call_embedding_model(self.file_params.txt_embed_model, [text])
                    embedding = embeddings_list[0]
                    embed_entity = KbotBizTxtEmbedding(
                        embed_id=str(uuid.uuid4()),
                        chunk_doc=text,
                        chunk_metadata=json.dumps({"chunk_type": ChunkType.TEXT,
                                                   "split_strategy": int(split_strategy),
                                                   "file_path": self.file_params.file_path,
                                                   "page": page}),
                        file_id=self.file_params.file_id,
                        embedding=embedding
                    )
                    embed_entities.append(embed_entity)

                embedding_repo = KbotBizTxtEmbeddingRepository()
                logger.debug(f"Attempting to save {len(embed_entities)} embeddings to database...")
                try:
                    result = await embedding_repo.create(kb_id=self.file_params.kb_id, embeddings=embed_entities)
                    if result:
                        logger.info(f"Successfully saved {len(embed_entities)} embeddings for {self.file_params.file_path}")
                        logger.debug(f"Database operation returned: {result}")
                    else:
                        msg = f"Failed to save embeddings for {self.file_params.file_path} (repository returned False)"
                        logger.error(msg)
                        await file_repo.update_file_status(self.file_params.file_id, FileStatus.PARSE_FAILED, msg)
                        return False
                except Exception as e:
                    msg = f"Exception while saving embeddings: {str(e)}"
                    logger.error(msg, exc_info=True)
                    await  file_repo.update_file_status(self.file_params.file_id, FileStatus.PARSE_FAILED, msg)
                    return False
            #### table
            embed_entities = []
            for each_content in tables_info:
                table_id = each_content['uuid']
                page = each_content['page']

                table_file_path = each_content['file_path']
                with open(table_file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    if text != "":
                        embeddings_list = await call_embedding_model(self.file_params.txt_embed_model, [text])
                        embedding = embeddings_list[0]
                        embed_entity = KbotBizTxtEmbedding(
                            embed_id=str(uuid.uuid4()),
                            chunk_doc=text,
                            chunk_metadata=json.dumps({"chunk_type": ChunkType.TEXT,
                                                       "split_strategy": int(split_strategy),
                                                       "file_path": self.file_params.file_path,
                                                       "page": page}),
                            file_id=self.file_params.file_id,
                            embedding=embedding
                        )
                        embed_entities.append(embed_entity)

                    embedding_repo = KbotBizTxtEmbeddingRepository()
                    logger.debug(f"Attempting to save {len(embed_entities)} embeddings to database...")
                    try:
                        result = await embedding_repo.create(kb_id=self.file_params.kb_id,
                                                             embeddings=embed_entities)
                        if result:
                            logger.info(
                                f"Successfully saved {len(embed_entities)} embeddings for {self.file_params.file_path}")
                            logger.debug(f"Database operation returned: {result}")
                        else:
                            msg = f"Failed to save embeddings for {self.file_params.file_path} (repository returned False)"
                            logger.error(msg)
                            await file_repo.update_file_status(self.file_params.file_id, FileStatus.PARSE_FAILED,
                                                               msg)
                            return False
                    except Exception as e:
                        msg = f"Exception while saving embeddings: {str(e)}"
                        logger.error(msg, exc_info=True)
                        await  file_repo.update_file_status(self.file_params.file_id, FileStatus.PARSE_FAILED, msg)
                        return False


            # # 获取按页分块的文本数组
            # text_chunks = PDFPlumberParser.get_text_chunks(text_content)

        # # 保存结果
        # parser.save_results()
        #
        # # 打印摘要
        # parser.logger.info_summary()

        return text_content, images_info, tables_info

    @staticmethod
    def get_text_chunks(text_content: List[Dict]) -> List[str]:
        """静态方法：从文本内容中提取按页分块的文本数组

        Args:
            text_content: 包含页码和文本的字典列表，每个字典包含'page'和'text'键

        Returns:
            按页分块的文本字符串列表
        """
        # 按页码排序
        sorted_content = sorted(text_content, key=lambda x: x['page'])

        # 提取文本
        text_chunks = [item['text'] for item in sorted_content]

        return text_chunks

    def __init__(self, file_params: FileParams):
        self.file_params = file_params
        self.pdf_path = file_params.file_path
        self.output_dir = Path(file_params.file_path).parent / "output"
        self.images_dir = self.output_dir / "images"
        self.tables_dir = self.output_dir / "tables"

        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)

        self.text_content = []
        self.images_info = []
        self.tables_info = []
        self.page_content = []  # 存储每页的完整内容（包含占位符）

    def extract_all_per_page(self):
        """提取PDF中的所有内容：文字、图片和表格"""
        logger.info(f"Parsing file: {self.pdf_path}")

        try:
            # 使用pdfplumber提取文字和表格
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"Parsing page {page_num} ")

                    # 提取文字和表格
                    page_text, page_tables = self._extract_text_and_tables_from_page(page, page_num)

                    # 提取图片（使用pdfminer.six替代fitz）
                    page_images = self._extract_images_from_page_pdfminer(page_num)

                    # 组合页面内容（文字 + 占位符）
                    combined_content = self._combine_page_content(page_text, page_images, page_tables, page_num)

                    self.page_content.append({
                        'page': page_num,
                        'content': combined_content
                    })

        except Exception as e:
            logger.info(f"解析PDF时出错: {e}")

        return self.text_content, self.images_info, self.tables_info

    def _extract_text_and_tables_from_page(self, page, page_num: int) -> Tuple[str, List[Dict]]:
        """从页面提取文字和表格"""
        page_text = ""
        page_tables = []

        try:
            # 提取表格
            tables = page.find_tables()

            for table_index, table in enumerate(tables):
                # 生成表格UUID
                table_uuid = str(uuid.uuid4())

                # 提取表格数据
                table_data = table.extract()

                if table_data:
                    # 保存为CSV
                    csv_filename = f"table_{table_uuid}.csv"
                    csv_path = self.tables_dir / csv_filename

                    # 转换为DataFrame并保存
                    df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data[0] else None)
                    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

                    # 记录表格信息
                    table_info = {
                        'uuid': table_uuid,
                        'filename': csv_filename,
                        'page': page_num,
                        'file_path': str(csv_path),
                        'rows': len(table_data),
                        'columns': len(table_data[0]) if table_data and table_data[0] else 0,
                        'bbox': table.bbox
                    }

                    self.tables_info.append(table_info)
                    page_tables.append(table_info)

                    logger.info(f"已保存表格: {csv_path} (第{page_num}页)")

            # 提取文字（排除表格区域）
            # 获取表格边界框
            table_bboxes = [table.bbox for table in tables]

            # 提取文字，排除表格区域
            if table_bboxes:
                # 使用crop方法排除表格区域
                filtered_page = page
                for bbox in table_bboxes:
                    # 这里简化处理，实际可以更精确地排除表格区域
                    pass
                page_text = filtered_page.extract_text() or ""
            else:
                page_text = page.extract_text() or ""

            # 记录文字内容
            if page_text.strip():
                self.text_content.append({
                    'page': page_num,
                    'text': page_text.strip()
                })

        except Exception as e:
            logger.info(f"提取第{page_num}页文字和表格时出错: {e}")
            page_text = page.extract_text() or ""

        return page_text, page_tables

    def _extract_images_from_page_pdfminer(self, page_num: int) -> List[Dict]:
        """使用pdfminer.six从页面提取图片（替代fitz）"""
        page_images = []

        try:
            with open(self.pdf_path, 'rb') as file:
                rsrcmgr = PDFResourceManager()
                device = PDFPageAggregator(rsrcmgr, laparams=LAParams())
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                pages = PDFPage.get_pages(file)

                # 跳到指定页面
                for current_page_num, page in enumerate(pages, 1):
                    if current_page_num == page_num:
                        interpreter.process_page(page)
                        layout = device.get_result()
                        page_images = self._extract_images_from_layout(layout, page_num)
                        break

        except Exception as e:
            logger.info(f"提取第{page_num}页图片时出错: {e}")

        return page_images

    def _extract_images_from_layout(self, layout, page_num: int) -> List[Dict]:
        """从layout中递归提取图片"""
        images = []

        for obj in layout:
            if isinstance(obj, LTImage):
                image_info = self._save_image_pdfminer(obj, page_num)
                if image_info:
                    images.append(image_info)
            elif isinstance(obj, LTFigure):
                # 递归处理图形对象
                sub_images = self._extract_images_from_layout(obj, page_num)
                images.extend(sub_images)

        return images

    def _save_image_pdfminer(self, lt_image, page_num: int) -> Dict:
        """保存pdfminer.six提取的图片"""
        try:
            # 去除bbox起始坐标x或y为0的图片，可能是背景图片
            bbox = getattr(lt_image, 'bbox', None)
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                if x0 == 0 or y0 == 0:
                    logger.info(f"忽略背景图片，bbox起始坐标为0 (第{page_num}页)")
                    return None

            result = self._image_data_from_stream(lt_image)
            if result is None:
                return None

            image_data, ext, pil_image = result

            # 获取图片尺寸
            if pil_image is not None:
                width, height = pil_image.size
            else:
                width = getattr(lt_image, 'width', 0)
                height = getattr(lt_image, 'height', 0)

            # 判断图片是否过小（小于200x200像素）
            if width < 200 or height < 200:
                logger.info(f"忽略小图片: {width}x{height}像素 (第{page_num}页)")
                return None

            image_uuid = str(uuid.uuid4())

            if pil_image is not None:
                image_path = self.images_dir / f"{image_uuid}.png"
                pil_image.save(image_path, format="PNG")
                ext = 'png'
            else:
                image_path = self.images_dir / f"{image_uuid}.{ext}"
                with open(image_path, 'wb') as f:
                    f.write(image_data)

            # 记录图片信息
            image_info = {
                'uuid': image_uuid,
                'filename': image_path.name,
                'page': page_num,
                'file_path': str(image_path),
                'width': width,
                'height': height,
                'bbox': bbox,
                'format': ext
            }

            self.images_info.append(image_info)
            logger.info(f"已保存图片: {image_path} (第{page_num}页)")
            return image_info

        except Exception as e:
            logger.info(f"保存第{page_num}页图片时出错: {e}")
            return None

    def _image_data_from_stream(self, lt_image):
        """从LTImage对象中提取图片数据"""
        if not hasattr(lt_image, 'stream') or lt_image.stream is None:
            return None

        stream = lt_image.stream
        filters = stream.get('Filter')

        if isinstance(filters, list):
            filters = filters[0]

        if filters:
            if filters.name == 'DCTDecode':
                ext = 'jpg'
                data = stream.get_rawdata()
                return data, ext, None
            elif filters.name == 'JPXDecode':
                ext = 'jp2'
                data = stream.get_rawdata()
                return data, ext, None
            elif filters.name == 'FlateDecode':
                # 需要用Pillow解码
                try:
                    width = int(stream.get('Width', 0))
                    height = int(stream.get('Height', 0))
                    color_space = stream.get('ColorSpace')
                    bpc = int(stream.get('BitsPerComponent', 8))
                    data = stream.get_data()

                    mode = None
                    if color_space:
                        if isinstance(color_space, list):
                            color_space = color_space[0]
                        if hasattr(color_space, 'name'):
                            if color_space.name == 'DeviceRGB':
                                mode = 'RGB'
                            elif color_space.name == 'DeviceGray':
                                mode = 'L'
                            elif color_space.name == 'DeviceCMYK':
                                mode = 'CMYK'

                    if mode is None:
                        mode = 'RGB'

                    pil_image = Image.frombytes(mode, (width, height), data)
                    return data, 'png', pil_image
                except Exception as e:
                    logger.info(f"FlateDecode图片解码失败: {e}")
                    return None
            else:
                ext = 'bin'
                data = stream.get_rawdata()
                return data, ext, None
        else:
            ext = 'bin'
            data = stream.get_rawdata()
            return data, ext, None

    def _combine_page_content(self, page_text: str, page_images: List[Dict], page_tables: List[Dict],
                              page_num: int) -> str:
        """组合页面内容，在适当位置插入占位符"""
        combined_content = f"\n{'=' * 20} 第 {page_num} 页 {'=' * 20}\n\n"

        # 添加图片占位符
        if page_images:
            combined_content += "\n=== 图片 ===\n"
            for img_info in page_images:
                combined_content += f"[image:{img_info['uuid']}]\n"

        # 添加表格占位符
        if page_tables:
            combined_content += "\n=== 表格 ===\n"
            for table_info in page_tables:
                combined_content += f"[table:{table_info['uuid']}]\n"

        # 添加文字内容
        if page_text.strip():
            combined_content += "\n=== 文字内容 ===\n"
            combined_content += page_text

        return combined_content

    def make_parsed_metadata(self):

            placeholders_mapping = {
                'images': [
                    {
                        'uuid': img['uuid'],
                        'placeholder': f"[image:{img['uuid']}]",
                        'filename': img['filename'],
                        'page': img['page'],
                        'file_path': img['file_path']
                    } for img in self.images_info
                ],
                'tables': [
                    {
                        'uuid': table['uuid'],
                        'placeholder': f"[table:{table['uuid']}]",
                        'filename': table['filename'],
                        'page': table['page'],
                        'file_path': table['file_path']
                    } for table in self.tables_info
                ]
            }
            json_string = json.dumps(placeholders_mapping, ensure_ascii=False, indent=2)
            return  json_string




    def save_results(self):
        """保存所有提取结果"""
        try:
            # 保存带占位符的完整文本
            full_text_with_placeholders = "\n".join([page['content'] for page in self.page_content])
            with open(self.output_dir / "extracted_text_with_placeholders.txt", 'w', encoding='utf-8') as f:
                f.write(full_text_with_placeholders)

            # 保存纯文本（不含占位符）
            pure_text = "\n".join([item['text'] for item in self.text_content])
            with open(self.output_dir / "extracted_text_only.txt", 'w', encoding='utf-8') as f:
                f.write(pure_text)

            # 保存图片信息
            with open(self.output_dir / "images_info.json", 'w', encoding='utf-8') as f:
                json.dump(self.images_info, f, ensure_ascii=False, indent=2)

            # 保存表格信息
            with open(self.output_dir / "tables_info.json", 'w', encoding='utf-8') as f:
                json.dump(self.tables_info, f, ensure_ascii=False, indent=2)

            # 保存占位符映射
            placeholders_mapping = {
                'images': [
                    {
                        'uuid': img['uuid'],
                        'placeholder': f"[image:{img['uuid']}]",
                        'filename': img['filename'],
                        'page': img['page'],
                        'file_path': img['file_path']
                    } for img in self.images_info
                ],
                'tables': [
                    {
                        'uuid': table['uuid'],
                        'placeholder': f"[table:{table['uuid']}]",
                        'filename': table['filename'],
                        'page': table['page'],
                        'file_path': table['file_path']
                    } for table in self.tables_info
                ]
            }

            with open(self.output_dir / "placeholders_mapping.json", 'w', encoding='utf-8') as f:
                json.dump(placeholders_mapping, f, ensure_ascii=False, indent=2)

            logger.info(f"\n所有结果已保存到: {self.output_dir}")

        except Exception as e:
            logger.info(f"保存结果时出错: {e}")

    def print_summary(self):
        logger.info("\n" + "=" * 50)
        logger.info("PDF解析完成!")
        logger.info("=" * 50)
        logger.info(f"提取的文字段落数: {len(self.text_content)}")
        logger.info(f"提取的图片数量: {len(self.images_info)}")
        logger.info(f"提取的表格数量: {len(self.tables_info)}")
        logger.info(f"输出目录: {self.output_dir}")

        pages_with_text = set(item['page'] for item in self.text_content)
        pages_with_images = set(item['page'] for item in self.images_info)
        pages_with_tables = set(item['page'] for item in self.tables_info)

        logger.info(f"包含文字的页数: {len(pages_with_text)}")
        logger.info(f"包含图片的页数: {len(pages_with_images)}")
        logger.info(f"包含表格的页数: {len(pages_with_tables)}")
        logger.info("=" * 50)


async def process_pdf(file_params: FileParams) -> bool:
    """
    处理文本文件，将其分割成指定大小的块，并调用嵌入微服务获取嵌入向量后写入数据库
    
    参数:
        file_params: 文件参数类
        
    返回:
        是否成功处理文件
    """
    # 检查文本嵌入模型是否指定
    if not check_text_file(file_params):
        return False
    # if text_length == 0:
    #     msg = f"Empty file: {file_params.file_path}"
    #     logger.info(msg)
    #     await file_repo.update_file_status(file_params.file_id, FileStatus.PARSED, msg)
    #     return True

    file_repo = KbotMdKbFilesRepository()
    parser = PDFPlumberParser(file_params)

    try:
        logger.debug(f"Processing pdf file: {file_params.file_path}")
        text_content, images_info, tables_info = await  parser.parse()
        parsed_metadata = parser.make_parsed_metadata()

        ok=file_repo.update_file_parsed_metadata(file_params.file_id, parsed_metadata)
        return ok
    except Exception as e:
        logger.error(f"Error processing pdf file: {file_params.file_path}, error: {e}")
        await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, str(e))
        return False
    #     split_strategy = int(file_params.paser.get("split_strategy", 1))
    #     chunk_size = int(file_params.paser.get("chunk_size", 500))
    #     overlap = int(file_params.paser.get("chunk_overlap", 50))
    #
    #     logger.debug(f"Chunk size: {chunk_size}, chunk overlap: {overlap}")
    #
    #     # 根据策略选择分割方式: 根据chunk size和overlap切片
    #     if split_strategy == SplitStrategy.SELF_SPLIT.value:
    #         # 文本分割逻辑
    #         if text_length <= chunk_size:
    #             logger.debug(f"Text length {text_length} <= chunk size {chunk_size}, no need to split.")
    #             chunks = [text]
    #         else:
    #             chunks = chunk_text(text, chunk_size, overlap)
    #     # 根据策略选择分割方式: 根据文档结构和段落切片
    #     elif split_strategy == SplitStrategy.BY_DOCSTRUCTURE.value:
    #         pass
    #     # 根据策略选择分割方式: 根据文档分页切片
    #     elif split_strategy == SplitStrategy.BY_PAGE.value:
    #         pass
    #     # 根据策略选择分割方式: 根据语义切片
    #     elif split_strategy == SplitStrategy.BY_SEMANTIC.value:
    #         pass
    #     else:
    #         msg = f"Invalid split strategy: {split_strategy}"
    #         logger.error(msg)
    #         await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
    #         return False
    #
    #     # 3.调用嵌入微服务获取嵌入向量
    #     logger.info(f"Calling embedding service for {file_params.file_path}...")
    #
    #     # 准备请求参数
    #     batch_size = settings["embed"]["batch_size"] or 0
    #     host = os.getenv("KBOT_EMBED_HOST", "localhost")
    #     port = os.getenv("KBOT_EMBED_PORT", "8001")
    #     embed_url = f"http://{host}:{port}/embed"
    #     logger.debug(f"Embedding URL: {embed_url}")
    #     headers = {"Content-Type": "application/json"}
    #     payload = {
    #         "model_id": int(file_params.txt_embed_model),
    #         "texts": chunks,
    #         "batch_size": int(batch_size)
    #     }
    #
    #     # 发送POST请求到嵌入微服务
    #     logger.info(f"Sending {len(chunks)} text chunks to embedding service...")
    #
    #     session = None
    #     try:
    #         session = aiohttp.ClientSession()
    #         response = await session.post(embed_url, headers=headers, json=payload)
    #
    #         logger.debug(f"Response status: {response.status}")
    #
    #         # 检查响应状态
    #         if response.status == 200:
    #             # 解析响应数据
    #             response_data = await response.json()
    #             embeddings = response_data["embeddings"]
    #             logger.info(f"Successfully obtained embeddings for {file_params.file_path}")
    #             embed_entities = []
    #
    #             for chunk, embedding in zip(chunks, embeddings):
    #                 # 保存嵌入向量到向量数据库
    #                 embed_entity = KbotBizTxtEmbedding(
    #                     embed_id=str(uuid.uuid4()),
    #                     chunk_doc=chunk,
    #                     chunk_metadata=json.dumps({"chunk_type": ChunkType.TEXT,
    #                                                 "split_strategy": int(split_strategy),
    #                                                 "chunk_size": int(chunk_size),
    #                                                 "chunk_overlap": int(overlap),
    #                                                 "file_path": file_params.file_path}),
    #                     file_id=file_params.file_id,
    #                     embedding=embedding
    #                 )
    #                 embed_entities.append(embed_entity)
    #
    #             embedding_repo = KbotBizTxtEmbeddingRepository()
    #             logger.debug(f"Attempting to save {len(embed_entities)} embeddings to database...")
    #             try:
    #                 result = await embedding_repo.create(kb_id=file_params.kb_id, embeddings=embed_entities)
    #                 if result:
    #                     logger.info(f"Successfully saved {len(embed_entities)} embeddings for {file_params.file_path}")
    #                     logger.debug(f"Database operation returned: {result}")
    #                 else:
    #                     msg = f"Failed to save embeddings for {file_params.file_path} (repository returned False)"
    #                     logger.error(msg)
    #                     await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
    #                     return False
    #             except Exception as e:
    #                 msg = f"Exception while saving embeddings: {str(e)}"
    #                 logger.error(msg, exc_info=True)
    #                 await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
    #                 return False
    #         else:
    #             response_text = await response.text()
    #             msg = f"Failed to get embeddings: HTTP {response.status}, {response_text}"
    #             logger.error(msg)
    #             await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
    #             return False
    #
    #     except Exception as e:
    #         msg = f"Error during embedding process: {str(e)}"
    #         logger.error(msg)
    #         await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
    #         return False
    #     finally:
    #         # 确保关闭会话
    #         if session is not None:
    #             await session.close()
    #     # 更新文件状态为已解析
    #     msg = f"File {file_params.file_path} processed: {len(chunks)} chunks created"
    #     logger.info(msg)
    #     await file_repo.update_file_status(file_params.file_id, FileStatus.PARSED, msg)
    #     return True
    #
    # except Exception as e:
    #     msg = f"Error in process_txt for {file_params.file_path}: {str(e)}"
    #     logger.error(msg)
    #     await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
    #     return False