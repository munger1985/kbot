import uuid
import json
import pdfplumber
import pandas as pd
import json

from pathlib import Path
from PIL import Image
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTImage, LTFigure
from loguru import logger
from .file_params import FileParams
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_biz_txt_embedding_repo import KbotBizTxtEmbeddingRepository
from dao.repositories.kbot_md_models_repo import KbotMdModelsRepository
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from core.dictionary import FileStatus, ChunkType, SplitStrategy
from utils.call_models import CallModel
from utils.common_methods import check_text_file
import traceback


class PDFPlumberParser:
    """PDF file parser class with optimized processing"""

    def __init__(self, file_params: FileParams, remove_header_footer: bool = True):
        self.file_params = file_params
        self.pdf_path = Path(file_params.file_path)
        self.output_dir = self.pdf_path.parent / "output"/file_params.file_id
        self.images_dir = self.output_dir / "images"
        self.tables_dir = self.output_dir / "tables"

        # Create output directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        self.text_contents: list[dict] = []
        self.text_chunks: list[dict] = []
        self.images_info: list[dict] = []
        self.tables_info: list[dict] = []
        self.page_content: list[dict] = []  # Stores complete page content with placeholders
        self.remove_header_footer = remove_header_footer

        self.chunk_size = 0
        self.chunk_overlap = 0

    async def parse(self) -> bool:
        """Main parsing method with optimized flow"""
        split_strategy = int(self.file_params.parser.get("split_strategy", SplitStrategy.FIXED_SIZE.value))
        file_repo = KbotMdKbFilesRepository()

        if split_strategy == SplitStrategy.PAGE.value:
            try:
                # Extract all content by page
                _, self.images_info, _  = self.extract_all_per_page()
                if not await self._process_embeddings_per_page():
                    return False

                # Save parsed metadata
                parsed_metadata = self.make_parsed_metadata()
                await file_repo.update_file_parsed_metadata(self.file_params.file_id, parsed_metadata)

                self.print_summary()
                return True

            except Exception as e:
                logger.error(f"Error processing PDF file: {str(e)}")
                logger.exception('asdasd',e)
                await file_repo.update_file_status(
                    self.file_params.file_id,
                    FileStatus.PARSE_FAILED,
                    str(e)
                )
                return False
        elif split_strategy == SplitStrategy.FIXED_SIZE.value:
            try:
                self.chunk_size = int(self.file_params.parser.get("chunk_size", 500))
                self.chunk_overlap = int(self.file_params.parser.get("chunk_overlap", 50))

                _, self.images_info, _= self.extract_all_by_fixed_size()
                # if not await self._process_images_embeddings(images_info):
                #     return False

                # Process text and table embeddings
                if not await self._process_embeddings_by_fixed_size():
                    return False

                # Save parsed metadata
                parsed_metadata = self.make_parsed_metadata()
                await file_repo.update_file_parsed_metadata(self.file_params.file_id, parsed_metadata)

                self.print_summary()
                return True

            except Exception as e:
                logger.exception("Error processing PDF file:  {}", e)
                tb = traceback.TracebackException.from_exception(e)
                errMsg= ''.join(tb.format())
                await file_repo.update_file_status(
                    self.file_params.file_id,
                    FileStatus.PARSE_FAILED,
                    errMsg
                )
                return False

        else:
            logger.warning(f"Unrecognized split strategy: {split_strategy}")
            return False
    async def _process_images_embeddings(self) -> list:
        if self.file_params.img2txt == 1:
        # if self.file_params.parser.get("extract_images", False):
            vlm_prompt_unique_name = "SYSTEM/image2text"
            vlm_model_unique_name = await KbotMdModelsRepository().get_unique_name_by_id(
            self.file_params.img2txt_model)  # type: ignore
            chunks = []
            chunk_metas = []
            for eachImage in self.images_info:
                description_file = Path(eachImage['file_path'] + ".description")
                if not description_file.exists():


                    image_description = await CallModel().call_vlm_model_for_parsing_picture(vlm_model_unique_name, # type: ignore
                                                                           eachImage['file_path'], vlm_prompt_unique_name) 
                    if image_description:
                        description_file.write_text(
                            image_description,
                            encoding='utf-8'
                        )
                        chunk_metas.append({
                            "chunk_type": ChunkType.IMAGE,
                            "page_num": eachImage['page_num'],
                            "image_id": eachImage['uuid'],
                        })
                        chunks.append(image_description)
            text_embedding_model = await KbotMdModelsRepository().get_unique_name_by_id(
                self.file_params.txt_embed_model # type: ignore
            )
            if not text_embedding_model:
                msg = f"text_embedding_model not found for id: {self.file_params.txt_embed_model}"
                logger.error(msg)
                await self._update_file_status(FileStatus.PARSE_FAILED, msg)
                return []
            embeddings_list= []
            if chunks:
                embeddings_list = await CallModel().call_embedding_model(text_embedding_model, chunks)
            if embeddings_list and len(embeddings_list) != len(chunks):
                msg = f"text_embedding_model  {text_embedding_model} returned invalid results (expected {len(chunks)}, got {len(embeddings_list) if embeddings_list else 0})"
                logger.error(msg)
                logger.error("failed file: {}",self.file_params.file_path)
                await self._update_file_status(FileStatus.PARSE_FAILED, msg)
                return []

                # Create embedding entities
            embed_entities = []
            for idx, (chunk, meta) in enumerate(zip(chunks, chunk_metas)):
                embed_entity = KbotBizTxtEmbedding(
                    kb_id=self.file_params.kb_id,
                    embed_id=meta['image_id'],
                    chunk_doc=chunk,
                    chunk_metadata=meta,
                    file_id=self.file_params.file_id,
                    embedding=embeddings_list[idx].embedding, # type: ignore
                    security_level=self.file_params.security_level
                )
                embed_entities.append(embed_entity)

            # Save all embeddings in one batch
            return embed_entities
        else:
            return []
        

    async def _process_embeddings_per_page(self) -> bool:
        """Process all content embeddings in a unified way"""
        model_unique_name = await KbotMdModelsRepository().get_unique_name_by_id(
            self.file_params.txt_embed_model  # type: ignore
        )
        if not model_unique_name:
            msg = f"Embedding model not found for id: {self.file_params.txt_embed_model}"
            logger.error(msg)
            await self._update_file_status(FileStatus.PARSE_FAILED, msg)
            return False

        # Prepare all content chunks for embedding
        chunks = []
        chunk_metas = []

        # Add text content
        for text_item in self.text_contents:
            if not text_item['text'].strip():
                continue

            chunks.append(text_item['text'])
            chunk_metas.append({
                "chunk_type": ChunkType.TEXT,
                "page_num": text_item['page_num']
            })

        # Add table content
        for table in self.tables_info:
            if not self.is_table_valid(table['file_path']):
                continue

            with open(table['file_path'], 'r', encoding='utf-8') as f:
                table_text = f.read()
                if table_text.strip():
                    chunks.append(table_text)
                    chunk_metas.append({
                        "chunk_type": ChunkType.TABLE,
                        "page_num": table['page_num']
                    })

        if not chunks:
            logger.warning("No valid content chunks found for embedding")
            return True  # Consider empty content as success

        # Get all embeddings in one call
        embeddings_list = await CallModel().call_embedding_model(model_unique_name, chunks)
        if not embeddings_list or len(embeddings_list) != len(chunks):
            msg = f"Embedding model {model_unique_name} returned invalid results (expected {len(chunks)}, got {len(embeddings_list) if embeddings_list else 0})"
            logger.error(msg)
            await self._update_file_status(FileStatus.PARSE_FAILED, msg)
            return False

        # Create embedding entities
        embed_entities = []
        for idx, (chunk, meta) in enumerate(zip(chunks, chunk_metas)):
            embed_entity = KbotBizTxtEmbedding(
                kb_id=self.file_params.kb_id,
                embed_id=str(uuid.uuid4()),
                chunk_doc=chunk,
                chunk_metadata=meta,
                file_id=self.file_params.file_id,
                embedding=embeddings_list[idx].embedding,
                security_level=self.file_params.security_level
            )
            embed_entities.append(embed_entity)

        image_embed_entities= await self._process_images_embeddings()
        embed_entities.extend(image_embed_entities)

        # Save all embeddings in one batch
        return await self._save_embeddings(embed_entities)


    async def _process_embeddings_by_fixed_size(self) -> bool:
        """Process text and table embeddings for by fixed size"""
        model_unique_name = await KbotMdModelsRepository().get_unique_name_by_id(
            self.file_params.txt_embed_model  # type: ignore
        )
        if not model_unique_name:
            msg = f"Embedding model not found for id: {self.file_params.txt_embed_model}"
            logger.error(msg)
            await self._update_file_status(FileStatus.PARSE_FAILED, msg)
            return False

        # Prepare all content chunks for embedding
        chunks = []
        chunk_metas = []

        # Add text content
        for text_item in self.text_chunks:
            if not text_item['text'].strip():
                continue

            chunks.append(text_item['text'])
            chunk_metas.append({
                "chunk_type": ChunkType.TEXT,
                "page_num": text_item['page_num']
            })

        # Add table content
        for table in self.tables_info:
            if not self.is_table_valid(table['file_path']):
                continue

            with open(table['file_path'], 'r', encoding='utf-8') as f:
                table_text = f.read()
                if table_text.strip():
                    chunks.append(table_text)
                    chunk_metas.append({
                        "chunk_type": ChunkType.TABLE,
                        "page_num": table['page_num']
                    })

        if not chunks:
            logger.warning("No valid content chunks found for embedding")
            return True  # Consider empty content as success

        # Get all embeddings in one call
        embeddings_list = await CallModel().call_embedding_model(model_unique_name, chunks)
        if not embeddings_list or len(embeddings_list) != len(chunks):
            msg = f"Embedding model {model_unique_name} returned invalid results (expected {len(chunks)}, got {len(embeddings_list) if embeddings_list else 0})"
            logger.error(msg)
            await self._update_file_status(FileStatus.PARSE_FAILED, msg)
            return False

        # Create embedding entities
        embed_entities = []
        for idx, (chunk, meta) in enumerate(zip(chunks, chunk_metas)):
            embed_entity = KbotBizTxtEmbedding(
                kb_id=self.file_params.kb_id,
                embed_id=str(uuid.uuid4()),
                chunk_doc=chunk,
                chunk_metadata=meta,
                file_id=self.file_params.file_id,
                embedding=embeddings_list[idx].embedding,
                security_level=self.file_params.security_level
            )
            embed_entities.append(embed_entity)
        image_embed_entities= await self._process_images_embeddings()
        embed_entities.extend(image_embed_entities)

        # Save all embeddings in one batch
        return await self._save_embeddings(embed_entities)


    async def _save_embeddings(self, embeddings: list[KbotBizTxtEmbedding]) -> bool:
        """Save embeddings to database with error handling"""
        if not embeddings:
            return False

        try:
            repo = KbotBizTxtEmbeddingRepository(kb_id=self.file_params.kb_id)
            await repo.initialize()
            result = await repo.create(kb_id=self.file_params.kb_id, embeddings=embeddings)
            if not result:
                msg = "Failed to save embeddings (repository returned False)"
                logger.error(msg)
                await self._update_file_status(FileStatus.PARSE_FAILED, msg)
                return False

            logger.info(f"Successfully saved {len(embeddings)} embeddings")
            return True

        except Exception as e:
            msg = f"Exception while saving embeddings: {str(e)}"
            logger.error(msg)
            await self._update_file_status(FileStatus.PARSE_FAILED, msg)
            return False

    async def _update_file_status(self, status: FileStatus, message: str) -> None:
        """Helper method to update file status"""
        await KbotMdKbFilesRepository().update_file_status(
            self.file_params.file_id,
            status,
            message
        )

    def remove_duplicate_prefix_suffix(self, text_content):
        """
        移除 text_content 中除第一个元素外所有元素的重复前缀和后缀
        保留第一个元素中的完整内容，其他元素中删除相同的前缀和后缀
        """
        if not text_content or len(text_content) < 2:
            return text_content

        # 获取第一个元素的完整内容作为参考
        first_content = text_content[0]['text']

        # 找出所有元素中共同的前缀和后缀
        common_prefix = ""
        common_suffix = ""

        # 找出共同前缀
        if len(text_content) > 1:
            # 从第二个元素开始比较
            for item in text_content[1:]:
                content = item['text']
                if not content:
                    continue

                # 找出当前元素与第一个元素的共同前缀
                current_prefix = ""
                min_len = min(len(first_content), len(content))
                for i in range(min_len):
                    if first_content[i] == content[i]:
                        current_prefix += first_content[i]
                    else:
                        break

                # 更新共同前缀（取最短的）
                if not common_prefix:
                    common_prefix = current_prefix
                else:
                    common_prefix = common_prefix[:len(current_prefix)]
                    if len(common_prefix) > len(current_prefix):
                        common_prefix = current_prefix[:len(current_prefix)]

        # 找出共同后缀
        if len(text_content) > 1:
            for item in text_content[1:]:
                content = item['text']
                if not content:
                    continue

                # 找出当前元素与第一个元素的共同后缀
                current_suffix = ""
                min_len = min(len(first_content), len(content))
                for i in range(1, min_len + 1):
                    if first_content[-i] == content[-i]:
                        current_suffix = first_content[-i] + current_suffix
                    else:
                        break

                # 更新共同后缀（取最短的）
                if not common_suffix:
                    common_suffix = current_suffix
                else:
                    common_suffix = current_suffix[:len(current_suffix)]
                    if len(common_suffix) > len(current_suffix):
                        common_suffix = current_suffix[:len(current_suffix)]

        print(f"检测到的共同前缀: '{common_prefix}'")
        print(f"检测到的共同后缀: '{common_suffix}'")

        # 生成新的列表
        new_text_content = []

        for i, item in enumerate(text_content):
            if i == 0:
                # 第一个元素保持不变
                new_text_content.append(item)
            else:
                # 其他元素中移除共同的前缀和后缀
                content = item['text']
                cleaned_content = content

                # 移除前缀
                if common_prefix and content.startswith(common_prefix):
                    cleaned_content = content[len(common_prefix):]

                # 移除后缀
                if common_suffix and cleaned_content.endswith(common_suffix):
                    cleaned_content = cleaned_content[:-len(common_suffix)]

                # 创建新的元素
                new_item = item.copy()
                new_item['text'] = cleaned_content.strip()
                new_text_content.append(new_item)

        print(f"原始元素数: {len(text_content)}")
        print(f"处理后元素数: {len(new_text_content)}")

        return new_text_content
    def extract_all_per_page(self) -> tuple[list[dict], list[dict], list[dict]]:
        """Extract all content from PDF by page"""
        logger.info(f"Parsing file: {self.pdf_path}")

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"Processing page {page_num}")

                    # Extract text and tables
                    page_text, page_tables = self._extract_text_and_tables(page, page_num)

                    # Extract images
                    page_images = self._extract_images_from_page(page_num)

                    # Combine content
                    combined = self._combine_page_content(page_text, page_images, page_tables, page_num)
                    self.page_content.append({'page_num': page_num, 'content': combined})

        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise

        return self.text_contents, self.images_info, self.tables_info

    def extract_all_by_fixed_size(self) -> tuple[list[dict], list[dict], list[dict]]:
        """Extract all content from PDF by page"""
        logger.info(f"Parsing file: {self.pdf_path}")

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"Processing page {page_num}")

                    # Extract text and tables
                    page_text, page_tables = self._extract_text_and_tables(page, page_num)

                    # Extract images
                    page_images = self._extract_images_from_page(page_num)

                    # Combine content
                    # combined = self._combine_page_content(page_text, page_images, page_tables, page_num)
                    # self.page_content.append({'page_num': page_num, 'content': page_text})

        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise
        for page in self.text_contents:
            page_num = page['page_num']
            content = page['text']
            for chunk_start in range(0, len(content), self.chunk_size - self.chunk_overlap):
                chunk_text = content[chunk_start:chunk_start + self.chunk_size]
                self.text_chunks.append({'page_num': page_num, 'text': chunk_text})
        return self.text_chunks, self.images_info, self.tables_info

    def _filter_header_footer(self, page, page_text: str) -> str:
        """过滤页眉页脚内容"""
        if not self.remove_header_footer or not page_text.strip():
            return page_text

        try:
            # 获取页面尺寸
            page_height = page.height
            page_width = page.width
            text_objects = page.objects.get("char", [])
            header_lines = [obj for obj in text_objects
                            if obj["y0"] > page.height * 0.9]
            header_height = max(header_lines, key=lambda x: x["top"])["top"]
            footer_lines = [obj for obj in text_objects
                            if obj["y0"] < page.height * 0.1]
            footer_height = max(footer_lines, key=lambda x: x["top"])["top"]

            # 使用动态检测的高度
            # header_height = getattr(self, '_detected_header_height', self.header_height)
            # footer_height = getattr(self, '_detected_footer_height', self.footer_height)

            # 提取主体内容区域（排除页眉页脚）
            main_content_bbox = (0, header_height + 10, page_width, footer_height - 10)

            # 确保边界框有效
            if main_content_bbox[1] >= main_content_bbox[3]:
                return page_text  # 如果页眉页脚太大，返回原始文本

            # 裁剪页面到主体内容区域
            main_content_page = page.crop(main_content_bbox)
            filtered_text = main_content_page.extract_text() or ""

            return filtered_text

        except Exception as e:
            print(f"过滤页眉页脚时出错: {e}，返回原始文本")
            return page_text
    def _extract_text_and_tables(self, page, page_num: int) -> tuple[str, list[dict]]:
        """Extract text and tables from a page"""
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
                    # 获取表头
                    headers = table_data[0] if table_data[0] else []

                    # 如果没有表头，生成默认表头
                    if not headers or not any(str(header).strip() if header is not None else '' for header in headers):
                        headers = [f"Column_{i + 1}" for i in range(len(table_data[0]) if table_data[0] else 0)]

                    # 将表格数据转换为JSON格式
                    json_data = []
                    for row_data in table_data[1:]:  # 跳过表头行
                        if row_data:  # 确保行数据不为空
                            row_dict = {}
                            for i, value in enumerate(row_data):
                                if i < len(headers):
                                    # 使用表头作为key，如果表头为空则使用默认列名
                                    header_str = str(headers[i]) if headers[i] is not None else ""
                                    key = header_str.strip() if header_str.strip() else f"Column_{i + 1}"
                                    row_dict[key] = value if value is not None else ""
                            if row_dict:  # 只添加非空行
                                json_data.append(row_dict)

                    # 保存为JSON文件
                    json_filename = f"table_{table_uuid}.json"
                    json_path = self.tables_dir / json_filename

                    # 保存JSON数据
                    import json
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=2)

                    # 记录表格信息
                    table_info = {
                        'uuid': table_uuid,
                        'filename': json_filename,
                        'page_num': page_num,
                        'file_path': str(json_path.absolute()),  # 使用绝对路径
                        'rows': len(json_data),
                        'columns': len(headers),
                        'bbox': table.bbox,
                        'headers': headers,  # 添加表头信息
                        'data': json_data  # 添加JSON数据
                    }

                    self.tables_info.append(table_info)
                    page_tables.append(table_info)

                    print(f"已保存表格: {json_path} (第{page_num}页)")

            # 提取文字（排除表格区域）
            # 获取表格边界框
            table_bboxes = [table.bbox for table in tables]

            # 提取文字，排除表格区域
            if table_bboxes:
                try:
                    # 创建排除表格区域的文本提取

                    # 合并所有表格区域为一个排除区域列表
                    exclude_areas = []
                    for bbox in table_bboxes:
                        x0, y0, x1, y1 = bbox
                        exclude_areas.append((x0, y0, x1, y1))

                    # 获取页面所有文本字符及其位置
                    words = page.extract_words()

                    # 过滤掉在表格区域内的文本
                    filtered_words = []
                    for word in words:
                        word_x0, word_y0, word_x1, word_y1 = word['x0'], word['top'], word['x1'], word['bottom']

                        # 检查单词是否在任意表格区域内
                        in_table = False
                        for table_x0, table_y0, table_x1, table_y1 in exclude_areas:
                            # 如果单词与表格区域重叠，则跳过
                            if (word_x0 < table_x1 and word_x1 > table_x0 and
                                    word_y0 < table_y1 and word_y1 > table_y0):
                                in_table = True
                                break

                        if not in_table:
                            filtered_words.append(word)

                    # 按行和位置重新组合文本
                    if filtered_words:
                        # 按y坐标分组（同一行）
                        lines = {}
                        for word in filtered_words:
                            y_key = int(word['top'] / 10)  # 近似分组
                            if y_key not in lines:
                                lines[y_key] = []
                            lines[y_key].append(word)

                        # 按行排序并组合文本
                        sorted_lines = []
                        for y_key in sorted(lines.keys()):
                            line_words = sorted(lines[y_key], key=lambda w: w['x0'])
                            line_text = ' '.join([w['text'] for w in line_words])
                            sorted_lines.append(line_text)

                        page_text = '\n'.join(sorted_lines)
                    else:
                        page_text = ""

                except Exception as e:
                    print(f"排除表格区域时出错: {e}")
                    page_text = page.extract_text() or ""
            else:
                page_text = page.extract_text() or ""

            # 应用页眉页脚过滤
            page_text = self._filter_header_footer(page, page_text)

            # 记录文字内容
            if page_text.strip():
                self.text_contents.append({
                    'page_num': page_num,
                    'text': page_text.strip()
                })

        except Exception as e:
            traceback.print_exc()
            print(f"提取第{page_num}页文字和表格时出错: {e}")
            page_text = page.extract_text() or ""
            # 应用页眉页脚过滤
            page_text = self._filter_header_footer(page, page_text)
        self.text_contents = self.remove_duplicate_prefix_suffix(self.text_contents)

        return page_text, page_tables

    def _extract_images_from_page(self, page_num: int) -> list[dict]:
        """Extract images from a page using pdfminer"""
        page_images = []

        try:
            with open(self.pdf_path, 'rb') as file:
                rsrcmgr = PDFResourceManager()
                device = PDFPageAggregator(rsrcmgr, laparams=LAParams())
                interpreter = PDFPageInterpreter(rsrcmgr, device)

                for current_page_num, page in enumerate(PDFPage.get_pages(file), 1):
                    if current_page_num == page_num:
                        interpreter.process_page(page)
                        layout = device.get_result()
                        page_images = self._process_layout_images(layout, page_num)
                        break

        except Exception as e:
            logger.error(f"Error extracting images from page {page_num}: {e}")

        return page_images

    def _process_layout_images(self, layout, page_num: int) -> list[dict]:
        """Process layout to extract images"""
        images = []

        for obj in layout:
            if isinstance(obj, LTImage):
                image_info = self._save_image(obj, page_num)
                if image_info:
                    images.append(image_info)
            elif isinstance(obj, LTFigure):
                images.extend(self._process_layout_images(obj, page_num))

        return images

    def _save_image(self, lt_image, page_num: int) -> dict | None:
        """Save an extracted image"""
        try:
            # Skip background images (starting at 0,0)
            if hasattr(lt_image, 'bbox'):
                x0, y0, _, _ = lt_image.bbox
                if x0 == 0 or y0 == 0:
                    logger.debug(f"Skipping background image at page {page_num}")
                    return None

            # Get image data
            result = self._get_image_data(lt_image)
            if not result:
                return None

            image_data, ext, pil_image = result
            image_uuid = str(uuid.uuid4())

            # Save image
            if pil_image:
                image_path = self.images_dir / f"{image_uuid}.png"
                pil_image.save(image_path, format="PNG")
                width, height = pil_image.size
            else:
                image_path = self.images_dir / f"{image_uuid}.{ext}"
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                width = getattr(lt_image, 'width', 0)
                height = getattr(lt_image, 'height', 0)

            # Skip small images
            if width < 200 or height < 200:
                logger.debug(f"Skipping small image ({width}x{height}) at page {page_num}")
                return None

            # Create image info
            image_info = {
                'uuid': image_uuid,
                'filename': image_path.name,
                'page_num': page_num,
                'file_path': str(image_path.absolute()),
                'width': width,
                'height': height,
                'bbox': getattr(lt_image, 'bbox', None),
                'format': ext
            }

            self.images_info.append(image_info)
            logger.debug(f"Saved image: {image_path.absolute()} (page {page_num})")
            return image_info

        except Exception as e:
            logger.error(f"Error saving image from page {page_num}: {e}")
            return None

    def _get_image_data(self, lt_image) -> tuple[bytes, str, Image.Image | None] | None:
        """Extract image data from LTImage object"""
        if not hasattr(lt_image, 'stream') or not lt_image.stream:
            return None

        stream = lt_image.stream
        filters = stream.get('Filter', [])
        if isinstance(filters, list):
            filters = filters[0] if filters else None

        if not filters:
            return stream.get_rawdata(), 'bin', None

        filter_name = getattr(filters, 'name', '')
        if filter_name == 'DCTDecode':
            return stream.get_rawdata(), 'jpg', None
        elif filter_name == 'JPXDecode':
            return stream.get_rawdata(), 'jp2', None
        elif filter_name == 'FlateDecode':
            try:
                width = int(stream.get('Width', 0))
                height = int(stream.get('Height', 0))
                color_space = stream.get('ColorSpace')
                data = stream.get_data()

                mode = 'RGB'  # default
                if color_space:
                    if isinstance(color_space, list):
                        color_space = color_space[0]
                    if hasattr(color_space, 'name'):
                        if color_space.name == 'DeviceGray':
                            mode = 'L'
                        elif color_space.name == 'DeviceCMYK':
                            mode = 'CMYK'

                pil_image = Image.frombytes(mode, (width, height), data)
                return data, 'png', pil_image
            except Exception as e:
                logger.error(f"Error decoding FlateDecode image: {e}")
                return None
        else:
            return stream.get_rawdata(), 'bin', None

    def _combine_page_content(self, text: str, images: list[dict], tables: list[dict], page_num: int) -> str:
        """Combine page content with placeholders
        especially for split by page strategy
        """
        content = [f"\n{'=' * 20} Page {page_num} {'=' * 20}\n"]

        if images:
            content.append("\n=== Images ===\n")
            content.extend(f"[image:{img['uuid']}]\n" for img in images)

        if tables:
            content.append("\n=== Tables ===\n")
            content.extend(f"[table:{table['uuid']}]\n" for table in tables)

        if text.strip():
            content.append("\n=== Text ===\n")
            content.append(text)

        return ''.join(content)

    def make_parsed_metadata(self) -> str:
        """Generate metadata JSON with placeholders"""
        valid_tables = [t for t in self.tables_info if self.is_table_valid(t['file_path'])]

        metadata = {
            'images': [
                {
                    'uuid': img['uuid'],
                    'placeholder': f"[image:{img['uuid']}]",
                    'filename': img['filename'],
                    'page_num': img['page_num'],
                    'file_path': img['file_path']
                } for img in self.images_info
            ],
            'tables': [
                {
                    'uuid': table['uuid'],
                    'placeholder': f"[table:{table['uuid']}]",
                    'filename': table['filename'],
                    'page_num': table['page_num'],
                    'file_path': table['file_path']
                } for table in valid_tables
            ]
        }

        return json.dumps(metadata, ensure_ascii=False, indent=2)

    # @staticmethod
    def is_table_valid(self, json_path: str) -> bool:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 检查是否为列表且不为空
                if not isinstance(data, list) or len(data) == 0:
                    return False

                # 检查是否至少包含一个非空行
                for row in data:
                    if isinstance(row, dict) and any(
                            value.strip() if isinstance(value, str) else value for value in row.values()):
                        return True

                return False

        except Exception:
            return False
    # def _is_table_valid(self, csv_path: str) -> bool:
    #     """Check if a table CSV file contains valid content"""
    #     try:
    #         with open(csv_path, 'r', encoding='utf-8') as f:
    #             content = f.read()
    #             return any(c.isalnum() or '\u4e00' <= c <= '\u9fff' for c in content)
    #     except Exception:
    #         return False

    # def save_results(self):
    #     """Save all extracted results to files"""
    #     try:
    #         # Save text with placeholders
    #         # full_text = "\n".join(page['content'] for page in self.page_content)
    #         # (self.output_dir / "extracted_text_with_placeholders.txt").write_text(full_text, encoding='utf-8')
    #
    #         # Save pure text
    #         # pure_text = "\n".join(item['text'] for item in self.text_content)
    #         # (self.output_dir / "extracted_text_only.txt").write_text(pure_text, encoding='utf-8')
    #
    #         # Save metadata files
    #         # (self.output_dir / "images_info.json").write_text(
    #         #     json.dumps(self.images_info, ensure_ascii=False, indent=2),
    #         #     encoding='utf-8'
    #         # )
    #
    #         # (self.output_dir / "tables_info.json").write_text(
    #         #     json.dumps(self.tables_info, ensure_ascii=False, indent=2),
    #         #     encoding='utf-8'
    #         # )
    #
    #         # (self.output_dir / "placeholders_mapping.json").write_text(
    #         #     self.make_parsed_metadata(),
    #         #     encoding='utf-8'
    #         # )
    #
    #         logger.info(f"All results saved to: {self.output_dir}")
    #
    #     except Exception as e:
    #         logger.error(f"Error saving results: {e}")

    def print_summary(self):
        """Print parsing summary"""
        logger.info("\n" + "=" * 50)
        logger.info("PDF Parsing Complete!")
        logger.info("=" * 50)
        logger.info(f"Text paragraphs extracted: {len(self.text_contents)}")
        logger.info(f"Images extracted: {len(self.images_info)}")
        logger.info(f"Tables extracted: {len(self.tables_info)}")
        logger.info(f"Output directory: {self.output_dir}")

        pages_with_text = {item['page_num'] for item in self.text_contents}
        pages_with_images = {item['page_num'] for item in self.images_info}
        pages_with_tables = {item['page_num'] for item in self.tables_info}

        logger.info(f"Pages with text: {len(pages_with_text)}")
        logger.info(f"Pages with images: {len(pages_with_images)}")
        logger.info(f"Pages with tables: {len(pages_with_tables)}")
        logger.info("=" * 50)

    # async def _process_embeddings(self) -> bool:
    #         """Process text and table embeddings with optimized database operations"""
    #         model_unique_name = await KbotMdModelsRepository().get_unique_name_by_id(
    #             self.file_params.txt_embed_model # type: ignore
    #         )
    #         if not model_unique_name:
    #             msg = f"Embedding model not found for id: {self.file_params.txt_embed_model}"
    #             logger.error(msg)
    #             await self._update_file_status(FileStatus.PARSE_FAILED, msg)
    #             return False

    #         # Process text embeddings
    #         text_embeddings = await self._create_text_embeddings(model_unique_name)
    #         if text_embeddings is None:
    #             return False

    #         # Process table embeddings
    #         table_embeddings = await self._create_table_embeddings(model_unique_name)
    #         if table_embeddings is None:
    #             return False

    #         return True

    # async def _create_text_embeddings(self, model_unique_name: str) -> list[KbotBizTxtEmbedding] | None:
    #     """Create embeddings for text content"""
    #     texts = [item['text'] for item in self.text_content if item['text'].strip()]
    #     if not texts:
    #         return []

    #     embeddings_list = await call_embedding_model(model_unique_name, texts)
    #     if not embeddings_list:
    #         msg = f"Embedding model {model_unique_name} returned None."
    #         logger.error(msg)
    #         await self._update_file_status(FileStatus.PARSE_FAILED, msg)
    #         return None

    #     embed_entities = []
    #     for idx, text_item in enumerate(self.text_content):
    #         if not text_item['text'].strip():
    #             continue

    #         embed_entity = KbotBizTxtEmbedding(
    #             embed_id=str(uuid.uuid4()),
    #             chunk_doc=text_item['text'],
    #             chunk_metadata=json.dumps({
    #                 "chunk_type": ChunkType.TEXT,
    #                 "split_strategy": SplitStrategy.BY_PAGE.value,
    #                 "file_path": str(self.pdf_path),
    #                 "page_num": text_item['page_num']
    #             }),
    #             file_id=self.file_params.file_id,
    #             embedding=embeddings_list[idx].embedding
    #         )
    #         embed_entities.append(embed_entity)

    #     return await self._save_embeddings(embed_entities)

    # async def _create_table_embeddings(self, model_unique_name: str) -> list[KbotBizTxtEmbedding] | None:
    #     """Create embeddings for table content"""
    #     texts = []
    #     valid_tables = []

    #     for table in self.tables_info:
    #         if not self._is_table_valid(table['file_path']):
    #             continue

    #         with open(table['file_path'], 'r', encoding='utf-8') as f:
    #             table_text = f.read()
    #             if table_text.strip():
    #                 texts.append(table_text)
    #                 valid_tables.append(table)

    #     if not texts:
    #         return []

    #     embeddings_list = await call_embedding_model(model_unique_name, texts)
    #     if not embeddings_list:
    #         msg = f"Embedding model {model_unique_name} returned None."
    #         logger.error(msg)
    #         await self._update_file_status(FileStatus.PARSE_FAILED, msg)
    #         return None

    #     embed_entities = []
    #     for idx, table in enumerate(valid_tables):
    #         with open(table['file_path'], 'r', encoding='utf-8') as f:
    #             table_text = f.read()
    #             if not table_text.strip():
    #                 continue

    #             embed_entity = KbotBizTxtEmbedding(
    #                 embed_id=str(uuid.uuid4()),
    #                 chunk_doc=table_text,
    #                 chunk_metadata=json.dumps({
    #                     "chunk_type": ChunkType.TEXT,
    #                     "split_strategy": SplitStrategy.BY_PAGE.value,
    #                     "file_path": str(self.pdf_path),
    #                     "page_num": table['page_num']
    #                 }),
    #                 file_id=self.file_params.file_id,
    #                 embedding=embeddings_list[idx].embedding
    #             )
    #             embed_entities.append(embed_entity)

    #     return await self._save_embeddings(embed_entities)


async def process_pdf(file_params: FileParams) -> bool:
    """
    Process PDF file by extracting content and generating embeddings

    Args:
        file_params: File parameters including path and processing options

    Returns:
        bool: True if processing succeeded, False otherwise
    """
    if not check_text_file(file_params):
        return False

    try:
        logger.info(f"Processing PDF file: {file_params.file_path}")
        parser = PDFPlumberParser(file_params)
        r = await parser.parse()
        if r:
            msg = f"Successfully parsed {file_params.file_path} (file id: {file_params.file_id})"
            await KbotMdKbFilesRepository().update_file_status(
                file_params.file_id,
                FileStatus.PARSED,
                str(msg)
            )
            return True
        else:
            msg = f"Failed to parse {file_params.file_path} (file id: {file_params.file_id})"
            await KbotMdKbFilesRepository().update_file_status(
                file_params.file_id,
                FileStatus.PARSE_FAILED,
                str(msg)
            )
            return False
    except Exception as e:
        msg = f"Error processing PDF file: {file_params.file_path}, error: {str(e)}"
        logger.error(msg)
        await KbotMdKbFilesRepository().update_file_status(
            file_params.file_id,
            FileStatus.PARSE_FAILED,
            msg
        )
        return False