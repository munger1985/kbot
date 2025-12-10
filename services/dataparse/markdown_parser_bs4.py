import re
import os
import json
import uuid
from pathlib import Path
import pandas as pd
import markdown
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import requests
from io import StringIO
from loguru import logger
from .file_params import FileParams
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from core.dictionary import FileStatus, ChunkType, SplitStrategy
from utils.call_models import CallModel
from .common import check_text_file, update_file_status, save_embeddings
from core.config.settings import get_prompt_config
from .summary_parser import SummaryParser


class MarkdownParser:
    def __init__(self, file_params: FileParams):
        self.markdown_file = file_params.file_path
        self.file_params = file_params
        self.base_dir = os.path.dirname(os.path.abspath(self.markdown_file))
        self.images_dir = os.path.join(self.base_dir, 'images')
        self.tables_dir = os.path.join(self.base_dir, 'tables')
        self.image_dict = []
        self.md = markdown.Markdown(extensions=['tables'])
        self.text_results = []
        self.prompt_config = get_prompt_config()
        self.create_dirs()

    def create_dirs(self):
        """创建存储目录"""
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        if not os.path.exists(self.tables_dir):
            os.makedirs(self.tables_dir)

    def read_markdown(self):
        """读取Markdown文件内容"""
        with open(self.markdown_file, 'r', encoding='utf-8') as f:
            return f.read()

    def _parse_paragraphs(self, text):
        """按照段落解析Markdown文本"""
        # 将markdown转换为HTML
        html = self.md.convert(text)
        soup = BeautifulSoup(html, 'html.parser')

        # 提取段落
        paragraphs = []
        for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol']):
            paragraphs.append(p.get_text().strip())

        return paragraphs

    def _parse_table(self, table_text):
        """
        解析markdown表格并转换为JSON，以表头作为key

        Args:
            table_text (str): Markdown表格文本

        Returns:
            list: 字典列表，每个字典代表一行，键为表头
        """
        # 将markdown表格转换为HTML
        html = self.md.convert(table_text)
        soup = BeautifulSoup(html, 'html.parser')

        table = soup.find('table')
        if not table:
            return []

        # 提取表头
        headers = []
        header_row = table.find('thead').find('tr')  # type: ignore
        for th in header_row.find_all('th'):  # type: ignore
            headers.append(th.get_text().strip())

        # 提取行数据
        result = []
        for tr in table.find('tbody').find_all('tr'):  # type: ignore
            row_data = {}
            cells = tr.find_all('td')  # type: ignore
            for i, cell in enumerate(cells):
                if i < len(headers):
                    row_data[headers[i]] = cell.get_text().strip()
            result.append(row_data)

        return result

    def extract_text_content(self, md_content):
        """提取纯文本内容"""
        html = markdown.markdown(md_content)
        # soup = BeautifulSoup(html, 'html.parser')
        text_results = {
            'paragraphs': [],
            'tables': []
        }

        # 首先，识别markdown中的表格部分
        table_pattern = r'(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n)+)'
        table_matches = re.finditer(table_pattern, md_content)

        # 提取表格及其位置
        table_positions = []
        for match in table_matches:
            start, end = match.span()
            table_positions.append((start, end, match.group(0)))

        # 处理文本，跳过表格部分
        last_end = 0
        for start, end, table_text in table_positions:
            # 添加表格前的文本
            if start > last_end:
                paragraphs = self._parse_paragraphs(md_content[last_end:start])
                # text_results['paragraphs'].extend(paragraphs)

            # 解析表格
            table_json = self._parse_table(table_text)
            text_results['tables'].append(table_json)

            last_end = end

        # 添加最后一个表格后的任何剩余文本
        if last_end < len(md_content):
            paragraphs = self._parse_paragraphs(md_content[last_end:])
            # text_results['titles'].extend(paragraphs)
        self.text_results = text_results

        blocks = self.parse_into_blocks(md_content)
        for i, block in enumerate(blocks):
            print(f"Block {i + 1}:")
            semanticChunk = f"Title: {block['title']} (Level {block['level']})\n" + f"Content: {block['content']}"
            text_results['paragraphs'].append(semanticChunk)
            # print(semanticChunk)

        return text_results

    def extract_tables(self, md_content):
        """提取表格并转换为JSON格式，保存到文件"""
        tables = []
        table_pattern = r'(\|.*\|[\r\n]+\|.*\|[\r\n]+(?:\|.*\|[\r\n]+)*)'
        table_matches = re.findall(table_pattern, md_content)

        for i, table_match in enumerate(table_matches):
            try:
                # 清理表格格式
                lines = table_match.strip().split('\n')
                cleaned_lines = []

                for line in lines:
                    line = line.strip()
                    if line.startswith('|') and line.endswith('|'):
                        cleaned_lines.append(line)

                if len(cleaned_lines) >= 2:
                    # 使用pandas解析表格
                    table_str = '\n'.join(cleaned_lines)
                    df = pd.read_csv(StringIO(table_str), sep='|').dropna(axis=1, how='all')
                    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

                    # 转换为JSON格式
                    table_json = {
                        'headers': df.columns.tolist(),
                        'rows': df.iloc[1:].values.tolist(),
                        'data': df.iloc[1:].to_dict('records')
                    }

                    # 生成随机ID并保存表格到文件
                    table_id = f"table_{uuid.uuid4().hex}"
                    table_filename = f"{table_id}.json"
                    table_path = os.path.join(self.tables_dir, table_filename)

                    import json
                    with open(table_path, 'w', encoding='utf-8') as f:
                        json.dump(table_json, f, indent=2, ensure_ascii=False)

                    tables.append({
                        'id': table_id,
                        'file_path': table_path,
                        'absolute_path': os.path.abspath(table_path),
                        'data': table_json
                    })
            except Exception as e:
                print(f"表格解析错误: {e}")
                continue

        return tables

    def parse_into_blocks(self, text):
        """
        将Markdown文本解析成标题和标题后的内容块

        Args:
            text (str): 要解析的Markdown文本

        Returns:
            list: 包含标题和内容的块列表
        """
        # 使用正则表达式匹配标题
        header_pattern = r'^(#{1,6})\s+(.+?)$'

        # 按行分割文本
        lines = text.split('\n')
        blocks = []
        current_title = ""
        current_level = 0
        current_content = []

        for line in lines:
            header_match = re.match(header_pattern, line)

            if header_match:
                # 如果已有内容，保存当前块
                if current_title or current_content:
                    blocks.append({
                        "title": current_title,
                        "level": current_level,
                        "content": "\n".join(current_content)
                    })

                # 开始新块
                current_level = len(header_match.group(1))
                current_title = header_match.group(2).strip()
                current_content = []
            else:
                # 将非标题行添加到当前内容中
                if line.strip():  # 忽略空行
                    current_content.append(line)

        # 添加最后一个块
        if current_title or current_content:
            blocks.append({
                "title": current_title,
                "level": current_level,
                "content": "\n".join(current_content)
            })

        return blocks

    def extract_and_save_images(self, md_content):
        """提取图片并保存到本地"""
        # 匹配完整的图片标记，包括alt文本和URL
        image_pattern = r'!\[(.*?)\]\((.*?)\)'
        image_matches = re.findall(image_pattern, md_content)
        saved_images = []

        for alt_text, img_url in image_matches:
            try:
                # 获取URL中的文件名（最后一个/后面的部分）
                url_filename = img_url.split('/')[-1].split('?')[0]  # 去除查询参数

                # 组合文件名：alt文本 + "_" + URL文件名
                if alt_text.strip() and url_filename.strip():
                    filename = f"{alt_text}_{url_filename}"
                elif alt_text.strip():
                    filename = alt_text
                elif url_filename.strip():
                    filename = url_filename
                else:
                    # 如果都没有，使用随机文件名
                    file_extension = os.path.splitext(img_url)[1] if '.' in img_url.split('?')[0] else '.jpg'
                    filename = f"{uuid.uuid4().hex}{file_extension}"

                # 清理文件名中的非法字符
                filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
                filename = filename.replace(' ', '_')

                save_path = os.path.join(self.images_dir, filename)

                # 处理图片URL
                parsed_url = urlparse(img_url)

                if parsed_url.scheme in ['http', 'https']:
                    # 下载网络图片
                    response = requests.get(img_url, stream=True, timeout=10)
                    if response.status_code == 200:
                        with open(save_path, 'wb') as f:
                            for chunk in response.iter_content(1024):
                                f.write(chunk)
                        saved_images.append({
                            'original_url': img_url,
                            'file_path': save_path,
                            'filename': filename,
                            'alt_text': alt_text,
                            'image_id': str(uuid.uuid4())
                        })
                    else:
                        print(f"图片下载失败: {img_url} (状态码: {response.status_code})")
                else:
                    # 处理本地图片路径（相对路径或绝对路径）
                    local_path = img_url
                    if not os.path.isabs(local_path):
                        # 如果是相对路径，转换为绝对路径
                        local_path = os.path.join(self.base_dir, local_path)

                    if os.path.exists(local_path):
                        import shutil
                        shutil.copy2(local_path, save_path)
                        saved_images.append({
                            'original_path': img_url,
                            'file_path': save_path,
                            'filename': filename,
                            'alt_text': alt_text,
                            'image_id': str(uuid.uuid4())

                        })
                    else:
                        print(f"本地图片不存在: {local_path}")

            except Exception as e:
                print(f"图片处理错误 ({img_url}): {e}")
                continue
        self.image_dict = saved_images
        return saved_images

    async def _process_images_embeddings(self) -> list:
        ## 1 means yes
        if self.file_params.img2txt == 1:

            # if self.file_params.parser.get("extract_images", False):
            vlm_prompt_unique_name = self.prompt_config.image2text

            if self.file_params.img2txt_model is None:
                msg = f"img2txt_model not found for id: {self.file_params.img2txt_model}"
                logger.error(msg)
                await update_file_status(self.file_params.file_id, FileStatus.PARSE_FAILED, msg)
                return []

            chunks = []
            chunk_metas = []
            img_chunk_num=1
            for eachImage in self.image_dict:
                description_file = Path(eachImage['file_path'] + ".description")
                if not description_file.exists():

                    image_description = await CallModel().call_vlm_model_for_parsing_picture(
                        self.file_params.img2txt_model,
                        vlm_prompt_unique_name,
                        eachImage['file_path'])
                    if image_description:
                        description_file.write_text(
                            image_description,
                            encoding='utf-8'
                        )
                        chunk_metas.append({
                            "chunk_type": ChunkType.IMAGE,
                            "image_id": eachImage['image_id'],
                            "page_num": 1,
                            'chunk_num': img_chunk_num,

                        })
                        img_chunk_num+=1
                        chunks.append(image_description)

            if not self.file_params.txt_embed_model:
                msg = f"text_embedding_model not found for id: {self.file_params.txt_embed_model}"
                logger.error(msg)
                await  update_file_status(self.file_params.file_id, FileStatus.PARSE_FAILED, msg)
                return []
            embeddings_list = []
            if chunks:
                embeddings_list = await CallModel().call_embedding_model(self.file_params.txt_embed_model, chunks)
            if embeddings_list and len(embeddings_list) != len(chunks):
                msg = f"text_embedding_model  {self.file_params.txt_embed_model} returned invalid results (expected {len(chunks)}, got {len(embeddings_list) if embeddings_list else 0})"
                logger.error(msg)
                logger.error("failed file: {}", self.file_params.file_path)
                await  update_file_status(self.file_params.file_id, FileStatus.PARSE_FAILED, msg)
                return []

            # Create embedding entities
            embed_entities = []
            for idx, (chunk, meta) in enumerate(zip(chunks, chunk_metas)):
                embed_entity = KbotBizTxtEmbedding(
                    kb_id=self.file_params.kb_id,
                    embed_id=meta['image_id'],
                    chunk_doc=chunk,
                    chunk_metadata=meta,  # type: ignore
                    biz_metadata=self.file_params.biz_metadata,
                    file_id=self.file_params.file_id,
                    embedding=embeddings_list[idx].embedding,  # type: ignore
                    security_level=self.file_params.security_level,
                    status=1
                )
                embed_entities.append(embed_entity)

            # Save all embeddings in one batch
            return embed_entities
        else:
            return []

    async def _process_embeddings(self) -> bool:
        """Process text and table embeddings for by fixed size"""

        if not self.file_params.txt_embed_model:
            msg = f"Embedding model not found for id: {self.file_params.txt_embed_model}"
            logger.error(msg)
            await update_file_status(self.file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False

        # Prepare all content chunks for embedding
        chunks = []
        chunk_metas = []
        paragraph_chunk_num=1
        table_chunk_num=1
        # Add table content
        for paragraph in self.text_results['paragraphs']:  # type: ignore
            chunks.append(paragraph)
            chunk_metas.append({
                "chunk_type": ChunkType.TEXT,
                "page_num": 1,
                'chunk_num': paragraph_chunk_num,

            })
            paragraph_chunk_num+=1
        for table in self.text_results['tables']:  # type: ignore
            table_str = json.dumps(table, ensure_ascii=False, indent=2)
            chunks.append(table_str)
            chunk_metas.append({
                "chunk_type": ChunkType.TABLE,
                "page_num": 1,
                'chunk_num': table_chunk_num,

            })
            table_chunk_num+=1

        if not chunks:
            logger.warning("No valid content chunks found for embedding")
            return True  # Consider empty content as success

        # Get all embeddings in one call
        embeddings_list = await CallModel().call_embedding_model(self.file_params.txt_embed_model, chunks)
        if not embeddings_list or len(embeddings_list) != len(chunks):
            msg = f"Embedding model {self.file_params.txt_embed_model} returned invalid results (expected {len(chunks)}, got {len(embeddings_list) if embeddings_list else 0})"
            logger.error(msg)
            await update_file_status(self.file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False

        # Create embedding entities
        embed_entities = []
        for idx, (chunk, meta) in enumerate(zip(chunks, chunk_metas)):
            embed_entity = KbotBizTxtEmbedding(
                kb_id=self.file_params.kb_id,
                embed_id=str(uuid.uuid4()),
                chunk_doc=chunk,
                chunk_metadata=meta,  # type: ignore
                biz_metadata=self.file_params.biz_metadata,
                file_id=self.file_params.file_id,
                embedding=embeddings_list[idx].embedding,
                security_level=self.file_params.security_level,
                status=1
            )
            embed_entities.append(embed_entity)
        image_embed_entities = await self._process_images_embeddings()
        embed_entities.extend(image_embed_entities)
        if self.file_params.enable_summary:
            logger.debug("启用摘要处理")
            summary_result = await SummaryParser.process_summary(file_params=self.file_params,
                                                                 embed_entities=embed_entities)

        # Save all embeddings in one batch
        return await save_embeddings(self.file_params, embed_entities)

    async def parse(self):
        """解析Markdown文件"""
        split_strategy = int(self.file_params.parser.get("split_strategy", SplitStrategy.DOC_STRUCTURE.value))
        file_repo = KbotMdKbFilesRepository()
        if split_strategy == SplitStrategy.DOC_STRUCTURE.value:

            md_content = self.read_markdown()
            self.extract_text_content(md_content)
            metadata = {
                'tables': self.extract_tables(md_content),
                'images': self.extract_and_save_images(md_content),
                'metadata': {
                    'source_file': self.markdown_file,
                    # 'images_directory': self.images_dir
                }
            }
            if not await self._process_embeddings():
                return False
            json_str = json.dumps(metadata, ensure_ascii=False, indent=2)

            await file_repo.update_file_parsed_metadata(self.file_params.file_id, json_str)

            return True
        else:
            logger.warning(f"Unrecognized split strategy: {split_strategy}")
            return False


async def process_markdown(file_params: FileParams) -> bool:
    """
    Process Excel file by extracting content and generating embeddings

    Args:
        file_params: File parameters including path and processing options

    Returns:
        bool: True if processing succeeded, False otherwise
    """
    if not await check_text_file(file_params):
        return False

    try:
        logger.info(f"Processing Markdown file: {file_params.file_path}")
        parser = MarkdownParser(file_params)
        r = await parser.parse()
        if r:
            msg = f"Successfully parsed {file_params.file_path} (file id: {file_params.file_id})"
            await update_file_status(
                file_params.file_id,
                FileStatus.PARSED,
                str(msg)
            )
            return True
        else:
            msg = f"Failed to parse {file_params.file_path} (file id: {file_params.file_id})"
            await update_file_status(
                file_params.file_id,
                FileStatus.PARSE_FAILED,
                str(msg)
            )
            return False
    except Exception as e:
        msg = f"Error processing Markdown file: {file_params.file_path}, error: {str(e)}"
        logger.exception(e)
        logger.error(msg)
        await update_file_status(
            file_params.file_id,
            FileStatus.PARSE_FAILED,
            msg
        )
        return False
# def main():
#     import sys
#     import json
#
#     if len(sys.argv) != 2:
#         print("用法: python markdown_parser.py <markdown文件路径>")
#         sys.exit(1)
#
#     markdown_file = sys.argv[1]
#
#     if not os.path.exists(markdown_file):
#         print(f"文件不存在: {markdown_file}")
#         sys.exit(1)
#
#     parser = MarkdownParser(markdown_file)
#     result = parser.parse()
#
#     # 输出结果
#     print("解析结果:")
#     print(json.dumps(result, indent=2, ensure_ascii=False))
