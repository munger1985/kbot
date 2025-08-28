import re
import os
import uuid
import pandas as pd
import markdown
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import requests
from io import StringIO
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

import os
import json
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET
from datetime import datetime
import shutil


class MarkdownParser:
    def __init__(self, file_params: FileParams ):
        self.markdown_file = file_params.file_path
        self.base_dir = os.path.dirname(os.path.abspath(self.markdown_file))
        self.images_dir = os.path.join(self.base_dir, 'images')
        self.tables_dir = os.path.join(self.base_dir, 'tables')
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

    def extract_text_content(self, md_content):
        """提取纯文本内容"""
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, 'html.parser')

        # 移除表格和图片标签，只保留文本内容
        for element in soup.find_all(['table', 'img']):
            element.decompose()

        return soup.get_text().strip()

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
                            'local_path': save_path,
                            'filename': filename,
                            'alt_text': alt_text
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
                            'local_path': save_path,
                            'filename': filename,
                            'alt_text': alt_text
                        })
                    else:
                        print(f"本地图片不存在: {local_path}")

            except Exception as e:
                print(f"图片处理错误 ({img_url}): {e}")
                continue

        return saved_images

    def parse(self):
        """解析Markdown文件"""
        md_content = self.read_markdown()

        result = {
            'text_content': self.extract_text_content(md_content),
            'tables': self.extract_tables(md_content),
            'images': self.extract_and_save_images(md_content),
            'metadata': {
                'source_file': self.markdown_file,
                'images_directory': self.images_dir
            }
        }

        return result


async def process_markdown(file_params: FileParams) -> bool:
    """
    Process Excel file by extracting content and generating embeddings

    Args:
        file_params: File parameters including path and processing options

    Returns:
        bool: True if processing succeeded, False otherwise
    """
    if not check_text_file(file_params):
        return False

    try:
        logger.info(f"Processing Excel file: {file_params.file_path}")
        parser = MarkdownParser(file_params)
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
        msg = f"Error processing Excel file: {file_params.file_path}, error: {str(e)}"
        logger.error(msg)
        await KbotMdKbFilesRepository().update_file_status(
            file_params.file_id,
            FileStatus.PARSE_FAILED,
            msg
        )
        return False
def main():
    import sys
    import json

    if len(sys.argv) != 2:
        print("用法: python markdown_parser.py <markdown文件路径>")
        sys.exit(1)

    markdown_file = sys.argv[1]

    if not os.path.exists(markdown_file):
        print(f"文件不存在: {markdown_file}")
        sys.exit(1)

    parser = MarkdownParser(markdown_file)
    result = parser.parse()

    # 输出结果
    print("解析结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
