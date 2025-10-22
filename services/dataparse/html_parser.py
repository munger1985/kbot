import os
import base64
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image
import io
import re

class HTMLRAGParser:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        初始化HTML解析器
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.soup = None
        
    def parse_html(self, html_file_path: str) -> List[Dict[str, Any]]:
        """
        解析HTML文件的主要入口方法
        
        Args:
            html_file_path: HTML文件路径
            
        Returns:
            List of chunks, 每个chunk是一个字典
        """
        # 阶段一：HTML预处理与去噪
        self._load_and_clean_html(html_file_path)
        
        # 阶段二：语义块识别与提取
        chunks = []
        
        # 识别文档结构
        structure = self._identify_document_structure()
        
        # 按结构顺序处理内容
        for section in structure:
            section_chunks = self._process_section(section)
            chunks.extend(section_chunks)
            
        return chunks
    
    def _load_and_clean_html(self, html_file_path: str):
        """阶段一：加载并清理HTML"""
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        self.soup = BeautifulSoup(html_content, 'lxml')
        
        # 移除噪音标签
        noise_tags = ['script', 'style', 'meta', 'link', 'noscript']
        for tag in noise_tags:
            for element in self.soup.find_all(tag):
                element.decompose()
                
        # 移除特定的噪音元素（根据AWR报告结构调整）
        noise_selectors = ['#header', '.header', '#footer', '.footer', '.navigation', '.nav']
        for selector in noise_selectors:
            for element in self.soup.select(selector):
                element.decompose()
    
    def _identify_document_structure(self) -> List[Dict[str, Any]]:
        """识别文档结构，返回章节列表"""
        structure = []
        
        # 查找所有标题标签来构建文档结构
        headings = self.soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for heading in headings:
            section = {
                'title': heading.get_text().strip(),
                'level': int(heading.name[1]),
                'element': heading,
                'content': self._get_section_content(heading)
            }
            structure.append(section)
            
        return structure
    
    def _get_section_content(self, heading_element) -> List:
        """获取章节内容，直到下一个同级或更高级别的标题"""
        content_elements = []
        current = heading_element.next_sibling
        
        while current:
            if current.name and current.name.startswith('h'):
                current_level = int(current.name[1])
                if current_level <= int(heading_element.name[1]):
                    break
                    
            content_elements.append(current)
            current = current.next_sibling
            
        return content_elements
    
    def _process_section(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理单个章节，生成chunks"""
        chunks = []
        section_title = section['title']
        
        # 首先添加章节标题作为独立的文本chunk
        title_chunk = {
            'content': f"# {section_title}",
            'metadata': {
                'type': 'section_title',
                'section_title': section_title,
                'content_type': 'text',
                'chunk_id': f"title_{hash(section_title)}"
            }
        }
        chunks.append(title_chunk)
        
        # 处理章节内容
        for element in section['content']:
            if not hasattr(element, 'name'):
                continue
                
            if element.name == 'table':
                # 处理表格
                table_chunk = self._process_table(element, section_title)
                if table_chunk:
                    chunks.append(table_chunk)
                    
            elif element.name == 'img':
                # 处理图片
                img_chunk = self._process_image(element, section_title)
                if img_chunk:
                    chunks.append(img_chunk)
                    
            else:
                # 处理文本内容
                text_chunks = self._process_text_element(element, section_title)
                chunks.extend(text_chunks)
                
        return chunks
    
    def _process_table(self, table_element, section_title: str) -> Optional[Dict[str, Any]]:
        """阶段三：处理表格并转换为描述性文本"""
        try:
            # 使用pandas读取HTML表格
            dfs = pd.read_html(str(table_element))
            if not dfs:
                return None
                
            table_df = dfs[0]
            
            # 获取表格标题/上下文
            table_context = self._get_table_context(table_element)
            
            # 将DataFrame转换为描述性文本
            table_text = self._dataframe_to_descriptive_text(table_df, table_context)
            
            chunk = {
                'content': table_text,
                'metadata': {
                    'type': 'table',
                    'section_title': section_title,
                    'content_type': 'table',
                    'table_context': table_context,
                    'table_data': table_df.to_dict('records'),  # 保存原始数据用于后续处理
                    'chunk_id': f"table_{hash(str(table_df))}",
                    'shape': f"{table_df.shape[0]}x{table_df.shape[1]}"
                }
            }
            return chunk
            
        except Exception as e:
            print(f"处理表格时出错: {e}")
            return None
    
    def _get_table_context(self, table_element) -> str:
        """获取表格的上下文描述"""
        # 查找表格前面的标题或描述性文本
        context = ""
        
        # 查找前一个兄弟元素
        prev_element = table_element.previous_sibling
        while prev_element:
            if hasattr(prev_element, 'name'):
                if prev_element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    context = prev_element.get_text().strip()
                    break
                elif prev_element.name in ['p', 'div']:
                    text = prev_element.get_text().strip()
                    if len(text) > 10:  # 避免太短的文本
                        context = text
                        break
            prev_element = prev_element.previous_sibling
            
        return context if context else "Data Table"
    
    def _dataframe_to_descriptive_text(self, df: pd.DataFrame, context: str) -> str:
        """将DataFrame转换为描述性文本"""
        descriptive_text = f"[Table: {context}]\n"
        descriptive_text += f"Table Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
        
        # 添加列名
        descriptive_text += "Columns: " + ", ".join([str(col) for col in df.columns]) + "\n\n"
        
        # 添加前几行数据作为示例（避免文本过长）
        sample_rows = min(10, df.shape[0])
        for i in range(sample_rows):
            row_data = [str(df.iloc[i][col]) for col in df.columns]
            descriptive_text += f"Row {i+1}: " + " | ".join(row_data) + "\n"
            
        if df.shape[0] > sample_rows:
            descriptive_text += f"... and {df.shape[0] - sample_rows} more rows\n"
            
        return descriptive_text
    
    def _process_image(self, img_element, section_title: str) -> Optional[Dict[str, Any]]:
        """阶段三：处理图片并生成描述"""
        try:
            # 获取图片信息
            img_src = img_element.get('src', '')
            img_alt = img_element.get('alt', '')
            
            # 获取图片上下文
            img_context = self._get_image_context(img_element)
            
            # 模拟调用VLM生成图片描述
            img_description = self._generate_image_description(img_src, img_context)
            
            chunk = {
                'content': img_description,
                'metadata': {
                    'type': 'image',
                    'section_title': section_title,
                    'content_type': 'image',
                    'image_context': img_context,
                    'image_alt': img_alt,
                    'image_src': img_src,
                    'chunk_id': f"image_{hash(img_src)}",
                    'description_source': 'vlm'  # 标记描述来源
                }
            }
            return chunk
            
        except Exception as e:
            print(f"处理图片时出错: {e}")
            return None
    
    def _get_image_context(self, img_element) -> str:
        """获取图片的上下文描述"""
        context = img_element.get('alt', '')
        
        if not context:
            # 查找图片周围的文本
            parent = img_element.parent
            if parent and hasattr(parent, 'get_text'):
                sibling_text = parent.get_text().strip()
                if sibling_text and len(sibling_text) > 10:
                    context = sibling_text
                    
        return context if context else "Chart or Diagram"
    
    def _generate_image_description(self, img_src: str, context: str) -> str:
        """
        模拟调用VLM生成图片描述
        在实际应用中，这里会调用真正的多模态模型API
        """
        # 这里模拟VLM返回的描述
        # 实际实现中，需要：
        # 1. 如果是base64图片，解码并保存为临时文件
        # 2. 如果是URL或文件路径，读取图片
        # 3. 调用VLM API获取描述
        
        if img_src.startswith('data:image'):
            # Base64图片
            image_type = "embedded image"
        else:
            image_type = "external image"
            
        # 模拟VLM生成的描述
        vlm_description = f"[Image: {context}]\n"
        vlm_description += f"This is a {image_type} from the AWR report. "
        vlm_description += f"It shows performance metrics related to '{context}'. "
        vlm_description += "The chart displays trends over time with key data points marked. "
        vlm_description += "Peak values and important thresholds are visible in the visualization."
        
        return vlm_description
    
    def _process_text_element(self, element, section_title: str) -> List[Dict[str, Any]]:
        """处理文本元素，进行分块"""
        text_content = element.get_text().strip()
        if not text_content or len(text_content) < 10:
            return []
        
        # 简单的文本分块（实际应用中可以使用更复杂的分块策略）
        chunks = []
        words = text_content.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < 50:  # 跳过太短的chunk
                continue
                
            chunk_text = ' '.join(chunk_words)
            chunk = {
                'content': chunk_text,
                'metadata': {
                    'type': 'text',
                    'section_title': section_title,
                    'content_type': 'text',
                    'chunk_id': f"text_{hash(chunk_text)}",
                    'word_count': len(chunk_words)
                }
            }
            chunks.append(chunk)
            
        return chunks


# 使用示例
def main():
    # 初始化解析器
    parser = HTMLRAGParser(chunk_size=800, chunk_overlap=100)
    
    # 解析HTML文件
    html_file = "oracle_awr_report.html"  # 替换为你的HTML文件路径
    chunks = parser.parse_html(html_file)
    
    # 输出结果
    print(f"共生成 {len(chunks)} 个chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"类型: {chunk['metadata']['content_type']}")
        print(f"章节: {chunk['metadata']['section_title']}")
        print(f"内容预览: {chunk['content'][:200]}...")
        print(f"元数据: {chunk['metadata']}")
        
    return chunks

if __name__ == "__main__":
    chunks = main()