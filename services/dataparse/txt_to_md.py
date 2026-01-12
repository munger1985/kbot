import re
import os
from pathlib import Path
from charset_normalizer import from_path

class TxtToMarkdownParser:
    # 预编译正则：提高 RAG 预处理大规模文档时的性能
    PATTERN_CN_HEADER = re.compile(r'^(第[一二三四五六七八九十百]+[章节部分]|[一二三四五六七八九十百]+[、]).*')
    PATTERN_NUM_HEADER = re.compile(r'^(\d+(\.\d+){0,3})\s+.*')
    PATTERN_EN_HEADER = re.compile(r'^[A-Z\s]{5,50}$')
    PATTERN_LIST_ITEM = re.compile(r'^(\d+\.|（?\d+）|[\-•·])\s*')

    def __init__(self):
        pass

    def _detect_and_read(self, file_path):
        """1. 编码自动识别：确保不同来源的 TXT 都能正确解析语义"""
        try:
            results = from_path(file_path)
            best_guess = results.best()
            if not best_guess:
                # 兜底逻辑：尝试 UTF-8 和 GBK
                for enc in ['utf-8', 'gbk', 'gb18030']:
                    try:
                        with open(file_path, 'r', encoding=enc) as f:
                            return f.read()
                    except:
                        continue
                raise ValueError(f"无法解析文件编码: {file_path}")
            return str(best_guess)
        except Exception as e:
            raise IOError(f"读取文件失败 {file_path}: {str(e)}")

    def _clean_and_join_paragraphs(self, text):
        """
        2. 语义合并优化：
        解决 TXT 常见的硬换行问题，确保一个完整的句子在一个 Chunk 内。
        """
        # 移除不可见字符，归一化换行
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        lines = text.split('\n')
        processed_lines = []
        buffer = ""

        for line in lines:
            stripped = line.strip()
            # 如果是空行，说明是物理分段
            if not stripped:
                if buffer:
                    processed_lines.append(buffer)
                    buffer = ""
                processed_lines.append("") 
                continue
            
            # 判断是否需要合并：
            # 如果 buffer 结尾不是结束符，且当前行不是新标题或列表，则合并
            if buffer and not re.search(r'[。！？!?.”"]$', buffer):
                if not (self.PATTERN_CN_HEADER.match(stripped) or 
                        self.PATTERN_NUM_HEADER.match(stripped) or 
                        self.PATTERN_LIST_ITEM.match(stripped)):
                    buffer += " " + stripped # 添加空格防止单词粘连
                    continue
            
            if buffer:
                processed_lines.append(buffer)
            buffer = stripped
            
        if buffer:
            processed_lines.append(buffer)
            
        return "\n".join(processed_lines)

    def _reconstruct_structure(self, text):
        """3. 结构重建：将平面文本转为有层级的 Markdown"""
        lines = text.split('\n')
        md_lines = []
        
        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                md_lines.append("")
                continue

            # 标题识别逻辑（影响 RAG 切分后的 Metadata 准确性）
            is_header = False
            if len(clean_line) < 60:
                if self.PATTERN_CN_HEADER.match(clean_line):
                    md_lines.append(f"## {clean_line}")
                    is_header = True
                elif self.PATTERN_NUM_HEADER.match(clean_line):
                    level = min(clean_line.count('.') + 2, 4)
                    md_lines.append(f"{'#' * level} {clean_line}")
                    is_header = True
                elif self.PATTERN_EN_HEADER.match(clean_line):
                    md_lines.append(f"# {clean_line}")
                    is_header = True
            
            if is_header:
                continue

            # 列表识别（对召回并列事实类信息很有帮助）
            if self.PATTERN_LIST_ITEM.match(clean_line):
                content = self.PATTERN_LIST_ITEM.sub("", clean_line).strip()
                md_lines.append(f"- {content}")
            else:
                md_lines.append(clean_line)
        
        return re.sub(r'\n{3,}', '\n\n', "\n".join(md_lines))

    def process(self, txt_path):
        """
        主处理入口
        :param txt_path: 固定的本地文件路径
        :return: Markdown 格式的字符串
        """
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"未找到文件: {txt_path}")

        # 1. 磁盘读取（内存化第一步）
        raw_text = self._detect_and_read(txt_path)
        
        # 2. 清洗与语义合并
        refined_text = self._clean_and_join_paragraphs(raw_text)
        
        # 3. 结构化处理
        markdown_content = self._reconstruct_structure(refined_text)
        
        # 4. 直接返回内存中的字符串对象，不再写入磁盘
        return markdown_content

# --- 使用示例 ---
if __name__ == "__main__":
    parser = TxtToMarkdownParser()
    try:
        # 这里传入真实的物理路径
        file_path = "/mnt/j/docs/paper.txt" 
        # 直接获取转换后的内容进行后续 Embedding 或切分
        md_text = parser.process(file_path)
        
        print("--- 转换成功，预览前100字符 ---")
        print(md_text[:1000])
    except Exception as e:
        print(f"处理失败: {e}")