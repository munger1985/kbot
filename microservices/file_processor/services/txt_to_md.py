import re
import os
from charset_normalizer import from_path

class TxtToMarkdownParser:
    # 增加对代码块特征的识别
    PATTERN_CN_HEADER = re.compile(r'^(第[一二三四五六七八九十百]+[章节部分]|[一二三四五六七八九十百]+[、]).*')
    PATTERN_NUM_HEADER = re.compile(r'^(\d+(\.\d+){0,3})\s+.*')
    PATTERN_EN_HEADER = re.compile(r'^[A-Z\s]{5,50}$')
    PATTERN_LIST_ITEM = re.compile(r'^(\d+\.|（?\d+）|[\-•·])\s*')
    # 识别常见的 SQL 关键字，用于保护代码不被错误合并
    PATTERN_SQL_START = re.compile(r'^(SELECT|INSERT|UPDATE|DELETE|CREATE|WITH|ALTER|DROP)\b', re.I)

    def __init__(self):
        pass

    def _detect_and_read(self, file_path):
        # ... (保持原有的编码识别逻辑不变)
        results = from_path(file_path)
        best_guess = results.best()
        if not best_guess:
            for enc in ['utf-8', 'gbk', 'gb18030']:
                try:
                    with open(file_path, 'r', encoding=enc) as f: return f.read()
                except: continue
            raise ValueError(f"无法解析: {file_path}")
        return str(best_guess)

    def _is_potential_code(self, line):
        """判断是否可能是 SQL 或代码片段"""
        # 如果包含大量 SQL 特征符号且缩进明显，或者匹配 SQL 关键字
        sql_features = line.count('(') + line.count(')') + line.count(',') + line.count('`')
        return sql_features > 3 or self.PATTERN_SQL_START.match(line.strip())

    def _clean_and_join_paragraphs(self, text):
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        lines = text.split('\n')
        processed_lines = []
        buffer = ""

        # 预编译结束符正则，限制匹配范围以提高性能
        RE_END_PUNCT = re.compile(r'[。！？!?.”"]$')

        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                if buffer:
                    processed_lines.append(buffer)
                    buffer = ""
                processed_lines.append("") 
                continue
            
            # --- 核心改进：代码/长行保护 ---
            # 1. 限制 buffer 长度，防止无限合并导致正则卡死
            # 2. 如果当前行看起来像 SQL 语句，强制结束 buffer，将其作为独立行处理
            is_header_or_list = (self.PATTERN_CN_HEADER.match(stripped) or 
                                 self.PATTERN_NUM_HEADER.match(stripped) or 
                                 self.PATTERN_LIST_ITEM.match(stripped))
            
            is_sql = self._is_potential_code(stripped)

            if buffer:
                # 如果缓冲区过大（例如超过1000字）或者遇到 SQL/标题，强制刷出
                if len(buffer) > 1000 or is_header_or_list or is_sql:
                    processed_lines.append(buffer)
                    buffer = stripped
                    continue
                
                # 如果 buffer 结尾不是结束符，尝试合并
                if not RE_END_PUNCT.search(buffer[-2:]): # 只检查最后两个字符，极大提高性能
                    buffer += " " + stripped
                    continue
                else:
                    processed_lines.append(buffer)
                    buffer = stripped
            else:
                buffer = stripped
            
        if buffer:
            processed_lines.append(buffer)
            
        return "\n".join(processed_lines)

    def _reconstruct_structure(self, text):
        lines = text.split('\n')
        md_lines = []
        
        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                md_lines.append("")
                continue

            # 增加 SQL 代码块的简单 Markdown 包裹
            if self._is_potential_code(clean_line) and len(clean_line) > 20:
                md_lines.append(f"```sql\n{clean_line}\n```")
                continue

            # 标题识别（增加长度限制，防止超长 SQL 行被误判为标题）
            is_header = False
            if len(clean_line) < 100: 
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
            
            if is_header: continue

            if self.PATTERN_LIST_ITEM.match(clean_line):
                content = self.PATTERN_LIST_ITEM.sub("", clean_line).strip()
                md_lines.append(f"- {content}")
            else:
                md_lines.append(clean_line)
        
        return re.sub(r'\n{3,}', '\n\n', "\n".join(md_lines))

    def process(self, txt_path):
        raw_text = self._detect_and_read(txt_path)
        refined_text = self._clean_and_join_paragraphs(raw_text)
        return self._reconstruct_structure(refined_text)