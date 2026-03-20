import re
from docling_core.types.doc.document import SectionHeaderItem


class SemanticLevelCorrector:
    def __init__(self):
        # 重新整理模式：从最显著的 L1 到 细分的 L2/L3
        self.patterns = [
            # Level 1: 绝对的顶级导航 (第X部分, 附件X)
            (re.compile(r'^第[一二三四五六七八九十\d]+[部分章节]\s*.*$'), 1),
            (re.compile(r'^附件[一二三四五六七八九十\d]*\s*[:：]?.*$'), 1),
            
            # Level 2: 条款主标题 (一、, 1., (一))
            (re.compile(r'^[一二三四五六七八九十]+\s*[.、：]'), 2),
            (re.compile(r'^[（\(][一二三四五六七八九十\d]+[）\)]'), 2),
            (re.compile(r'^第[一二三四五六七八九十\d]+[条款项]'), 2),
            
            # Level 3: 数字细分层级 (1.1, 1.1.1)
            (re.compile(r'^\d+\.\d+(\.\d+)?'), 3),
            # 只有孤立的数字 "1." 且后面跟了空格或字符，才作为二级或三级
            (re.compile(r'^\d+\s*[.、：]'), 2), 
        ]
        self.MAX_TITLE_LENGTH = 35

    def get_level(self, item, text: str) -> int | None:
        text = text.strip()
        
        # --- 新增：硬核过滤黑名单 ---
        # 1. 过滤掉长度太短且无意义的符号（如 "分；", "值：", "1."）
        if len(text) < 2 or text.endswith(('；', ';', '。')):
            return None
            
        # 2. 过滤掉明显的正文/回答开头
        if text.startswith(("答：", "答:", "注：", "注:", "说：", "说:")):
            return None
            
        # --- 原有正则逻辑 (确保 L1/L2 明确) ---
        for pattern, level in self.patterns:
            if pattern.match(text):
                return level
        
        # 3. 增强 Fallback：只有满足特定条件的才允许作为 L1
        if isinstance(item, SectionHeaderItem):
            # 只有长度适中且不含标点符号的才可能是真正的大标题
            if 2 <= len(text) <= 20 and "，" not in text:
                return 1
                
        return None