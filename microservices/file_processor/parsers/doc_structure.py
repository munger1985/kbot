import re
from docling_core.types.doc.document import SectionHeaderItem


class SemanticLevelCorrector:
    def __init__(self):
        # 更加严谨的正则：增加对末尾特征的限制
        self.patterns = [
            (re.compile(r'^第[一二三四五六七八九十\d]+[章节]\s*.*$'), 1), # 章/节 通常是独立标题
            (re.compile(r'^第[一二三四五六七八九十\d]+[条款项]'), 2), # 条/款/项 可能嵌套
            (re.compile(r'^\d+(\.\d+){0,2}\s+'), 1), 
        ]
        # 长度阈值：超过此长度的“标题”大概率是正文内容
        self.MAX_TITLE_LENGTH = 40 

    def get_level(self, item, text: str) -> int | None:
        text = text.strip()
        if not text: return None
        
        # --- 强力阻断逻辑 ---
        # 1. 长度过滤：法律或业务文档标题很少超过 30 字
        if len(text) > 30: return None
        
        # 2. 标点过滤：标题末尾如果出现“。”、“；”或多个“，”，大概率是正文
        if text.endswith(("。", "；", "：")) or text.count("，") >= 2:
            return None

        # 3. 排除纯日期或纯数据开头的句子
        if re.match(r'^\d+[月日年]', text): # 排除 "7 月前..." "2023年..."
            return None

        # --- 原有正则逻辑 ---
        for pattern, level in self.patterns:
            if pattern.match(text):
                return level
                
        # 兜底：只有 Docling 认为是 SectionHeaderItem 且极其短才放行
        if isinstance(item, SectionHeaderItem) and len(text) < 15:
            return 1
        return None