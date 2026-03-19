import re
from docling_core.types.doc.document import SectionHeaderItem


class SemanticLevelCorrector:
    """
    Semantic-based header level corrector for document structure analysis.
    Identifies true header elements by semantic patterns and filters out false positives.
    """
    def __init__(self):
        # More rigorous regex patterns with end-feature constraints
        self.patterns = [
            # Chapter/Section headers (typically top-level)
            (re.compile(r'^第[一二三四五六七八九十\d]+[章节]\s*.*$'), 1),
            # Article/Clause/Item (may be nested)
            (re.compile(r'^第[一二三四五六七八九十\d]+[条款项]'), 2),
            # Numeric hierarchical headers (e.g., 1, 1.1, 1.1.1)
            (re.compile(r'^\d+(\.\d+){0,2}\s+'), 1), 
        ]
        # Length threshold: "headers" exceeding this length are likely body content
        self.MAX_TITLE_LENGTH = 40 

    def get_level(self, item, text: str) -> int | None:
        """
        Determine semantic level of text element (header detection).
        
        Args:
            item: Document element object (from docling)
            text: Text content to analyze
            
        Returns:
            int | None: Semantic level (1/2) if header, None if body content
        """
        text = text.strip()
        if not text: 
            return None
        
        # --- Strong blocking logic ---
        # 1. Length filter: Legal/business document headers rarely exceed 30 characters
        if len(text) > 30: 
            return None
        
        # 2. Punctuation filter: Headers rarely end with full stop/semicolon or have multiple commas
        if text.endswith(("。", "；", "：")) or text.count("，") >= 2:
            return None

        # 3. Exclude sentences starting with pure date/data patterns
        if re.match(r'^\d+[月日年]', text): # Exclude "7月前...", "2023年..."
            return None

        # --- Original regex matching logic ---
        for pattern, level in self.patterns:
            if pattern.match(text):
                return level
                
        # Fallback: Only allow if Docling identifies as SectionHeaderItem and extremely short
        if isinstance(item, SectionHeaderItem) and len(text) < 15:
            return 1
        return None