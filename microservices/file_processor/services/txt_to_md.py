import re
import os
from charset_normalizer import from_path

class TxtToMarkdownParser:
    """Converts plain text files to structured Markdown format with intelligent parsing.
    
    This parser handles:
    - Automatic encoding detection for Chinese text files
    - Paragraph reconstruction and line break normalization
    - Header and list item recognition (Chinese/English/numeric formats)
    - SQL code block detection and protection
    - Structural formatting to valid Markdown syntax
    
    Attributes:
        PATTERN_CN_HEADER: Regex pattern for Chinese numbered headers (e.g., 第一章, 一、)
        PATTERN_NUM_HEADER: Regex pattern for numeric hierarchical headers (e.g., 1., 1.1, 1.1.1)
        PATTERN_EN_HEADER: Regex pattern for uppercase English headers
        PATTERN_LIST_ITEM: Regex pattern for list items (numbered/bulleted)
        PATTERN_SQL_START: Regex pattern for SQL keyword detection
    """
    # Regex patterns for structural element recognition
    PATTERN_CN_HEADER = re.compile(r'^(第[一二三四五六七八九十百]+[章节部分]|[一二三四五六七八九十百]+[、]).*')
    PATTERN_NUM_HEADER = re.compile(r'^(\d+(\.\d+){0,3})\s+.*')
    PATTERN_EN_HEADER = re.compile(r'^[A-Z\s]{5,50}$')
    PATTERN_LIST_ITEM = re.compile(r'^(\d+\.|（?\d+）|[\-•·])\s*')
    # SQL keyword detection for code block protection
    PATTERN_SQL_START = re.compile(r'^(SELECT|INSERT|UPDATE|DELETE|CREATE|WITH|ALTER|DROP)\b', re.I)

    def __init__(self):
        """Initialize the TxtToMarkdownParser with predefined regex patterns."""
        pass

    def _detect_and_read(self, file_path: str) -> str:
        """Detect file encoding and read content with fallback mechanisms.
        
        Uses charset_normalizer for primary encoding detection, with fallback to
        common Chinese encodings (utf-8, gbk, gb18030) if detection fails.
        
        Args:
            file_path: Path to the text file to read
            
        Returns:
            Decoded text content as string
            
        Raises:
            ValueError: If no valid encoding can be detected for the file
        """
        results = from_path(file_path)
        best_guess = results.best()
        if not best_guess:
            for enc in ['utf-8', 'gbk', 'gb18030']:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        return f.read()
                except (UnicodeDecodeError, IOError):
                    continue
            raise ValueError(f"Unable to decode file: {file_path}")
        return str(best_guess)

    def _is_potential_code(self, line: str) -> bool:
        """Determine if a line contains SQL/code content to prevent misprocessing.
        
        Uses heuristic checks for SQL syntax features and keyword detection to
        identify code lines that should be preserved as-is.
        
        Args:
            line: Text line to analyze
            
        Returns:
            True if line appears to contain SQL/code, False otherwise
        """
        # Count SQL-specific syntax characters as heuristic
        sql_features = line.count('(') + line.count(')') + line.count(',') + line.count('`')
        # Consider code if multiple SQL features exist OR line starts with SQL keyword
        return sql_features > 3 or self.PATTERN_SQL_START.match(line.strip()) # type: ignore
    


    def _clean_and_join_paragraphs(self, text: str) -> str:
        """Clean text and reconstruct proper paragraphs while protecting code integrity.
        
        Removes control characters, normalizes line breaks, and intelligently joins
        lines into paragraphs based on sentence terminators. Protects SQL/code lines
        and structural elements (headers/lists) from being merged.
        
        Args:
            text: Raw text content to clean and restructure
            
        Returns:
            Cleaned text with proper paragraph breaks and preserved structural elements
        """
        # Remove non-printable control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Standardize line breaks across platforms
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        lines = text.split('\n')
        processed_lines = []
        buffer = ""

        # Precompile regex for sentence terminators (performance optimization)
        RE_END_PUNCT = re.compile(r'[。！？!?.”"]$')

        for line in lines:
            stripped = line.strip()
            
            # Handle empty lines (paragraph breaks)
            if not stripped:
                if buffer:
                    processed_lines.append(buffer)
                    buffer = ""
                processed_lines.append("") 
                continue
            
            # Check for structural elements or code that need protection
            is_header_or_list = (self.PATTERN_CN_HEADER.match(stripped) or 
                                 self.PATTERN_NUM_HEADER.match(stripped) or 
                                 self.PATTERN_LIST_ITEM.match(stripped))
            
            is_sql = self._is_potential_code(stripped)

            if buffer:
                # Force buffer flush if:
                # 1. Buffer exceeds safe length (prevent performance issues)
                # 2. Encountering structural element or SQL code
                if len(buffer) > 1000 or is_header_or_list or is_sql:
                    processed_lines.append(buffer)
                    buffer = stripped
                    continue
                
                # Join lines if buffer doesn't end with sentence terminator
                # Only check last 2 characters (significant performance optimization)
                if not RE_END_PUNCT.search(buffer[-2:]):
                    buffer += " " + stripped
                    continue
                else:
                    # End of paragraph - flush buffer
                    processed_lines.append(buffer)
                    buffer = stripped
            else:
                buffer = stripped
            
        # Flush any remaining content in buffer
        if buffer:
            processed_lines.append(buffer)
            
        return "\n".join(processed_lines)

    def _reconstruct_structure(self, text: str) -> str:
        """Convert cleaned text to properly formatted Markdown with structural elements.
        
        Identifies headers, list items, and SQL code blocks, then formats them
        according to Markdown syntax. Preserves empty lines for readability.
        
        Args:
            text: Cleaned text with proper paragraph breaks
            
        Returns:
            Structured Markdown-formatted text
        """
        lines = text.split('\n')
        md_lines = []
        
        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                md_lines.append("")
                continue

            # Wrap SQL code in Markdown code blocks (with length check to avoid false positives)
            if self._is_potential_code(clean_line) and len(clean_line) > 20:
                md_lines.append(f"```sql\n{clean_line}\n```")
                continue

            # Header formatting (length limit prevents misclassifying long code lines)
            is_header = False
            if len(clean_line) < 100: 
                if self.PATTERN_CN_HEADER.match(clean_line):
                    md_lines.append(f"## {clean_line}")
                    is_header = True
                elif self.PATTERN_NUM_HEADER.match(clean_line):
                    # Calculate header level (max level 4 for readability)
                    level = min(clean_line.count('.') + 2, 4)
                    md_lines.append(f"{'#' * level} {clean_line}")
                    is_header = True
                elif self.PATTERN_EN_HEADER.match(clean_line):
                    md_lines.append(f"# {clean_line}")
                    is_header = True
            
            if is_header:
                continue

            # Format list items as Markdown unordered lists
            if self.PATTERN_LIST_ITEM.match(clean_line):
                content = self.PATTERN_LIST_ITEM.sub("", clean_line).strip()
                md_lines.append(f"- {content}")
            else:
                # Regular text line - add as-is
                md_lines.append(clean_line)
        
        # Normalize multiple blank lines to improve readability
        return re.sub(r'\n{3,}', '\n\n', "\n".join(md_lines))

    def process(self, txt_path: str) -> str:
        """Main processing method to convert TXT file to structured Markdown.
        
        Executes the full parsing pipeline:
        1. Read and decode file with encoding detection
        2. Clean text and reconstruct paragraphs
        3. Apply Markdown structural formatting
        
        Args:
            txt_path: Path to the input text file
            
        Returns:
            Fully formatted Markdown string ready for further processing
        """
        raw_text = self._detect_and_read(txt_path)
        refined_text = self._clean_and_join_paragraphs(raw_text)
        return self._reconstruct_structure(refined_text)

    
