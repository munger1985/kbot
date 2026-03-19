import re

def sanitize_text_for_json(text: str, keep_newline: bool = False) -> str:
    """
    Sanitize special characters from text to make it suitable for storage in JSON fields.
    
    Removes control characters, zero-width characters, BOM markers, and problematic
    JSON characters (backslashes, quotes) while handling newlines based on configuration.

    Args:
        text: Raw input text to sanitize
        keep_newline: Whether to preserve newlines (convert to spaces) - default False (remove entirely)
    
    Returns:
        Sanitized text ready for JSON storage
    """
    if not text or not isinstance(text, str):
        return text or ""
    
    # Remove all control characters (ASCII 0-31, 127 is DEL)
    # Includes: \n \r \t \b \f \v etc.
    if keep_newline:
        # Keep only newlines, convert them to spaces
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Normalize newlines/carriage returns to single space
        cleaned = cleaned.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    else:
        # Remove all control characters completely
        cleaned = re.sub(r'[\x00-\x1f\x7f]', '', text)
    
    # Remove zero-width characters and Unicode control characters
    cleaned = re.sub(r'[\u200b-\u200f\u2028-\u202f]', '', cleaned)
    
    # Remove BOM (Byte Order Mark)
    cleaned = cleaned.replace('\ufeff', '')
    
    # Remove characters that may break JSON (backslashes and quotes)
    # These can be escaped instead of removed if preservation is needed
    cleaned = cleaned.replace('\\', '').replace('"', '').replace("'", "")
    
    # Collapse multiple consecutive spaces into single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Trim leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def sanitize_text_for_oracle_json(text: str, max_length: int | None = None) -> str:
    """
    Specialized text sanitization method for Oracle JSON fields.
    
    Extends the base JSON sanitization with length limiting for Oracle database
    compatibility, handling both special characters and field size constraints.

    Args:
        text: Raw input text to sanitize
        max_length: Optional maximum length limit - text will be truncated if exceeded
    
    Returns:
        Sanitized text optimized for Oracle JSON storage
    """
    if not text or not isinstance(text, str):
        return text or ""
    
    # 1. Sanitize special characters using base method
    cleaned = sanitize_text_for_json(text, keep_newline=False)
    
    # 2. Truncate if max length specified (handles Chinese characters appropriately)
    if max_length and cleaned:
        # Simple truncation (doesn't preserve full characters - suitable for most scenarios)
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
    
    return cleaned


# Usage examples
if __name__ == "__main__":
    # Test cases
    test_cases = [
        "Normal text",
        "Line 1\nLine 2\r\nLine 3",
        "Text with\ttabs\tincluded",
        'Text with "quotes" and \\backslashes\\',
        "Text with special control chars\x00\x01\x02",
        "Unicode control chars\u200b\u200c\u200d",
        "  Text with leading/trailing spaces  ",
        "Text with    multiple    spaces",
        "",  # Empty string
        None,  # None value
    ]
    
    for i, text in enumerate(test_cases):
        result = sanitize_text_for_oracle_json(text)
        print(f"Test {i+1}:")
        print(f"  Original: {repr(text)}")
        print(f"  Sanitized: {repr(result)}")
        print()