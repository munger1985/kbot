import re

def sanitize_text_for_json(text: str, keep_newline: bool = False) -> str:
    """
    Sanitize special characters from text to make it suitable for storage in JSON fields.

    Removes control characters, zero-width characters, BOM markers, while handling
    newlines based on configuration. Preserves quotes and escapes them properly.

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

    # DO NOT remove quotes or backslashes - they will be properly escaped by json.dumps()
    # Removing them breaks JSON structure

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


def sanitize_dict_for_oracle_json(data: dict | None, max_length: int | None = None) -> dict:
    """
    Sanitize a dictionary for Oracle JSON field storage.

    Recursively cleans all string values in the dictionary to ensure they're safe
    for Oracle JSON storage. Handles nested dictionaries and lists.

    Args:
        data: Dictionary to sanitize (can be None)
        max_length: Optional maximum length limit for string values

    Returns:
        Sanitized dictionary with all string values cleaned
    """
    if not data or not isinstance(data, dict):
        return data or {}

    result = {}
    for key, value in data.items():
        # Sanitize key as well
        clean_key = sanitize_text_for_json(key, keep_newline=False)

        # Skip empty keys after sanitization
        if not clean_key:
            continue

        if value is None:
            result[clean_key] = None
        elif isinstance(value, str):
            result[clean_key] = sanitize_text_for_oracle_json(value, max_length)
        elif isinstance(value, dict):
            result[clean_key] = sanitize_dict_for_oracle_json(value, max_length)
        elif isinstance(value, list):
            result[clean_key] = sanitize_list_for_oracle_json(value, max_length)
        elif isinstance(value, (int, float, bool)):
            result[clean_key] = value
        else:
            # Unknown types - convert to string and sanitize
            try:
                str_val = str(value)
                result[clean_key] = sanitize_text_for_oracle_json(str_val, max_length)
            except Exception:
                # If conversion fails, use empty string
                result[clean_key] = ""

    return result


def sanitize_list_for_oracle_json(data: list | None, max_length: int | None = None) -> list:
    """
    Sanitize a list for Oracle JSON field storage.

    Recursively cleans all string values in the list to ensure they're safe
    for Oracle JSON storage. Handles nested dictionaries and lists.

    Args:
        data: List to sanitize (can be None)
        max_length: Optional maximum length limit for string values

    Returns:
        Sanitized list with all string values cleaned
    """
    if not data or not isinstance(data, list):
        return data or []

    result = []
    for item in data:
        if item is None:
            result.append(None)
        elif isinstance(item, str):
            result.append(sanitize_text_for_oracle_json(item, max_length))
        elif isinstance(item, dict):
            result.append(sanitize_dict_for_oracle_json(item, max_length))
        elif isinstance(item, list):
            result.append(sanitize_list_for_oracle_json(item, max_length))
        elif isinstance(item, (int, float, bool)):
            result.append(item)
        else:
            # Unknown types - convert to string and sanitize
            try:
                str_val = str(item)
                result.append(sanitize_text_for_oracle_json(str_val, max_length))
            except Exception:
                # If conversion fails, use empty string
                result.append("")

    return result


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