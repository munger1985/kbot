import re

def sanitize_text_for_json(text: str, keep_newline: bool = False) -> str:
    """
    清理文本中的特殊字符，使其适合存储到JSON字段中
    
    Args:
        text: 原始文本
        keep_newline: 是否保留换行符（转换为空格），默认为False（完全移除）
    
    Returns:
        清理后的文本
    """
    if not text or not isinstance(text, str):
        return text or ""
    
    # 移除所有控制字符（ASCII 0-31，127是DEL）
    # 包括：\n \r \t \b \f \v 等
    if keep_newline:
        # 只保留换行符，将其转换为空格
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # 将换行符、回车符统一替换为空格
        cleaned = cleaned.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    else:
        # 完全移除所有控制字符
        cleaned = re.sub(r'[\x00-\x1f\x7f]', '', text)
    
    # 移除零宽字符和Unicode控制字符
    cleaned = re.sub(r'[\u200b-\u200f\u2028-\u202f]', '', cleaned)
    
    # 移除BOM标记
    cleaned = cleaned.replace('\ufeff', '')
    
    # 移除可能破坏JSON的字符（反斜杠和引号）
    # 如果需要保留这些字符，可以用转义，但既然要去掉就一起去掉
    cleaned = cleaned.replace('\\', '').replace('"', '').replace("'", "")
    
    # 将多个连续空格合并为一个
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # 去除首尾空格
    cleaned = cleaned.strip()
    
    return cleaned


def sanitize_text_for_oracle_json(text: str, max_length: int = None) -> str:
    """
    专为Oracle JSON字段设计的文本清理方法
    
    Args:
        text: 原始文本
        max_length: 可选的最大长度限制，超过会截断
    
    Returns:
        清理后的文本
    """
    if not text or not isinstance(text, str):
        return text or ""
    
    # 1. 清理特殊字符
    cleaned = sanitize_text_for_json(text, keep_newline=False)
    
    # 2. 如果指定了最大长度，进行截断（注意中文字符）
    if max_length and cleaned:
        # 简单截断（不考虑完整字符，适用于大部分场景）
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
    
    return cleaned


# 使用示例
if __name__ == "__main__":
    # 测试用例
    test_cases = [
        "正常文本",
        "第一行\n第二行\r\n第三行",
        "带\t制表符\t的文本",
        "有\"引号\"和\\反斜杠\\的文本",
        "包含特殊控制字符\x00\x01\x02的文本",
        "Unicode控制字符\u200b\u200c\u200d",
        "  前后有空格  ",
        "多个    连续    空格",
        "",  # 空字符串
        None,  # None值
    ]
    
    for i, text in enumerate(test_cases):
        result = sanitize_text_for_oracle_json(text)
        print(f"测试{i+1}:")
        print(f"  原始: {repr(text)}")
        print(f"  清理后: {repr(result)}")
        print()