import tiktoken
from loguru import logger

def truncate_by_token(text: str, max_tokens: int = 7000, model_name: str = "gpt-3.5-turbo") -> str:
    """
    针对 Embedding 限制，按行截断 Markdown 表格，确保不超限且行完整。
    默认 7500 是为了给 Prompt 其他部分留出空间（模型上限通常是 8192）。
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base") # 默认编码

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    logger.warning(f"Text length ({len(tokens)}) exceeds max_tokens ({max_tokens}). Truncating...")

    lines = text.split('\n')
    truncated_lines = []
    current_tokens = 0
    
    # 预留给 "[Table Truncated...]" 的 token 数
    suffix = "\n\n[Note: Table truncated due to length constraints.]"
    max_tokens -= len(encoding.encode(suffix))

    for line in lines:
        line_tokens = len(encoding.encode(line + '\n'))
        if current_tokens + line_tokens > max_tokens:
            break
        truncated_lines.append(line)
        current_tokens += line_tokens

    return "\n".join(truncated_lines) + suffix