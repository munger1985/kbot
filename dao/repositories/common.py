from loguru import logger

async def es_in_date_format(date_str: str) -> str:
    """
    将日期字符串转换为ES兼容的ISO格式
    输入格式: "2025-07-31 12:00:00"
    输出格式: "2025-07-31T12:00:00"
    """
    try:
        if not date_str:
            return date_str
        
        # 替换空格为T，形成ISO格式
        if ' ' in date_str:
            return date_str.replace(' ', 'T')
        return date_str
    except Exception as e:
        logger.warning(f"日期格式转换失败 {date_str}: {e}")
        return date_str

async def es_out_date_format(date_str: str) -> str:
    """
    将ES的ISO格式日期转换回原始格式
    输入格式: "2025-07-31T12:00:00"
    输出格式: "2025-07-31 12:00:00"
    """
    try:
        if not date_str:
            return date_str
        
        # 替换T为空格，恢复原始格式
        if 'T' in date_str:
            return date_str.replace('T', ' ')
        return date_str
    except Exception as e:
        logger.warning(f"日期格式还原失败 {date_str}: {e}")
        return date_str