import torch
from loguru import logger

def get_optimal_attn_implementation() -> str:
    """
    Public method: Automatically select the optimal attention implementation based on current environment.
    Fallback order: flash_attention_2 -> sdpa (PyTorch 2.0+ built-in) -> eager (original mode)
    
    Returns:
        str: Name of the optimal attention implementation ("flash_attention_2", "sdpa", or "eager")
    """
    if not torch.cuda.is_available():
        return "eager"

    try:
        # 1. Check for Flash Attention 2 availability
        # is_flash_attn_2_available internally checks both flash_attn package and hardware compatibility
        from transformers.utils import is_flash_attn_2_available # type: ignore
        if is_flash_attn_2_available():
            logger.debug("当前环境支持 Flash Attention 2.0")
            return "flash_attention_2"
    except Exception:
        pass

    # 2. Check for PyTorch SDPA (Scaled Dot Product Attention)
    # SDPA is generally supported in CUDA environments with PyTorch 2.0+
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        logger.debug("当前环境支持 SDPA（PyTorch 2.0+ 内置）")
        return "sdpa"

    logger.debug("当前环境不支持硬件加速，将使用标准 Eager 模式")
    return "eager"
