import torch
from loguru import logger

def get_optimal_attn_implementation() -> str:
    """
    公共方法：根据当前环境自动选择最佳的注意力实现。
    降级顺序：flash_attention_2 -> sdpa (PyTorch 2.0+ 内置) -> eager (原始模式)
    """
    if not torch.cuda.is_available():
        return "eager"

    try:
        # 1. 检查 Flash Attention 2
        # is_flash_attn_2_available 内部会同时检查 flash_attn 包和硬件兼容性
        from transformers.utils import is_flash_attn_2_available # type: ignore
        if is_flash_attn_2_available():
            logger.debug("环境支持 Flash Attention 2.0")
            return "flash_attention_2"
    except Exception:
        pass

    # 2. 检查 PyTorch SDPA (Scaled Dot Product Attention)
    # 只要是 PyTorch 2.0+ 且是 CUDA 环境，通常都支持 sdpa
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        logger.debug("环境支持 SDPA (PyTorch 2.0+ Built-in)")
        return "sdpa"

    logger.debug("无硬件加速可用，使用标准 Eager 模式")
    return "eager"