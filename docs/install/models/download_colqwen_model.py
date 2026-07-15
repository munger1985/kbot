"""下载 ColQwen2 视觉嵌入模型及 base model（Phase 2 多模态检索）。

ColQwen2 = base model + LoRA adapter，两者都必须下载。
base:   ~1.5 GB
adapter: ~4.5 GB
总计约 6 GB。

需先安装: pip install colpali-engine huggingface_hub torch
"""

import os

BASE_MODEL_ID = "vidore/colqwen2-base"
ADAPTER_ID = "vidore/colqwen2-v1.0"

# 默认保存路径（与其他模型一致）
DEFAULT_BASE_DIR = os.path.expanduser("/home/chris/models/colqwen2-base")
DEFAULT_ADAPTER_DIR = os.path.expanduser("/home/chris/models/colqwen2-v1.0")


def download_hf(repo_id: str, save_dir: str, label: str):
    """通过 huggingface_hub 下载"""
    from huggingface_hub import snapshot_download
    print(f"[{label}] {repo_id} → {save_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=save_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[{label}] ✅ 完成")


def verify():
    for label, d in [("Base", DEFAULT_BASE_DIR), ("Adapter", DEFAULT_ADAPTER_DIR)]:
        ok = os.path.isdir(d) and any(
            f.endswith(".safetensors") for f in os.listdir(d)
        )
        print(f"  {'✅' if ok else '❌'} {label}: {d}")


if __name__ == "__main__":
    download_hf(BASE_MODEL_ID, DEFAULT_BASE_DIR, "Base")
    download_hf(ADAPTER_ID, DEFAULT_ADAPTER_DIR, "Adapter")
    verify()
    print()
    print("下一步:")
    print(f"  1. ai_model 表 model_path = \"{DEFAULT_ADAPTER_DIR}\"")
    print(f"  2. 重启服务")
