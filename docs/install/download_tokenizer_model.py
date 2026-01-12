import os
from huggingface_hub import snapshot_download

# 1. 设置使用 Hugging Face 镜像站（国内下载更快）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 指定模型名称和本地路径
# BAAI/bge-small-en-v1.5 纯英文文档，性能最高，速度最快
# BAAI/bge-base-zh-v1.5 纯中文文档，中文压缩率和语义表达最精准
# BAAI/bge-m3 中英混排/复杂文档，兼容性最强，省去语种检测

# 英文模型
model_id = "BAAI/bge-small-en-v1.5"
local_dir = "./models/bge-small-en-v1.5"

# 执行下载
print(f"正在从镜像站下载模型 {model_id}...")
snapshot_download(
    repo_id=model_id, 
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 直接复制文件而非创建软链接
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.bin"]  # 过滤不必要的文件格式
)

print(f"模型已成功下载至: {local_dir}")

# 中文模型
model_id = "BAAI/bge-base-zh-v1.5"
local_dir = "./models/bge-base-zh-v1.5"

# 执行下载
print(f"正在从镜像站下载模型 {model_id}...")
snapshot_download(
    repo_id=model_id, 
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 直接复制文件而非创建软链接
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.bin"]  # 过滤不必要的文件格式
)

print(f"模型已成功下载至: {local_dir}")