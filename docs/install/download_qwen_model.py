from modelscope import snapshot_download

# 下载两个模型
models_to_download = [
    'Qwen/Qwen3-Reranker-4B',
    'Qwen/Qwen3-Embedding-4B'
]

for model_name in models_to_download:
    model_dir = snapshot_download(model_name)
    print(f"{model_name} 保存路径: {model_dir}")