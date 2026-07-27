from modelscope import snapshot_download

# KBot 当前只使用文本 Embedding；重排由 LLM 完成。
models_to_download = [
    'Qwen/Qwen3-Embedding-4B'
]

for model_name in models_to_download:
    model_dir = snapshot_download(model_name)
    print(f"{model_name} 保存路径: {model_dir}")
