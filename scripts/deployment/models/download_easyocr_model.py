import os
import easyocr

# 设置环境变量使用国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def pre_download_easyocr_models():
    """使用国内镜像预下载模型"""
    languages = ['ch_sim', 'en']
    
    for lang in languages:
        print(f"正在从国内镜像下载 {lang} 的检测模型...")
        try:
            reader = easyocr.Reader([lang], gpu=False, download_enabled=True)
            print(f"✓ {lang} 模型下载完成!")
        except Exception as e:
            print(f"✗ {lang} 模型下载失败: {e}")
            # 可以尝试手动下载
            manual_download_guide(lang)

def manual_download_guide(lang):
    """提供手动下载指南"""
    print(f"\n手动下载 {lang} 模型指南:")
    print("1. 访问 https://hf-mirror.com/")
    print("2. 搜索 'JaidedAI/EasyOCR'")
    print("3. 下载对应语言的模型文件")
    print("4. 将文件放到 ~/.EasyOCR/model/ 目录下")

# 在应用启动时调用
if __name__ == "__main__":
    pre_download_easyocr_models()
