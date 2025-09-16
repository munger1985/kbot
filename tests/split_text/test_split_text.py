import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径，以便导入backend模块
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.chunk_text import chunk_text

def test_split_text_with_post_doc():
    """
    测试split_text方法，使用post_doc.txt作为输入，
    chunk_size设置为100，overlap设置为20
    """
    # 获取当前文件所在目录
    current_dir = Path(__file__).parent
    
    # 读取post_doc.txt文件内容
    with open(current_dir / "post_doc.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    # 调用split_text方法，设置参数
    chunks = chunk_text(text=text, chunk_size=500, overlap=50)
    
    # 输出结果
    print("\n测试结果：")
    print(f"原始文本长度: {len(text)} 字符")
    print(f"分块数量: {len(chunks)}")
    print("\n各分块内容:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- 块 {i+1} (长度: {len(chunk)}) ---")
        print(chunk)

if __name__ == "__main__":
    test_split_text_with_post_doc()