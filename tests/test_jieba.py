import asyncio
import sys
from pathlib import Path

# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Use absolute imports from project root
from services.search.fulltext_preprocessor import preprocess_for_fulltext

async def test_preprocess_for_fulltext():
    text = "这是一个测试句子"
    processed = await preprocess_for_fulltext(text)
    print(f"原始文本: {text}")
    print(f"处理后文本: {processed}")

if __name__ == "__main__":
    asyncio.run(test_preprocess_for_fulltext())
