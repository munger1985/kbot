import asyncio
import os
import sys
import aiohttp
from pathlib import Path
from dotenv import load_dotenv

# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(project_root))

# Use absolute imports from project root
from backend.dao.repositories.kbot_biz_txt_embedding import KbotBizTxtEmbeddingRepository

async def main():
    """End-to-end test with real embedding service and database"""
    # Load environment variables
    load_dotenv()
    
    # 1. Prepare test text
    test_text = "文艺复兴"
    
    # 2. Get embedding from service
    embed_host = os.getenv("KBOT_EMBED_HOST", "localhost")
    embed_port = os.getenv("KBOT_EMBED_PORT", "8001")
    model_id = 21
    
    embed_url = f"http://{embed_host}:{embed_port}/embed"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model_id": int(model_id),
        "texts": [test_text],
        "batch_size": 1
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(embed_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    print(f"Embedding service error: HTTP {response.status}, {text}")
                    return
                
                data = await response.json()
                embedding = data["embeddings"][0]
                print("Successfully got embedding vector")
    except Exception as e:
        print(f"Embedding service unavailable: {str(e)}")
        return

    # 3. Perform similarity search
    repo = KbotBizTxtEmbeddingRepository()
    try:
        results = await repo.get_similar_embeddings(
            kb_id=1,  # Use a known KB ID that has data
            embedding=embedding,
            similarity_threshold=0.5,
            top_k=5
        )
        
        print(f"\nFound {len(results)} similar embeddings:")
        for i, item in enumerate(results, 1):
            print(f"{i}. ID: {item.id}, Text: {item.text[:50]}...")

    except Exception as e:
        print(f"Similarity search failed: {str(e)}")

if __name__ == '__main__':
    print("Starting embedding similarity test...")
    asyncio.run(main())