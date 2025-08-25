import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv


# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Use absolute imports from project root
from utils.common_methods import lob_to_string
from services.chat.agent_chat import Agent

async def main():
    """End-to-end test with real embedding service and database"""
    # Load environment variables
    load_dotenv()
    
    # 1. Prepare test text
    test_text = "文艺复兴是什么时候，郑和下西洋又是什么时候？"
    
    agent = Agent(1,9)
    r = await agent.chat(test_text)
    if r is None:
        print("No results found.")
        return
    for res in r: # type: ignore
        print("\n===========================================================\n")
        print(f"file_id: {res.file_id}")
        print(f"content: {res.content}")
        print(f"page_num: {res.page_num}")
        print(f"similarity: {res.similarity}")
        print(f"weight: {res.weight}")
        print(f"rerank_score: {res.reranker_score}")
        print("\n===========================================================\n")




if __name__ == '__main__':
    print("Starting embedding similarity test...")
    asyncio.run(main())