import asyncio
import sys
import os
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dao.repositories.kbot_md_db_conf_repo import KbotMdDbConfRepository

async def test_get_by_kbid():
    """Test getting database configuration by KB ID."""
    repo = KbotMdDbConfRepository()
    kb_id = 24
    
    print(f"Fetching database configuration for kb_id: {kb_id}")
    db_conf = await repo.get_by_kbid(kb_id)
    
    if db_conf is None:
        print(f"No database configuration found for kb_id: {kb_id}")
    else:
        print(f"Found database configuration:")
        print(f"DB ID: {db_conf.db_id}")
        print(f"DB Type: {db_conf.db_type}")
        print(f"Connection String: {db_conf.db_conn_str}")
        print(f"Display Name: {db_conf.db_display_name}")

if __name__ == "__main__":
    asyncio.run(test_get_by_kbid())