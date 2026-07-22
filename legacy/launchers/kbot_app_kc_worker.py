"""Compatibility launcher; deployable implementation lives in apps.knowledge_core_projection."""
from apps.knowledge_core_projection.main import main

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
