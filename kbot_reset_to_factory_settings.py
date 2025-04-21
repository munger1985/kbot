
### init logging
from loguru import logger

from kb_api import list_kbs_from_db, delete_kb

logger.add("factory_reset.log", rotation="5 MB")



def main():
    kbs=list_kbs_from_db()
    for kb in kbs:
        delete_kb(knowledge_base_name=kb)

if __name__ == "__main__":
    main()
