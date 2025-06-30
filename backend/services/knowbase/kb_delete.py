import os
from urllib.parse import unquote
from typing import List, Optional
from backend.api.schemas.kb_response import BaseResponse
from backend.core.log.logger import logger
from backend.core.config import settings
from backend.api.schemas.kb_response import DeleteResponse
from backend.dao.repositories.kbot_kb_repo import check_exists_by_name
from backend.dao.repositories.kbot_kb_files_repo import KBotKBFilesRepository

class KBDelete:
    def __init__(self):
        self.kb_files_repo = KBotKBFilesRepository()
    def delete_files(self,
                     knowledge_base_name: str,
                     batch_id: Optional[int] = None,
                     file_id: Optional[List[int]] = None,
                     ) -> DeleteResponse:
        failed_files = []
        success_cnt = 0
        knowledge_base_name = unquote(knowledge_base_name)
        kbid = check_exists_by_name(knowledge_base_name)
        if kbid is None:
            return DeleteResponse(  # return DeleteResponse
                code=404, 
                msg=f"Knowledge base {knowledge_base_name} does not exist",
                data={"deleted_count": success_cnt, "failed_files": failed_files}
            )
        
        # scenario 1: delete single file
        if file_id is not None:
            files = self.kb_files_repo.get_by_id(file_id)
        # scenario 2: delete files in a batch
        elif batch_id is not None:
            files = get_by_batch_id(batch_id)
        # scenario 3: delete all files in a knowledge base
        else: 
            files = get_by_kb_id(kbid)

        # deleting files
        for file in files:
            logger.info("Deleting file: {}", file.filepath)
            # check if the file exists
            if os.path.exists(file.filepath):
                # python delete a file from os
                os.remove(file.filepath)
                success_cnt += 1
            else:
                failed_files[file.filename] = 'File does not exist'

        return DeleteResponse(code=200, 
                            msg=f"File deletion completed.", 
                            data={"deleted_count": success_cnt, "failed_files": failed_files})

    def delete_records(knowledge_base_name: str,
                    batch_id: Optional[int] = None,
                    file_id: Optional[List[int]] = None
                    ) -> DeleteResponse:
        failed_records = []
        success_cnt = 0
        knowledge_base_name = unquote(knowledge_base_name)
        kbid = check_exists_by_name(knowledge_base_name)
        if kbid is None:
            return DeleteResponse(  # return DeleteResponse
                code=404, 
                msg=f"Knowledge base {knowledge_base_name} does not exist",
                data={"deleted_count": success_cnt, "failed_records": failed_records}
            )
        
        # scenario 1: delete single file
        if file_id is not None:
            files = get_by_id(file_id)
        # scenario 2: delete files in a batch
        elif batch_id is not None:
            files = get_by_batch_id(batch_id)
        # scenario 3: delete all files in a knowledge base
        else: 
            files = get_by_kb_id(kbid)

        # deleting records


        return DeleteResponse(code=200, msg=f"File deletion ended", data={"failed_files": failed_files})