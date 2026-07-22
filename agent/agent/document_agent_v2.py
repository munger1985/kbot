"""DocumentAgentV2: task orchestration only, with no KC/DB implementation."""
from knowledge_core.application.task_dto import KnowledgeTask, KnowledgeTaskResult
from skills.knowledge_retrieval_v2 import KnowledgeRetrievalSkillV2


class DocumentAgentV2:
    def __init__(self, *, retrieval_skill: KnowledgeRetrievalSkillV2):
        self._retrieval_skill = retrieval_skill

    async def retrieve(self, task: KnowledgeTask) -> KnowledgeTaskResult:
        return await self._retrieval_skill.execute(task)
