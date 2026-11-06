"""交付审计通过 Runtime UoW 写入；不在 Skill 内访问 ORM/Session。"""

from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.wire import encode


class RuntimeKnowledgeArtifactRecorder:
    def __init__(self, *, uow_factory, context):
        self.uow_factory, self.context = uow_factory, context

    async def record(self, run_ref, turn_ref, artifact_type, value):
        require(run_ref == str(self.context.run_id)
                and turn_ref == str(self.context.config_snapshot["app_knowledge"]["turn_ref"]),
                "证据交付审计跨 Run/Turn")
        async with self.uow_factory() as uow:
            identity = await uow.artifacts.record_knowledge(
                context=self.context, artifact_type=artifact_type, schema_version=value.schema_version,
                payload=encode(value), turn_ref=turn_ref)
            await uow.commit()
        return identity
