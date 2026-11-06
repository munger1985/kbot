"""正式 App 证据 Composer Skill；主结果、模型交付和最终引用分别记录。"""

import inspect
from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.runtime import AppRetrievalArtifact, KnowledgeCompositionArtifact
from knowledge_algorithms.contracts.wire import decode, encode
from knowledge_algorithms.retrieval.selection import PayloadCost
from agent_runtime.runtime import SkillArtifact, SkillResult
from agent_runtime.adapters.knowledge_context import owner_app_id


class AppKnowledgeComposerSkill:
    def __init__(self, *, composer_factory, content_port_factory):
        self.composer_factory, self.content_port_factory = composer_factory, content_port_factory

    async def execute(self, context):
        candidates = [item for item in context.input_artifacts if item.artifact_type == "RETRIEVAL_EVIDENCE"]
        require(len(candidates) == 1, "Composer 必须消费唯一正式证据 Artifact")
        payload = candidates[0].payload
        require(type(payload) is dict and set(payload) == {"retrieval_ref", "evidence"}, "证据包装合同无效")
        wrapper = decode(AppRetrievalArtifact, payload)
        artifact = wrapper.evidence
        configuration = context.config_snapshot["app_knowledge"]
        require(artifact.run_ref == str(context.run_id) and artifact.turn_ref == str(configuration["turn_ref"])
                and artifact.evidence_set.owner_binding_ref == owner_app_id(context), "证据跨 App/Run/Turn")
        composer = self.composer_factory(context)
        if inspect.isawaitable(composer):
            composer = await composer
        port = self.content_port_factory(context, payload["retrieval_ref"], artifact.evidence_set)
        result = await composer.compose(run_ref=artifact.run_ref, turn_ref=artifact.turn_ref,
            model_snapshot_ref=configuration["model_snapshot_ref"], evidence_set=artifact.evidence_set,
            content_port=port, question=context.original_input, budget=decode(PayloadCost, configuration["payload_budget"]),
            search_only=configuration.get("search_only", False))
        value = KnowledgeCompositionArtifact(wrapper.retrieval_ref, artifact.evidence_set, result.delivery,
            result.answer, result.labels, result.excluded, result.selection)
        await composer.recorder.record(artifact.run_ref, artifact.turn_ref, "KNOWLEDGE_COMPOSITION", value)
        return SkillResult(artifact=SkillArtifact(artifact_type="KNOWLEDGE_COMPOSITION",
            schema_version="knowledge.composition.v1", payload=encode(value)))
