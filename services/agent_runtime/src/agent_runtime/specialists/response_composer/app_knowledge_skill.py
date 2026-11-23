"""正式 App 证据 Composer Skill；主结果、模型交付和最终引用分别记录。"""

import inspect
from dataclasses import replace
from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.runtime import AppRetrievalArtifact, KnowledgeCompositionArtifact
from knowledge_algorithms.contracts.wire import decode, encode
from knowledge_algorithms.retrieval.selection import PayloadCost
from agent_runtime.runtime import SkillArtifact, SkillResult
from agent_runtime.adapters.knowledge_context import owner_app_id


class AppKnowledgeComposerSkill:
    def __init__(self, *, composer_factory, content_port_factory, question_images_factory=None, question_ocr_reader=None):
        self.composer_factory, self.content_port_factory = composer_factory, content_port_factory
        self.question_images_factory = question_images_factory
        self.question_ocr_reader = question_ocr_reader

    async def execute(self, context, *, on_preview=None):
        candidates = [item for item in context.input_artifacts if item.artifact_type == "RETRIEVAL_EVIDENCE"]
        require(len(candidates) == 1, "Composer 必须消费唯一正式证据 Artifact")
        payload = candidates[0].payload
        wrapper = decode(AppRetrievalArtifact, payload)
        artifact = wrapper.evidence
        configuration = context.config_snapshot["app_knowledge"]
        require(artifact.run_ref == str(context.run_id) and artifact.turn_ref == str(configuration["turn_ref"])
                and artifact.evidence_set.owner_binding_ref == owner_app_id(context), "证据跨 App/Run/Turn")
        port = self.content_port_factory(context, payload["retrieval_ref"], artifact.evidence_set)
        search_only = configuration.get("search_only", False)
        model_input = {}
        if self.question_images_factory is not None and not search_only and owner_app_id(context) == "knowledge_retrieval":
            images = await self.question_images_factory(context)
            if images:
                ocr_model = (context.config_snapshot["agent"].get("models") or {}).get("query_ocr")
                if ocr_model is not None and self.question_ocr_reader is not None:
                    images = await self.question_ocr_reader(context, images)
                    model_input["ocr_model_snapshot_ref"] = ocr_model["config_fingerprint"]
                    model_input["transcription_requested"] = bool(wrapper.transcription_requested)
                model_input["question_images"] = images
        budget = decode(PayloadCost, configuration["payload_budget"])
        metadata = context.config_snapshot.get("client_metadata") or {}
        text_only = owner_app_id(context) == "knowledge_retrieval" and not metadata.get("query_images")
        snapshot_ref = configuration["model_snapshot_ref"]
        if text_only:
            text_model = (context.config_snapshot["agent"].get("models") or {}).get("composer_llm")
            require(bool(text_model), "纯文字知识问答缺少冻结的文本回答模型")
            snapshot_ref = text_model["config_fingerprint"]
            budget = replace(budget, images=0)
        composer = self.composer_factory(context, text_only=text_only)
        if inspect.isawaitable(composer):
            composer = await composer
        preview = {"on_preview": on_preview} if on_preview is not None else {}
        result = await composer.compose(run_ref=artifact.run_ref, turn_ref=artifact.turn_ref,
            model_snapshot_ref=snapshot_ref, evidence_set=artifact.evidence_set,
            content_port=port, question=context.original_input, budget=budget,
            search_only=search_only, **model_input, **preview)
        value = KnowledgeCompositionArtifact(wrapper.retrieval_ref, artifact.evidence_set, result.delivery,
            result.answer, result.labels, result.excluded, result.selection)
        # 组合结果交给 Runtime 的任务结果流程；交付与引用审计已由 Composer 单独记录。
        return SkillResult(artifact=SkillArtifact(artifact_type="KNOWLEDGE_COMPOSITION",
            schema_version="knowledge.composition.v1", payload=encode(value)))
