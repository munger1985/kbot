"""正式 Runtime 的 App 知识装配；每次调用重签短期委托，冻结版本不跨 Run 复用。"""

import asyncio
import base64
from hashlib import sha256
from pathlib import Path
import time

from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.identity import AuthorizedScope, fingerprint
from knowledge_algorithms.contracts.runtime import DelegationClaims
from knowledge_algorithms.contracts.wire import decode, encode
from knowledge_algorithms.retrieval.selection import PayloadCost
from platform_clients.knowledge_computation import KnowledgeComputationClient
from platform_clients.knowledge_retrieval import AppKnowledgeClient
from platform_core.contracts import AuthContext
from platform_core.security.knowledge_delegation import create_knowledge_delegation_codec
from agent_runtime.adapters.app_knowledge import AppEvidenceContentPort
from agent_runtime.adapters.knowledge_model import KnowledgeEvidenceModelGateway
from agent_runtime.adapters.knowledge_recorder import RuntimeKnowledgeArtifactRecorder
from agent_runtime.application.knowledge_delivery import KnowledgeEvidenceComposer
from .knowledge import KnowledgeExecutionBinding, freeze_knowledge_context


class ByteUpperBoundTokenizer:
    """无真实 tokenizer 时采用明确的 UTF-8 字节上界，不低估上下文成本。"""

    def count(self, text):
        return len(text.encode("utf-8"))


class RuntimeKnowledgeIntegration:
    def __init__(self, *, settings, caller_service, uow_factory, prompt_resolver, session=None):
        self.settings, self.caller = settings, caller_service
        self.uow_factory, self.prompts = uow_factory, prompt_resolver
        self.clients = {owner: AppKnowledgeClient(owner=owner, base_url=config.base_url,
            audience=config.audience, caller_service=caller_service,
            timeout_seconds=config.timeout_seconds, session=session,
            delegation_codec=create_knowledge_delegation_codec())
            for owner, config in {"knowledge_retrieval": settings.knowledge_retrieval, "aiops": settings.aiops}.items()}

    def auth_context(self, context, scope=None):
        snapshot = context.policy_snapshot.get("auth_context")
        require(type(snapshot) is dict, "知识执行缺少服务端身份快照")
        original = AuthContext.model_validate(snapshot)
        return original.model_copy(update={"calling_service": self.caller,
            "app_id": context.config_snapshot["agent"]["owner_app_id"],
            "scopes": ("knowledge.scope.create", "knowledge.retrieve", "knowledge.content.read", "model.inference")})

    async def prepare(self, *, command, run_id, models):
        spec = command.execution_spec
        binding = spec.knowledge_binding
        if binding is None:
            return None
        now = int(time.time())
        spec_hash = fingerprint(spec.model_dump(mode="json"))
        owner = spec.owner_app_id
        client = self.clients[owner]
        require(type(command.policy_snapshot.get("auth_context")) is dict, "Run 缺少可信用户上下文")
        original = AuthContext.model_validate(command.policy_snapshot["auth_context"])
        require(str(original.domain_id) == str(command.domain_id)
                and (original.asserted_user_id or original.client_id) == str(command.actor_id),
                "Run 与内部用户身份不一致")
        claims = DelegationClaims(owner, client.audience, str(command.actor_id),
            str(getattr(command, "tenant_id", None)) if getattr(command, "tenant_id", None) else None,
            str(command.domain_id), str(spec.consumer_agent_version_id), str(run_id), spec_hash,
            "bootstrap:" + spec_hash, ("knowledge.scope.create", "knowledge.retrieve", "knowledge.content.read"), now + 900)
        context = original.model_copy(update={"calling_service": self.caller, "scopes": claims.operations})
        scope = decode(AuthorizedScope, await client.for_delegation(encode(claims)).create_scope(
            ("library:" + str(binding.library_id)) if binding.library_id else ("ops-query:" + str(binding.query_ticket)), auth_context=context))
        claims = DelegationClaims(**{**{field: getattr(claims, field) for field in claims.__dataclass_fields__},
                                     "scope_hash": scope.scope_hash})
        model = models.get("query_vlm") or models.get("composer_llm")
        require(type(model) is dict and type(model.get("max_tokens")) is int and model["max_tokens"] > 0,
                "知识回答模型必须提供 Model Serving 技术输出上限")
        return freeze_knowledge_context(trusted_spec=KnowledgeExecutionBinding(str(run_id), spec_hash, owner),
            scope=scope, delegation=claims, profile_ref=binding.profile_ref,
            turn_ref=str((command.policy_snapshot.get("aiops_parent") or {}).get("turn_id") or command.turn_id or run_id), model_snapshot_ref=model["config_fingerprint"],
            payload_budget=PayloadCost(self.settings.knowledge.payload_tokens,
                self.settings.knowledge.payload_images if models.get("query_vlm") else 0,
                self.settings.knowledge.payload_bytes),
            retain_until=now + self.settings.knowledge.evidence_retention_seconds, search_only=bool(spec.runtime_policy.get("search_only", False)))

    async def query(self, context):
        configuration = context.config_snapshot["app_knowledge"]
        scope = decode(AuthorizedScope, configuration["scope"])
        client = self.clients[scope.owner_binding_ref].for_delegation(configuration["delegation"])
        auth_context = self.auth_context(context, scope)
        metadata = context.config_snapshot.get("client_metadata") or {}
        images = metadata.get("query_images") or []
        require(type(images) is list and len(images) <= 4, "查询图片数超过预算")
        handles, total = [], 0
        conversation = context.config_snapshot.get("conversation") or {}
        for image in images:
            require(conversation.get("conversation_id"), "查询图片必须属于正式会话")
            root = (Path(self.settings.attachments.local_storage_path).resolve() / "conversation-query-images"
                    / str(conversation["conversation_id"])).resolve()
            target = Path(image["storage_uri"]).resolve()
            require(target.parent == root and target.suffix in {".png", ".jpg", ".webp"}, "查询图片不在本会话受控目录")
            raw = await asyncio.to_thread(self._read_image, target, min(8388608, 8388608 - total))
            total += len(raw)
            require(len(raw) == image["byte_size"] and sha256(raw).hexdigest() == image["content_sha256"],
                    "查询图片不可变摘要不匹配")
            result = await client.register_query_image({"scope": encode(scope),
                "image_base64": base64.b64encode(raw).decode("ascii"), "mime_type": image["mime_type"]},
                auth_context=auth_context)
            handles.append(result["image_handle"])
        text = context.original_input
        for artifact in context.input_artifacts:
            if artifact.artifact_type == "CONTEXT_REWRITE":
                rewritten = (artifact.payload or {}).get("standalone_query")
                if rewritten:
                    text = rewritten
            elif artifact.artifact_type == "DOCUMENT_SCOPE":
                text = artifact.payload["query"]
        from agent_runtime.application.knowledge_query import plan_knowledge_query
        prompt = await self.prompts.resolve("agent_runtime.knowledge_query_plan")
        config = self.settings.llm
        client = KnowledgeComputationClient(base_url=config.base_url, audience=config.audience,
            caller_service=self.caller, timeout_seconds=config.timeout_seconds,
            service_scopes=("model.inference",))
        return await plan_knowledge_query(text=text, image_handles=handles,
            model=context.config_snapshot["agent"]["models"]["context_llm"], prompt=prompt.content,
            client=client, auth_context=auth_context)

    @staticmethod
    def _read_image(target, maximum):
        require(maximum > 0, "查询图片总大小超过 8 MiB")
        with target.open("rb") as stream:
            raw = stream.read(maximum + 1)
        require(0 < len(raw) <= maximum, "查询图片大小超过预算")
        return raw

    def content_port(self, context, retrieval_ref, evidence):
        configuration = context.config_snapshot["app_knowledge"]
        client = self.clients[evidence.owner_binding_ref].for_delegation(configuration["delegation"])
        return AppEvidenceContentPort(client=client, auth_context=self.auth_context(context),
            retrieval_ref=retrieval_ref, evidence_set=evidence)

    async def composer(self, context):
        models = context.config_snapshot["agent"]["models"]
        supports_images = bool(models.get("query_vlm"))
        model = models["query_vlm"] if supports_images else models["composer_llm"]
        config = self.settings.vlm if supports_images else self.settings.llm
        prompt = await self.prompts.resolve("agent_runtime.app_knowledge_compose")
        gateway = KnowledgeEvidenceModelGateway(client=KnowledgeComputationClient(base_url=config.base_url,
            audience=config.audience, caller_service=self.caller, timeout_seconds=config.timeout_seconds,
            service_scopes=("model.inference",)), auth_context=self.auth_context(context),
            model_snapshot_ref=model["config_fingerprint"],
            model_payload={"model_id": model["model_id"], "max_tokens": model["max_tokens"]},
            system_prompt=prompt.content + self._agent_context(context), supports_images=supports_images)
        return KnowledgeEvidenceComposer(tokenizer=ByteUpperBoundTokenizer(),
            image_cost=lambda raw, mime: self.settings.knowledge.image_token_cost, model_gateway=gateway,
            artifact_recorder=RuntimeKnowledgeArtifactRecorder(uow_factory=self.uow_factory, context=context),
            max_payload_bytes=self.settings.knowledge.payload_bytes, max_materialized_items=128, max_cover_states=20000,
            image_mime_types=frozenset({"image/png", "image/jpeg", "image/webp"}) if supports_images else frozenset())

    @staticmethod
    def _agent_context(context):
        import json
        agent = context.config_snapshot["agent"]
        instruction = agent.get("instruction") or ""
        language = context.config_snapshot.get("language") or context.config_snapshot.get("language_policy") or {}
        business = (agent.get("config") or {}).get("business_knowledge")
        return ("\n当前执行语言约束：\n" + json.dumps(language, ensure_ascii=False)
            + "\n当前 Agent 指令：\n" + instruction
            + ("\n业务知识使用边界（历史案例不等于当前诊断证据）：\n"
               + json.dumps(business, ensure_ascii=False) if business else ""))

    async def text_inputs(self, context, artifact):
        from knowledge_algorithms.contracts.runtime import AppRetrievalArtifact
        wrapper = decode(AppRetrievalArtifact, artifact.payload)
        require(wrapper.evidence.run_ref == str(context.run_id), "Hybrid 输入跨 Run")
        evidence = wrapper.evidence.evidence_set
        port = self.content_port(context, wrapper.retrieval_ref, evidence)
        rows, consumed = [], 0
        for item in evidence.evidence:
            if item.mime_type != "text/plain" or consumed + item.byte_count > 262144:
                continue
            if item.key not in await port.revalidate((item.key,)):
                continue
            raw = b"".join([part async for part in port.content(item.key, max_bytes=item.byte_count)])
            require(sha256(raw).hexdigest() == item.content_hash and len(raw) == item.byte_count,
                    "Hybrid 正文内容摘要失效")
            consumed += len(raw)
            rows.append({"key": item.key.stable_key, "text": raw.decode("utf-8")})
        require(rows, "Hybrid 没有可交付的合法正文，不能凭句柄提取条件")
        return {"source_kind": "APP_KNOWLEDGE_TEXT", "items": rows}
    async def open_history(self, *, snapshot, auth_context, retrieval_ref, key):
        configuration = snapshot["app_knowledge"]
        owner = snapshot["agent"]["owner_app_id"]
        claims = {**configuration["delegation"], "expires_at": int(time.time()) + 300}
        context = auth_context.model_copy(update={"calling_service": self.caller,
            "scopes": ("knowledge.content.read",)})
        return await self.clients[owner].for_delegation(claims).open_historical_content(
            retrieval_ref, key, auth_context=context)
    async def release_snapshot(self, snapshot):
        from types import SimpleNamespace
        configuration = snapshot.get("app_knowledge")
        if not configuration:
            return
        claims = {**configuration["delegation"], "expires_at": int(time.time()) + 300}
        context = SimpleNamespace(policy_snapshot=snapshot["cleanup_policy"], config_snapshot=snapshot)
        await self.clients[claims["owner_binding_ref"]].for_delegation(claims).release_run(
            configuration["scope"], auth_context=self.auth_context(context))
