"""正式 App API/Worker 共用的知识基础能力装配，无旧 KC fallback。"""

from pathlib import Path
from knowledge_algorithms.contracts.cursor import CursorCodec
from knowledge_algorithms.projections.builder import ProjectionBuilder
from platform_clients.knowledge_computation import KnowledgeComputationClient
from platform_clients.knowledge_models import NativeRepresentationClient
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.identity import uuid7
from platform_core.security.knowledge_delegation import create_knowledge_delegation_codec, knowledge_signing_key
from .knowledge_ingestion import install_app_knowledge
from .knowledge_retrieval import install_knowledge_retrieval
from .knowledge_profiles import ConservativeTokenizer
from ..application.operations_documents import OperationsDocuments
from ..application.operations_knowledge_authorization import OperationsKnowledgeAuthorization
from ..application.knowledge_pipeline import KnowledgePipelineWorker
from ..adapters.knowledge_models import KnowledgeModelGateway
from ..domain.knowledge_ingestion import BINDING


def scope_context(scope, *, caller, operation):
    return AuthContext(principal_kind=PrincipalKind.SERVICE, client_id=caller, calling_service=caller,
        app_id=BINDING.binding_ref, domain_id=str(scope.domain_id),
        tenant_id=str(scope.tenant_id) if scope.tenant_id else None, asserted_user_id=scope.subject_ref,
        request_id=str(uuid7()), trace_id=str(uuid7()), scopes=(operation,))


def install_business_knowledge(app, *, settings, uow_factory, caller_service=None, register_routes=True):
    caller, configuration = caller_service or settings.api.service_name, settings.knowledge
    parse, projection, profiles = configuration.profiles()
    tokenizer = ConservativeTokenizer()
    compute = KnowledgeComputationClient(base_url=settings.clients.knowledge_core.base_url, caller_service=caller,
        audience=settings.clients.knowledge_core.audience, timeout_seconds=settings.clients.knowledge_core.timeout_seconds)
    ingestion = install_app_knowledge(app, uow_factory=uow_factory, store_root=Path(configuration.storage_root),
        compute_client=compute, register_routes=register_routes,
        context_factory=lambda scope: scope_context(scope, caller=caller,
            operation="knowledge.parse.compute"))
    dense = NativeRepresentationClient(base_url=settings.clients.model_embedding.base_url, caller_service=caller,
        audience=settings.clients.model_embedding.audience, service_scopes=("model.inference",))
    visual = NativeRepresentationClient(base_url=settings.clients.model_visual.base_url, caller_service=caller,
        audience=settings.clients.model_visual.audience, service_scopes=("model.inference",))
    specs = projection.representations
    bindings = {spec.space_id: {"model_ref": spec.model_ref,
        "served_model_name": configuration.served_model_names[spec.model_ref]} for spec in specs}
    def gateway(access, *, image_reader):
        return KnowledgeModelGateway(visual_client=visual, dense_client=dense,
            auth_context=scope_context(access, caller=caller, operation="model.inference"),
            image_reader=image_reader, models=bindings, max_elements=projection.max_representation_elements)
    prefix = caller.split("-")[0]
    retrieval = install_knowledge_retrieval(app, audience=settings.api.service_name, register_routes=register_routes,
        delegation_codec=create_knowledge_delegation_codec(),
        allowed_callers=(prefix + "-main-api", prefix + "-agent-runtime-api", prefix + "-agent-runtime-worker"),
        uow_factory=uow_factory, object_store=ingestion.store, authorization=OperationsKnowledgeAuthorization(service=None),
        profiles=profiles, tokenizer=tokenizer, image_cost=lambda image: configuration.image_token_cost,
        cursor_codec=CursorCodec(knowledge_signing_key("cursor"), max_bytes=8192),
        model_gateway_factory=gateway)
    app.state.knowledge_projection_profile = projection
    app.state.knowledge_tokenizer = tokenizer
    app.state.knowledge_gateway_factory = gateway
    app.openapi_schema = None
    documents = OperationsDocuments(uow_factory=uow_factory, ingestion=ingestion,
        projection=app.state.knowledge_projection_service, retrieval=retrieval,
        parse_profile=parse, pipeline=business_pipeline(app, settings=settings, worker_id=caller + "-knowledge"))
    app.state.operations_documents = documents
    return documents


def business_pipeline(app, *, settings, worker_id):
    configuration = settings.knowledge
    async def allowed(content):
        return content
    def builder(access, document):
        async def image_reader(handle):
            raw = await app.state.knowledge_ingestion_service.store.read(handle, max_bytes=configuration.max_input_bytes)
            from io import BytesIO
            from PIL import Image
            with Image.open(BytesIO(raw)) as image:
                mime = Image.MIME[image.format]
            return raw, mime
        gateway = app.state.knowledge_gateway_factory(access, image_reader=image_reader)
        return ProjectionBuilder(BINDING, app.state.knowledge_tokenizer, gateway)
    return KnowledgePipelineWorker(ingestion=app.state.knowledge_ingestion_service,
        projection=app.state.knowledge_projection_service, builder_factory=builder,
        profile=app.state.knowledge_projection_profile, allowed_policy=allowed,
        policy_revision="aiops-reviewed-source/v1", worker_id=worker_id,
        lease_seconds=configuration.lease_seconds, poll_interval_seconds=min(configuration.poll_seconds, 1.))
