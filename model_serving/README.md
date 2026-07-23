# Model Serving

`model_serving` is the shared model-serving package. It owns model provider
adapters, pools, model configuration CRUD and category-specific inference
services. It does not own Agent/Skill or Knowledge Core business data.

`common/entities/ai_model.py` and `common/model_repository.py` are the sole
owners of the `AIModelEntity` catalog. Other services read a sanitized model
definition through `platform_clients.AIModelConfigClient`; they must not import
this entity or repository. In the current single-schema deployment the
service still uses the shared Oracle connection, while the HTTP boundary keeps
the later move to a service-owned database mechanical.

Deployable processes are under `apps/ai_models_*`:

- `ai_models_embedding`: text embeddings and similarity
- `ai_models_llm`: chat completion and tool calling
- `ai_models_vlm`: vision-language inference
- `ai_models_visual`: image embeddings

All processes expose `/internal/v1/models` management endpoints restricted to their
model category. `DELETE` archives a definition instead of physically deleting
it, so Collection bindings and audit history remain safe. API keys are accepted
for writes but never returned by the management DTO.
