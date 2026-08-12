"""不初始化数据库和凭据的 Data Query OpenAPI 契约应用。"""

from fastapi_offline import FastAPIOffline

from data_query.api import managed_dataset_router, management_router, model_reference_router, runtime_router


def create_data_query_contract_app() -> FastAPIOffline:
    app = FastAPIOffline(
        title="KBot Data Query Internal API",
        version="4.0.0",
    )
    app.include_router(management_router)
    app.include_router(managed_dataset_router)
    app.include_router(runtime_router)
    app.include_router(model_reference_router)
    return app


__all__ = ["create_data_query_contract_app"]
