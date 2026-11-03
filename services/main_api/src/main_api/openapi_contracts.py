"""Main API 对外契约快照工厂。"""

from fastapi import FastAPI

from main_api.api.integrations import router as integration_router
from main_api.api.ops import router as ops_router
from main_api.api.output_authority import router as output_authority_router


def create_aiops_public_contract_app() -> FastAPI:
    """创建 Main API 映射使用的 AIOps 公开契约快照。"""
    app = FastAPI(title="KBot AIOps Public Contract", version="1.0.0")
    app.include_router(ops_router)
    app.include_router(integration_router)
    return app


def create_main_internal_contract_app() -> FastAPI:
    """内部授权查询单独生成契约，不混入公开 Main API 快照。"""
    app = FastAPI(title="KBot Main API Internal Contract", version="1.0.0")
    app.include_router(output_authority_router)
    return app
