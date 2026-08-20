"""检查 KM 重新索引状态跟踪代码是否由当前环境正确加载。"""

from __future__ import annotations

import inspect

from km_asset_app.application.assets import KmAssetService
from km_asset_app.application.worker import KmAssetWorker
from knowledge_core.api.status_router import router as knowledge_status_router
from knowledge_core.application.status import KnowledgeCoreStatusService
from platform_clients import KnowledgeCoreClient


def main() -> int:
    """打印实际加载位置并校验重新索引跟踪链路。"""
    asset_source = inspect.getsource(KmAssetService.reindex_asset)
    worker_source = inspect.getsource(KmAssetWorker._kc_status_sync)
    status_source = inspect.getsource(
        KnowledgeCoreStatusService.get_discovery_reindex_operation
    )
    client_source = inspect.getsource(
        KnowledgeCoreClient.get_reindex_discovery_status
    )
    route_paths = {
        route.path
        for route in knowledge_status_router.routes
    }
    expected_route = (
        "/internal/v1/knowledge/domains/{domain_id}/bundles/{bundle_id}"
        "/revisions/{bundle_revision_id}/reindex-discovery/{generation}"
    )
    checks = {
        "KM_API_PERSISTS_TRACKING_JOB": (
            "kc-reindex-status:" in asset_source
            and '"tracking_status"' in asset_source
        ),
        "KM_WORKER_POLLS_REINDEX_OPERATION": (
            "get_reindex_discovery_status" in worker_source
        ),
        "KC_STATUS_AGGREGATES_REINDEX_JOBS": (
            "reindex_generation" in status_source
            and 'status = "SUCCEEDED"' in status_source
        ),
        "KC_CLIENT_HAS_STATUS_METHOD": (
            "get_reindex_discovery_status" in client_source
            and "reindex-discovery" in client_source
            and "generation" in client_source
        ),
        "KC_STATUS_ROUTE_REGISTERED": expected_route in route_paths,
    }
    print(
        "km_asset_service =",
        inspect.getsourcefile(KmAssetService),
    )
    print(
        "km_asset_worker =",
        inspect.getsourcefile(KmAssetWorker),
    )
    print(
        "knowledge_status =",
        inspect.getsourcefile(KnowledgeCoreStatusService),
    )
    print(
        "knowledge_client =",
        inspect.getsourcefile(KnowledgeCoreClient),
    )
    for name, passed in checks.items():
        print(f"{name} = {passed}")
    if all(checks.values()):
        print("KM 重新索引状态跟踪代码加载检查通过")
        return 0
    print("KM 重新索引状态跟踪代码加载检查失败：当前环境仍在加载旧代码")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
