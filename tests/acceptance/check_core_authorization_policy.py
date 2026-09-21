"""验收 KBot 全部业务 App 只使用一套核心授权规则。"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages/platform_core/src"))
sys.path.insert(0, str(ROOT / "services/main_api/src"))

from fastapi.routing import APIRoute
from main_api.app import create_main_api_app
from main_api.application import authorize_business_app_route
from platform_core.authorization import (
    APP_AUTHORIZATION_POLICIES,
    AgentAccessMode,
    resolve_business_app_id,
)


FOUNDATION_SQL = ROOT / "database/oracle/main_api/001_access_control.sql"
ORACLE_SQL_ROOT = ROOT / "database/oracle"
SCANNED_SOURCE_ROOTS = (
    ROOT / "services",
    ROOT / "packages",
    ROOT / "ui",
    ROOT / "tools/dev_console",
)
FORBIDDEN_HUMAN_AGENT_GRANT_TOKENS = (
    "KBOT_OPS_AGENT_GRANT",
    "KBOT_KM_AGENT_GRANT",
    "/agent-grants",
    "agents:authorize",
    "AIOpsAgentGrant",
    "KmAgentGrant",
    "authorize_private_agent",
    "private_agent_grant",
)


def _source_text() -> str:
    parts: list[str] = []
    for root in SCANNED_SOURCE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if "build" in path.parts or "__pycache__" in path.parts:
                continue
            if path.suffix not in {".py", ".js", ".html"}:
                continue
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _decorated_route_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    router_names = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "APIRouter"
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Name)
            and decorator.func.value.id in router_names
            for decorator in node.decorator_list
        ):
            names.add(node.name)
    return names


def _declared_route_names(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    permission_prefixes = (
        "aiops:",
        "knowledge_retrieval:",
    )
    return {
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and not node.value.startswith(permission_prefixes)
        and re.fullmatch(r"[a-z][a-z0-9_]*", node.value)
    }


def _unprotected_business_routes(path: Path) -> set[str]:
    """以模块内调用图确认公开业务路由最终进入统一授权入口。"""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    calls = {
        name: {
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
        }
        for name, node in functions.items()
    }
    protected = {
        name
        for name, names in calls.items()
        if names
        & {
            "authorize_app_request",
            "authorize_app_request_any",
            "_require_platform",
        }
    }
    protected.update(
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "main_api.api.runs"
        for alias in node.names
        if alias.name == "_require_use"
    )
    changed = True
    while changed:
        expanded = {
            name for name, names in calls.items() if names & protected
        }
        changed = not expanded.issubset(protected)
        protected.update(expanded)

    protected_routers: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Name) or node.value.func.id != "APIRouter":
            continue
        dependencies = next(
            (
                keyword.value
                for keyword in node.value.keywords
                if keyword.arg == "dependencies"
            ),
            None,
        )
        dependency_names = {
            child.args[0].id
            for child in ast.walk(dependencies)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "Depends"
            and child.args
            and isinstance(child.args[0], ast.Name)
        } if dependencies is not None else set()
        if dependency_names & protected:
            protected_routers.update(
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            )

    unprotected: set[str] = set()
    for name, node in functions.items():
        route_routers = {
            decorator.func.value.id
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr in {"get", "post", "put", "patch", "delete"}
            and isinstance(decorator.func.value, ast.Name)
        }
        if not route_routers or name in {"login", "change_password"}:
            continue
        if name not in protected and not route_routers & protected_routers:
            unprotected.add(name)
    return unprotected


def _route_dependencies(route: APIRoute) -> set[object]:
    """递归取得 FastAPI 已构建路由的全部依赖调用。"""

    calls: set[object] = set()
    pending = list(route.dependant.dependencies)
    while pending:
        dependency = pending.pop()
        if dependency.call is not None:
            calls.add(dependency.call)
        pending.extend(dependency.dependencies)
    return calls


def main() -> int:
    sql = FOUNDATION_SQL.read_text(encoding="utf-8")
    source = _source_text()
    registered = set(APP_AUTHORIZATION_POLICIES)
    catalog_apps = {
        app_id
        for app_id, member_assignable in re.findall(
            r"INTO KBOT_PLATFORM_APP .*?VALUES \('([^']+)', .*?, '([YN])'\)",
            sql,
        )
        if member_assignable == "Y"
    }
    assert catalog_apps == registered, (
        f"业务 App 与核心授权注册表不一致：catalog={sorted(catalog_apps)} "
        f"registry={sorted(registered)}"
    )

    permission_rows = re.findall(
        r"INTO KBOT_PERMISSION VALUES \('([^']+)', '([^']+)',",
        sql,
    )
    catalog_permissions = {permission for permission, _ in permission_rows}
    for permission, app_id in permission_rows:
        assert permission.startswith(f"{app_id}:"), (
            f"权限 {permission} 不属于声明的 App {app_id}"
        )

    schema_runner_tree = ast.parse(
        (ROOT / "scripts/db/apply_oracle_schema.py").read_text(
            encoding="utf-8"
        )
    )
    foundation_permissions = ast.literal_eval(
        next(
            node.value
            for node in schema_runner_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "PLATFORM_FOUNDATION_PERMISSIONS"
                for target in node.targets
            )
        )
    )
    assert foundation_permissions == catalog_permissions, (
        "规范权限目录与 Schema Runner 初始化目录不一致"
    )

    for app_id, policy in APP_AUTHORIZATION_POLICIES.items():
        assert policy.use_permission == f"{app_id}:use"
        assert resolve_business_app_id(app_id) == app_id
        assert resolve_business_app_id(policy.public_slug) == app_id
        assert policy.use_permission in catalog_permissions
        policy.assert_role_permissions((policy.use_permission,))
        try:
            policy.assert_role_permissions(())
        except ValueError:
            pass
        else:
            raise AssertionError(f"App {app_id} 允许角色绕过 use 权限")
        assert policy.human_agent_access == AgentAccessMode.DOMAIN_ACTIVE
        assert policy.machine_agent_access == AgentAccessMode.CLIENT_ALLOWLIST
        assert re.search(
            rf"INTO KBOT_APP_ROLE VALUES \('{re.escape(app_id)}', 'user',",
            sql,
        ), f"App {app_id} 缺少系统 user 角色"
        assert re.search(
            rf"SELECT '{re.escape(app_id)}', 'user', PERMISSION_CODE[\s\S]*?"
            rf"{re.escape(policy.use_permission)}",
            sql,
        ), f"App {app_id} 的 user 角色未绑定 use 权限"
        for scope, permission in policy.api_scope_permissions.items():
            assert permission in catalog_permissions, (
                f"机器 Scope {scope} 映射到不存在的权限 {permission}"
            )
        if policy.api_scope_permissions:
            assert policy.api_route_scopes, f"App {app_id} 未登记机器公开路由"

    oracle_sql = "\n".join(
        path.read_text(encoding="utf-8")
        for path in ORACLE_SQL_ROOT.rglob("*.sql")
    )
    for token in FORBIDDEN_HUMAN_AGENT_GRANT_TOKENS:
        assert token not in source, f"人类 Agent Grant 模型重新出现：{token}"
    for table_name in ("KBOT_OPS_AGENT_GRANT", "KBOT_KM_AGENT_GRANT"):
        assert re.search(
            rf"CREATE\s+TABLE\s+{table_name}\b",
            oracle_sql,
            flags=re.IGNORECASE,
        ) is None, f"人类 Agent Grant Schema 重新出现：{table_name}"

    api_client_sql = (
        ROOT / "database/oracle/main_api/002_app_api_clients.sql"
    ).read_text(encoding="utf-8")
    assert "KBOT_APP_API_CLIENT_AGENT" in api_client_sql
    assert "KBOT_PERMISSION" not in api_client_sql, (
        "API Client DDL 不得重复维护核心权限目录"
    )

    generic_consumers = {
        f"{app_id}:{suffix}"
        for app_id in registered
        for suffix in ("member_manage", "role_manage", "api_key_manage")
    }
    unconsumed = sorted(
        permission
        for permission in catalog_permissions
        if permission not in source and permission not in generic_consumers
    )
    assert not unconsumed, f"权限目录存在无消费者权限：{unconsumed}"

    main_api_api = ROOT / "services/main_api/src/main_api/api"
    direct_app_requires = []
    for path in main_api_api.glob("*.py"):
        if path.name in {"access_management.py", "domains.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if (
            "access_control_service.require(" in text
            or "access_control_service.snapshot(" in text
        ):
            direct_app_requires.append(path.name)
    assert not direct_app_requires, (
        "公开 App 路由绕过统一授权入口：" + ", ".join(direct_app_requires)
    )
    application = create_main_api_app(enable_access_log=False)
    unprotected_routes: dict[str, list[str]] = {}
    unknown_app_routes: list[str] = []
    checked_sources: dict[Path, tuple[set[str], set[str]]] = {}
    for route in application.routes:
        if not isinstance(route, APIRoute) or not route.path.startswith(
            "/api/v1/apps/"
        ):
            continue
        relative = route.path.removeprefix("/api/v1/apps/")
        public_slug = relative.split("/", 1)[0]
        if public_slug != "{app_id}":
            try:
                resolve_business_app_id(public_slug)
            except ValueError:
                unknown_app_routes.append(route.path)
        if authorize_business_app_route not in _route_dependencies(route):
            unprotected_routes.setdefault("runtime", []).append(route.path)
            continue
        endpoint_name = getattr(route.endpoint, "__name__", "")
        if endpoint_name in {"login", "change_password"} and "/auth/" in route.path:
            continue
        source_name = inspect.getsourcefile(route.endpoint)
        if source_name is None:
            unprotected_routes.setdefault("unknown", []).append(route.path)
            continue
        source_path = Path(source_name).resolve()
        try:
            source_path.relative_to(main_api_api.resolve())
        except ValueError:
            unprotected_routes.setdefault(source_path.name, []).append(route.path)
            continue
        decorated, unprotected = checked_sources.setdefault(
            source_path,
            (
                _decorated_route_names(source_path),
                _unprotected_business_routes(source_path),
            ),
        )
        if endpoint_name not in decorated or endpoint_name in unprotected:
            unprotected_routes.setdefault(source_path.name, []).append(route.path)
    assert not unknown_app_routes, (
        "公开业务路由使用了未注册 App：" + ", ".join(sorted(unknown_app_routes))
    )
    assert not unprotected_routes, (
        "公开业务路由未进入统一授权入口："
        + "; ".join(
            f"{path}={routes}"
            for path, routes in sorted(unprotected_routes.items())
        )
    )

    agent_route_files = {
        "knowledge_retrieval": main_api_api / "knowledge_retrieval_app.py",
        "km_asset": main_api_api / "km_asset_app.py",
        "aiops": main_api_api / "aiops_app.py",
    }
    for app_id, path in agent_route_files.items():
        text = path.read_text(encoding="utf-8")
        assert "filter_readable_agents(" in text and "can_read_agent(" in text, (
            f"App {app_id} 的 Agent 可见性未使用核心策略执行器"
        )
    endpoint_scope_checks = sorted(
        path.name
        for path in main_api_api.glob("*.py")
        if "require_app_api_scope(" in path.read_text(encoding="utf-8")
    )
    assert not endpoint_scope_checks, (
        "机器 Scope 只能由核心注册表判定，端点不得重复校验："
        + ", ".join(endpoint_scope_checks)
    )
    unbounded_patterns = sorted(
        f"{app_id}:{route.relative_path_pattern}"
        for app_id, policy in APP_AUTHORIZATION_POLICIES.items()
        for route in policy.api_route_scopes
        if ".*" in route.relative_path_pattern
    )
    assert not unbounded_patterns, (
        "机器公开路由不得使用无边界通配：" + ", ".join(unbounded_patterns)
    )
    access_repository = (
        ROOT / "services/main_api/src/main_api/repositories/access_control.py"
    ).read_text(encoding="utf-8")
    assert access_repository.count(
        'func.concat(AppDomainEntity.app_id, ":use")'
    ) >= 2, "登录 App 和 Domain 列表必须显式要求对应 use 权限"
    app_api_key_source = (
        ROOT / "services/main_api/src/main_api/application/app_api_key.py"
    ).read_text(encoding="utf-8")
    assert "APP_API_KEY_AGENT_INVALID" in app_api_key_source, (
        "API Client 创建必须校验当前 Domain 的 ACTIVE Agent"
    )

    ops_path = main_api_api / "ops.py"
    assert _decorated_route_names(ops_path) == _declared_route_names(
        ops_path, "_route_permissions"
    ), "AIOps 公开端点与显式权限登记不一致"

    knowledge_path = main_api_api / "knowledge.py"
    knowledge_tree = ast.parse(knowledge_path.read_text(encoding="utf-8"))
    knowledge_assignment = next(
        node
        for node in knowledge_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "KNOWLEDGE_ROUTE_PERMISSIONS"
            for target in node.targets
        )
    )
    assert _decorated_route_names(knowledge_path) == set(
        ast.literal_eval(knowledge_assignment.value)
    ), "Knowledge 公开端点与显式权限登记不一致"

    print("核心授权策略验收通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
