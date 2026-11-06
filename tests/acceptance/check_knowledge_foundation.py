"""知识基础能力的单命令离线回归；绝不运行实库、真实模型、部署或重启测试。"""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def main():
    sources = [str(path) for path in sorted(ROOT.glob("packages/*/src"))]
    sources += [str(path) for path in sorted(ROOT.glob("services/*/src"))]
    environment = dict(os.environ)
    # 强制使用当前仓库，避免同名 editable 安装串到另一套数据库平台。
    environment["PYTHONPATH"] = os.pathsep.join(sources)
    oracle = (ROOT / "database/oracle").is_dir()
    aiops_tests = sorted({str(path.relative_to(ROOT))
        for pattern in ("test_aiops_*knowledge*.py", "test_aiops_case*.py", "test_aiops_*case*.py",
                        "test_aiops_output*.py", "test_aiops_report_output.py", "test_aiops_report_source_revision.py")
        for path in (ROOT / "tests/unit/aiops_agent").glob(pattern)})
    commands = [
        ["-m", "unittest", "discover", "-s", "tests/unit/knowledge_algorithms"],
        ["-m", "unittest", "discover", "-s", "tests/unit/knowledge_ingestion"],
        ["-m", "unittest", "discover", "-s", "tests/unit/knowledge_core", "-p", "test_parse_computation.py"],
        ["-m", "unittest", "discover", "-s", "tests/unit/platform_clients"],
        ["-m", "unittest", "discover", "-s", "tests/unit/knowledge_foundation"],
        ["-m", "pytest", "-q", "tests/unit/knowledge_retrieval_app", "tests/unit/agent_runtime"],
        ["-m", "pytest", "-q", *aiops_tests, "--tb=short"],
        ["tests/acceptance/check_app_owned_knowledge_schema.py"],
        ["tests/acceptance/check_4_0_boundaries.py" if oracle else "tests/acceptance/check_architecture_boundaries.py"],
        ["tests/acceptance/check_oracle_schema.py" if oracle else "tests/acceptance/check_postgresql_schema.py"],
    ]
    if not oracle:
        commands.append(["-m", "unittest", "discover", "-s", "tests/unit/model_serving", "-p", "test_colqwen2_local.py"])
    failures = []
    for arguments in commands:
        print("离线检查：" + " ".join(arguments), flush=True)
        result = subprocess.run([sys.executable, *arguments], cwd=ROOT, env=environment, check=False)
        if result.returncode:
            failures.append(arguments)
    if failures:
        print("离线检查失败：" + repr(failures))
        return 1
    print("知识基础能力离线回归通过；不代表真实检索质量、实库验证或发布已通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
