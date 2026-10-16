"""生成从线上 AIOps Schema 42 追平 KBot 4.1 的单文件升级脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OPERATIONS_DIR = ROOT / "database" / "oracle" / "operations"
SOURCE_SCRIPTS = (
    OPERATIONS_DIR / "apply_aiops_schema_43.sql",
    OPERATIONS_DIR / "apply_aiops_schema_44.sql",
    OPERATIONS_DIR / "apply_aiops_schema_45.sql",
    OPERATIONS_DIR / "upgrade_kbot_41_retire_km_asset.sql",
)
OUTPUT_PATH = (
    ROOT
    / "database"
    / "oracle"
    / "generated"
    / "kbot_41"
    / "apply_kbot_41_from_aiops_schema_42.sql"
)

HEADER = """-- KBot 线上 Schema 追平 KBot 4.1 的单文件升级脚本。
-- 本文件由 tools/db/render_kbot_41_online_upgrade.py 生成，请勿手工修改。
--
-- 唯一起点：AIOPS / 42 / aiops-oracle-v32。
-- 最终目标：AIOPS / 45 / aiops-oracle-v45-shared-monitor-locator，
--           并从 KBot 4.1 主产品 Schema 退役 km_asset 私有对象和授权边界。
--
-- 使用 KBot Schema 所有者在 SQL Developer 中以 Run Script（F5）执行。
-- 执行前必须：
--   1. 停止 Main API、AIOps API、Worker、Scheduler、DB Executor、KM API、
--      KM Worker、Slack Worker 及其他 KBot 写入服务；
--   2. 使用 Data Pump 完成当前 Schema 的可恢复备份并验证 dump；
--   3. 确认本环境后续运行 KBot 4.1；继续运行 KM 4.0 的环境不得执行本脚本；
--   4. 确认所有既有 AIOps Target 都已设置责任组。
--
-- Oracle DDL 会自动提交。脚本失败时不要跳过报错继续执行；应依据最后完成的
-- Schema 版本和输出恢复备份或由 DBA 审核后续升级起点。
"""


def render_upgrade_sql() -> str:
    sections = [HEADER.rstrip()]
    for source in SOURCE_SCRIPTS:
        relative_name = source.relative_to(ROOT).as_posix()
        sections.extend(
            (
                f"-- ===== 开始升级阶段：{relative_name} =====",
                source.read_text(encoding="utf-8").rstrip(),
                f"-- ===== 完成升级阶段：{relative_name} =====",
            )
        )
    sections.append("PROMPT === KBot 4.1 Schema 全部追平完成 ===")
    return "\n\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验已生成文件是否与当前源脚本一致",
    )
    args = parser.parse_args()
    rendered = render_upgrade_sql()
    if args.check:
        if not OUTPUT_PATH.exists() or OUTPUT_PATH.read_text(
            encoding="utf-8"
        ) != rendered:
            raise SystemExit(f"升级脚本需要重新生成：{OUTPUT_PATH}")
        return 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
