"""从规范 DDL 生成只新增输出任务表的 Oracle 55 至 56 升级工件。"""

import argparse
import re

from tools.db.render_aiops_rebuild_schema import ROOT, _load_canonical_sections


OUTPUT_PATH = ROOT / "database/oracle/operations/upgrade_aiops_output_jobs_56.sql"


def render_upgrade_sql() -> str:
    manifest, sections = _load_canonical_sections()
    if manifest["schema_version"] != 56:
        raise ValueError("输出任务增量升级只对应规范 Schema 56")
    scripts = dict(sections)
    messaging = scripts["005_ops_messaging.sql"]
    start = messaging.index("-- 独立产物任务不复用诊断 Run 身份")
    additions = messaging[start:].strip()
    views = scripts["006_ops_fks_views.sql"]
    version_view = re.search(
        r"CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS.*?;",
        views, flags=re.DOTALL,
    )
    if version_view is None:
        raise ValueError("规范 Schema 缺少版本视图")
    return """-- 自动生成：tools/db/render_aiops_output_jobs_upgrade.py
-- 仅适用于已验收的 Schema 55；新增两张任务表，不删除或回填现有数据。
-- 由 Schema 所有者在备份后、停止受影响写入的发布窗口中执行；Oracle DDL 会自动提交。
-- 若中途失败，保留已创建对象并由管理员核对后续语句，不执行全量重建或自动删除。
WHENEVER OSERROR EXIT FAILURE ROLLBACK
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK
SET SQLBLANKLINES ON

DECLARE
    l_version NUMBER;
    l_contract VARCHAR2(128);
    l_existing NUMBER;
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION INTO l_version, l_contract
      FROM KBOT_V_OPS_SCHEMA_VERSION;
    IF l_version <> 55 OR l_contract <> 'aiops-oracle-v55-inspection-template-lifecycle' THEN
        raise_application_error(-20156, '仅允许从已验收的 AIOps Schema 55 升级');
    END IF;
    SELECT COUNT(*) INTO l_existing FROM user_tables
     WHERE table_name IN ('KBOT_OPS_OUTPUT_JOB', 'KBOT_OPS_OUTPUT_JOB_ATTEMPT');
    IF l_existing <> 0 THEN
        raise_application_error(-20157, '任务表已存在；请人工核对是否为中断升级，禁止覆盖');
    END IF;
END;
/

""" + additions + "\n\n" + version_view.group(0) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只核对生成物，不修改文件")
    args = parser.parse_args()
    rendered = render_upgrade_sql()
    if args.check:
        if not OUTPUT_PATH.exists() or OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
            print("AIOps 输出任务增量升级工件已过期")
            return 1
        print("AIOps 输出任务增量升级工件与规范 DDL 一致")
        return 0
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print("已生成 AIOps 输出任务增量升级工件，尚未应用数据库")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
