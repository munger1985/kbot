# Oracle 原生报告交付

AWR、ASH、AWR 对比快捷入口保持 Oracle 实例筛选，不以商业许可声明控制入口或执行。执行账号、目标容器及数据库真实授权仍按目标连接配置生效。

## 报告流程

1. 按入口参数核对实例身份、时间和可用快照；沿用目录的快照绑定及依赖关系，不猜测快照编号。
2. 执行 Oracle 原生报告工具。AWR 对比先生成两段 AWR，再生成对比正文。
3. 依据真实执行回执和完整 HTML 直接交付下载链接；证据评估和最终交付均不调用通用诊断模型，不生成实施手册。
4. 空正文、零行、截断、缺少 HTML 结束标签或非受控来源不能满足报告完成义务，也不提供正式报告下载。比较失败时保留已经完成的配套报告。

## 包权限不足

Oracle 的报告接口没有执行权限时，也可能返回 ORA-00904，而不是 ORA-01031。
仅对三个固定报告工具，Driver 会在错误后读取当前目标的 SYS.DBA_PROCEDURES 与 SYS.ALL_PROCEDURES：
确认接口已安装但对当前账号不可见，才映射为 PRIVILEGE_MISSING。
接口不存在、仍可见或无法核查时保留原始字段错误分类，不按错误文本猜测权限。

应由 DBA 在目标连接所指向的容器内，核对诊断账号对 SYS.DBMS_WORKLOAD_REPOSITORY 的 EXECUTE 权限。
SELECT ANY DICTIONARY 允许读取快照，并不提供这个包的执行权限；PUBLIC 同义词存在也不代表已授权。
产品不会自动授予权限、切换 SYS 连接或更改商业许可。
专用监控账号的部署授权脚本位于 scripts/deployment/aiops_observability/oracle/，需由管理员审核后执行。

## 验证

运行 tests/unit/aiops_agent/test_aiops_native_oracle_reports.py，
并回归 test_aiops_workload_report_download.py、test_aiops_turn_answer.py、
test_aiops_dynamic_query.py 与 test_aiops_discovery_binding.py。
线上验证必须区分服务 readiness、报告工具执行和实际 HTML 下载；数据库缺少权限时不得声称报告已经恢复生成。
