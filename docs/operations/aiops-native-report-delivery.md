# Oracle 原生报告交付

AWR、ASH、AWR 对比报告使用固定目录工具生成完整 HTML，文件交付不依赖诊断模型。

自然语言请求中的相对日期以本轮创建时间及 Asia/Shanghai 为准，明确时间窗口覆盖模型从旧会话推测的日期。AWR 对比按较早窗口为基准、较晚窗口为比较绑定快照；发现快照后确定性续执行，不把输入顺序错误解释为证据缺失。

明确要求 AWR 分析时，能力解析同时冻结 include_analysis，两个窗口分别采集负载、等待与 Top SQL 结构化事实。模型只流式生成一次分析，不解析报告 HTML，不自动生成实施文档。分析中断保留已输出正文及成功报告附件；仅生成文件的请求不调用分析模型。

回归入口：tests/unit/aiops_agent/test_aiops_native_oracle_reports.py 与 test_aiops_capability_routing_fallback.py，覆盖用户原始复合请求、过期日期、反序区间、真实工具合同、单次流式生成和中断保留。

能力解析 Prompt 文件基线保持 1.0.0；部署按项目现有同步流程更新 BUILTIN / FILE_SEED，不覆盖网页维护版本。
