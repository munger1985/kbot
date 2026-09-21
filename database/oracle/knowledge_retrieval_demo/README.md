# 知识检索 App 问数 Demo CRM 数据

这里存放 知识检索 App 问数 Demo 使用的外部 Oracle 数据库对象。它不是 KBot 内部
Schema，不应通过 KBot 的 Oracle 初始化流程执行，也不包含任何真实账号、密码或 DSN。

## 对象范围

- `DEMO_CRM_CUSTOMER`：客户主数据
- `DEMO_CRM_CONTACT`：客户联系人
- `DEMO_CRM_OPPORTUNITY`：销售机会
- `DEMO_CRM_ACTIVITY`：可统计的沟通活动摘要，不保存沟通正文
- `DEMO_CRM_CUSTOMER_OVERVIEW`：一行一个客户的经营总览视图
- `DEMO_CRM_PIPELINE`：一行一个销售机会的问数视图
- `DEMO_CRM_ACTIVITY_LOG`：一行一个沟通活动的问数视图

`001_crm_schema.sql` 只创建表、索引和视图，不灌入测试数据。`002_crm_demo_data.sql`
提供一组可由 SQL Developer 手动执行的演示数据。沟通正文、会议纪要和
客户反馈后续作为文档进入 Knowledge Core，不与问数表重复保存。

测试数据脚本包含 10 个客户、10 个联系人、12 个销售机会和 12 条沟通活动摘要，覆盖
重点客户、区域分布、销售阶段、赢单、暂停、长期未跟进和风险活动等演示场景。脚本末尾
带有三条只读检查查询；脚本本身只执行插入和提交，执行前请确认目标 Schema 为空或尚未
执行过该脚本。

## 问文 PDF

`documents/` 下的 PDF 与上述客户数据保持一致，可在后续 知识检索 App/KC 流程中作为
用户资料上传。当前包含 7 份客户会议纪要和 1 份重点客户需求与风险摘要。可使用下面的
命令重新生成：

```bash
conda run -n cube python database/oracle/knowledge_retrieval_demo/generate_demo_pdfs.py
```

生成脚本使用虚构客户和联系人信息，不包含真实客户资料。

## 外部 Oracle 接入顺序

1. 在外部 Oracle 的 Demo Schema 执行 `001_crm_schema.sql`。
2. 为 Data Query 准备一个只读账号，至少允许读取上述表和视图，以及 Data Query
   Schema 发现所需的 Oracle 元数据。若账号不是对象所有者，`allowed_schemas` 应填
   对象所有者的 Schema 名称，而不是登录账号名称。
3. 在 知识检索 App 的问数模型流程中创建外部 `ORACLE` 数据源：填写外部主机、端口、
   Oracle Service Name、允许 Schema 和 TLS 设置；凭据通过 Data Query 的受控凭据流程
   提交，不写入脚本或仓库。
4. 连接测试成功后执行 Schema 发现，只选择三个问数视图作为第一版语义模型的数据集：
   `DEMO_CRM_CUSTOMER_OVERVIEW`、`DEMO_CRM_PIPELINE`、`DEMO_CRM_ACTIVITY_LOG`。
5. 后续再根据测试数据确认语义名称、同义词、指标、筛选条件和验证问题，然后发布问数
   模型并绑定 知识检索 Agent。

外部 Oracle 的真实连接、建表和权限执行结果需要在目标环境验证；当前仓库只提交可审计
的 DDL，不在本地猜测或保存外部连接凭据。
