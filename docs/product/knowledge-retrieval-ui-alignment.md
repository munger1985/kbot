# 知识检索 App 页面对齐

日期：2026-11-06。KBot 以 Ammolite Portal 的生产知识检索组件为基准，使用共享页面与控件样式；认证和 Domain 沿用各自平台，不修改 AIOps 主题。页面映射见 `ui/knowledge_retrieval/reference-pages.json`。

## 当前业务能力

业务入口保留知识检索、上传、审核、解析和文件；资料库管理使用 App 自有 libraries/documents，而不是公共 KC 或 Bundle。上传只表示已登记摄取任务，解析／索引 READY 和人工审核独立；审核通过且索引就绪才可发布。权限由平台入口与当前资料库成员共同核验，不默认授权同域全体用户。

知识页面提供直接检索与正式 Agent 问答。支持文本与最多 4 张、合计 8 MiB 查询图片、结果类型、数量、结构化筛选排序，分别展示主结果与关联图／图题／引用原文、通道状态和续取。图片方向是否可用依赖实际服务配置，不能把字节重复标成语义相似检索。正式回答展示 Runtime 实际交付与引用，不恢复旧正文唯一门禁。

内容通过公开 BFF 核权读取精确版本；不暴露内部 URL／对象句柄。主动 HTML／可执行原文仅下载。Domain 或资料库切换释放图片 URL、清空结果／游标、取消请求并核验请求代次；KBot 切域失败恢复原域权限、操作和资料列表。

## 实现与平台差异

- KBot：`ui/knowledge_retrieval/js/knowledge-library-workspace.js`、`knowledge-chat.js`、`knowledge-shell.js`。
- Ammolite：`KnowledgeLibraryWorkspace.vue`、`KnowledgeRetrievalView.vue` 和 `KnowledgeCompositionResults.vue`。
- KBot 使用已授权域列表和重新签发 Token；Ammolite 保留 Tenant／Domain 上下文。二者不共享检索和文档。
- 数据源／问数／X Search 保持各自现有合同，不扩展本次知识存储边界。

## 离线验收

`sync_knowledge_reference_styles.mjs` 从真实组件生成本 App 样式工件和来源 SHA。跨项目浏览器测试编译真实 Vue 组件，对照 9 页空／有数据及 4 种宽度；另覆盖 KBot 13 页、导航／权限／登录和 Domain 成功／失败恢复。工作区交互覆盖查询图片、图文关联、发布、上传网络失败同键重试以及忽略 abort 的迟到响应隔离。

~~~bash
node tests/contract/sync_knowledge_reference_styles.mjs --write
node tests/contract/sync_knowledge_reference_styles.mjs
node tests/contract/knowledge_page_alignment.mjs
node tests/contract/knowledge_retrieval_ui_style.mjs
~~~

所有浏览器数据为隔离夹具，不代表真实模型／数据库端到端或线上部署。Ammolite 还执行 `apps/portal-web` 的 `npm run build`。六项接入、离线命令与剩余真实门禁见[发布说明](../operations/app-owned-knowledge-release-readiness.md)。
