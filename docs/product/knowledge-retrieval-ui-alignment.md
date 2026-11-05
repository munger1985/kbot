# 知识检索 App 页面对齐

KBot 知识检索以 Ammolite Portal 的真实知识检索组件为页面基准，不复用 AIOps 的蓝色主题。页面来源、路由和平台差异登记在 `ui/knowledge_retrieval/reference-pages.json`。

## 页面结构

业务功能为知识检索、上传资料、资料审核、解析任务和知识库文件；管理配置为知识库管理、数据源、语义模型和 Agent 管理。KBot 既有工作台、X 实时搜索和 Domain 管理保留在扩展功能中。

知识检索使用会话侧栏、Agent 选择、连续问答和引用预览。资料上传支持独立文件与同一 Bundle 两种组织方式，包含文件角色、必要成员、SHA-256 和真实受理回执；受理成功不等于可检索。审核通过进入处理流水线，退回必须填写原因。解析任务和文件目录由 KC 执行分页、状态和关键词筛选，不在浏览器截断业务数据。

文件预览及原文通过 Main API 的公开 BFF 转发；不暴露内部地址、对象存储 URI 或服务凭据。可执行类型的原始文档只下载，不以 App 同源 Blob 页面执行。工作域切换继续使用已授权会话域列表与重新签发的 Token，并重新加载权限和资源。

## 平台差异

- 登录、首次改密和 Domain 权限沿用 KBot；资料处理页面仍要求 `knowledge_retrieval:knowledge_core_manage`，不因拆分导航放宽后端授权。
- 知识库配置、对象发现、模型验证发布使用右侧抽屉及现有 KBot 合同。KBot 暂无数据源更新、资源分享、完整解析结果和视觉资产详情合同，不显示不可操作入口。
- Ammolite 零知识库的解析页目前会保留加载状态；KBot 显示明确空态，不复制该缺陷。
- 本次只有 API 路由与客户端转发变化，不涉及数据库 Schema、初始化或部署变更。

## 样式和验收

`tests/contract/sync_knowledge_reference_styles.mjs` 从 Ammolite 的共享样式与生产组件生成 App 自有工件，记录来源 SHA-256。运行时不依赖 Ammolite 仓库，不修改 AIOps 样式。

```bash
node tests/contract/sync_knowledge_reference_styles.mjs --write
node tests/contract/sync_knowledge_reference_styles.mjs
node tests/contract/knowledge_page_alignment.mjs
node tests/contract/knowledge_retrieval_ui_style.mjs
conda run -n kbot4 python tests/contract/test_knowledge_retrieval_ui_style.py
conda run -n kbot4 python tests/unit/platform/test_knowledge_core_client.py
conda run -n kbot4 python -m unittest discover -s tests/unit/main_api -p test_main_api.py
conda run -n kbot4 python tests/acceptance/check_openapi_contracts.py
python3 tests/acceptance/check_4_0_boundaries.py
```

跨项目测试实际编译 Ammolite Vue 组件，对照 9 个页面的空/有数据、桌面/移动宽度、布局与控件规格，并验证上传及审核交互。KBot 页面回归覆盖 13 个入口、授权、登录和 Domain 切换。所有浏览器数据均为隔离夹具；此类回归不代表线上部署或真实依赖端到端验收。

默认引用 `../ammolite_cube/apps/portal-web` 的测试依赖，可通过 `AMMOLITE_ROOT` 指定参照仓库；截图和差异文件写入临时目录。
