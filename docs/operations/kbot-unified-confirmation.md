# 静态 App 统一确认框

KBot 的 AIOps、知识检索和媒体工作台统一使用 `ui/shared/kbot-confirm.js`，不再各自拼装确认框或调用浏览器 `confirm()`。视觉规格沿用 Ammolite 的 `ConfirmDialog`，不改变服务端权限、版本或幂等校验。

## 原因与边界

业务系统旧确认框被 Shell 移入 `.platform-home-page` 后，命中直接子元素 `width: 100%` 的布局规则，导致宽屏下铺满页面。此前各页面又分别按 ID 修补尺寸，无法保证新入口一致。

共享组件通过原生 `dialog.showModal()` 挂到 `body`，保持顶层模态和浏览器焦点约束；宽度不超过 468px，窄屏两侧各保留 24px。长名称自动换行，高度受视口限制并可滚动。普通表单、详情抽屉、人工补证与报告参数表单不属于操作确认，不强行改为确认框。

## 调用方式

各 App Shell 暴露 `confirmAction({ title, message, confirmLabel, tone })`，返回 `Promise<boolean>`。`tone` 为 `warning` 或 `danger`。业务代码必须 `await` 返回结果，只有确认后才能发出变更请求。

AIOps 另提供：

- `confirmActionWithOption`：返回 `{ confirmed, optionSelected, reason }`。运维资料的物理文件删除选项每次默认关闭，取消时必须返回 `optionSelected: false`。
- `confirmActionWithReason`：额外传入 `reason: { label, placeholder, maxLength }`，用于下线原因；必填且不能只有空格，提交前去除首尾空白。

标题、提示及业务对象名称按纯文本渲染。默认聚焦取消按钮；Esc、取消、遮罩点击均不提交，关闭后恢复触发元素焦点。已有确认尚未结束时，重复调用直接返回未确认，不替换原操作。双击确认只结算一次。

## 已统一的入口

- AIOps：业务系统上线/下线与删除草稿、离开未保存工作区、Agent 删除、监控源启停/删除与 Webhook 轮换、巡检计划生命周期、恢复演练审核、模板操作、运维资料和知识库删除、案例发布、会话删除、运维事实确认、跳过补证、受控变更批准与人工结果回填、未来观察取消。
- 知识检索：Domain 停用、Agent 归档、Knowledge Core 删除、会话及搜索记录删除、失败文件重新解析。
- 媒体工作台：模型状态变更/归档/永久删除、图片生成记录删除。

浏览器离开/刷新标签页的 `beforeunload` 安全提示受浏览器控制，不能替换为应用自定义组件。尚未发布、属于其他工作区改动的新功能不纳入本次迁移提交。

## 验证

`node tests/contract/kbot_confirmation_dialog.mjs` 使用真实业务系统 HTML、Shell、业务脚本和 Chromium，以隔离请求验证上线/下线、版本头、单次提交、取消、原因校验、复选项重置、长文本安全、抽屉层级与焦点恢复，并检查 2048、1440、420、320px 视口。只审计受版本控制的应用脚本，不扫描第三方库或其他任务尚未发布的新增页面。

本组件仅替换 KBot 的静态前端实现；Ammolite 已使用自己的共享 Vue 组件，不重复增加第二套确认框。
