# AIOps 多轮人机协同 (HITL) 诊断交互方案

## 一、设计目标

实现 AIOps 诊断 Agent 在分析过程中**动态判断数据不足 → 向用户索要数据 → 用户回填 → 继续分析**的多轮交互能力。核心原则：

- **不预判**：Planner 不在规划阶段决定何时需要用户介入（此时无运行时数据）
- **动态决策**：由 RCA 引擎（DBAnalysisSkill）在分析过程中根据实际证据判断是否充分
- **可多轮**：支持 N 轮往复，直到根因收敛
- **容错**：用户执行 SQL 报错时，Agent 能够自我修正并给出替代方案

---

## 二、架构设计

### 2.1 核心思想：从"预排工具"到"状态挂起"

```
❌ 旧思路: Planner 预排 ask-user-skill → 执行到它 → 中断
   └─ 问题: Planner 没有运行时数据，无法预判何时需要用户介入

✅ 新思路: Planner 只做宏观阶段规划 → Skill 运行时动态判断 → yield WAIT_FOR_USER
   └─ 任何 Skill 都可以触发中断，无需 Planner 感知
```

### 2.2 完整交互流程

```
用户: "数据库突然变慢，帮我看看"
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Planner 生成: [db-metric-skill → db-analysis-skill]        │
│  宏观路径: 采集 → 分析。不预判是否需要用户介入               │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  db-metric-skill 执行:                                       │
│  ├─ Prometheus: CPU 45%, IOPS 正常, 内存正常                  │
│  ├─ 专家 SQL: db_active_session_wait()                       │
│  │   → 大量 "enq: TX - row lock contention"                  │
│  └─ 数据汇总到 ctx.metric_results / ctx.monitor_results       │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  db-analysis-skill 执行 (第1次):                              │
│                                                              │
│  ├─ [新增] 数据充分性前置检查 (get_llm_json, ~500 tokens):    │
│  │   "基于 Prometheus + 等待事件，能否给出确定性根因？"         │
│  │                                                           │
│  ├─ LLM 判定: insufficient                                   │
│  │   {                                                       │
│  │     "verdict": "insufficient",                            │
│  │     "reason": "排除资源瓶颈，等待事件指向行锁，"             │
│  │               "但缺少锁持有者细节，无法定位阻塞源头",        │
│  │     "sql_to_run": "SELECT s.sid, s.serial#, ...",         │
│  │     "expected_fields": ["blocking_sid", "username", ...]   │
│  │   }                                                       │
│  │                                                           │
│  ├─ 🔴 yield { type: WAIT_FOR_USER, content: {...} }         │
│  └─ return (不抛异常)                                        │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Orchestrator 检测 WAIT_FOR_USER:                             │
│  ├─ 保存 ctx 完整快照 → kbot_ops_pending_request             │
│  ├─ current_step_index = idx (不是 idx+1，当前 skill 未完成)   │
│  ├─ 更新 kbot_md_conv_context.is_suspended = 1               │
│  ├─ 向前端推送 WAIT_FOR_USER 事件                             │
│  └─ 正常结束 SSE 流 (SSE 连接关闭)                             │
└─────────────────────────────────────────────────────────────┘
  │
  │  ⏳ 用户去数据库执行 SQL，可能耗时几分钟到几小时
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  POST /ops/chat/resume                                       │
│  { request_id, user_data: { rows: [...] } }                  │
│  (或 user_note: "Ora-00942: 没有 v$lock 视图查询权限")        │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Orchestrator.resume():                                      │
│  ├─ 从 DB 恢复 ctx 完整快照                                   │
│  ├─ ctx["is_resuming"] = True                                │
│  ├─ ctx["hitl_history"].append({ request_id, user_data })     │
│  │   ↑ 关键：追加而非覆盖，形成排查历史 Timeline                 │
│  ├─ 重新注入基础设施 (_prometheus_client 等)                   │
│  └─ 从 current_step_index (= idx) 重新驱动 db-analysis-skill  │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  db-analysis-skill 执行 (第2次):                              │
│  ├─ ctx["is_resuming"] = True → 跳过充分性检查               │
│  ├─ 注入 ctx["hitl_history"] 到 LLM prompt:                   │
│  │   "第1轮排查: 你执行了锁查询 → 发现 SID 102 持有排他锁"     │
│  │   "第2轮排查: 执行计划缺失 → 用户提供了 explain plan"       │
│  ├─ LLM 有完整证据链 → 流式输出最终 RCA                        │
│  └─ 如果还不够 → 可以再次 WAIT_FOR_USER (天然支持多轮)         │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 状态机定义

```
                    ┌──────────────┐
                    │    IDLE      │
                    └──────┬───────┘
                           │ POST /ops/chat
                           ▼
                    ┌──────────────┐
              ┌─────│  EXECUTING   │◄──────────────┐
              │     └──────┬───────┘                │
              │            │ skill yields            │
              │            │ WAIT_FOR_USER           │
              │            ▼                         │
              │     ┌──────────────┐    resume      │
              │     │  SUSPENDED   │────────┼───────┘
              │     └──────┬───────┘        │
              │            │ timeout/cancel  │
              │            ▼                 │
              │     ┌──────────────┐        │
              │     │  TERMINATED  │        │
              │     └──────────────┘        │
              │                            │
              │     ┌──────────────┐        │
              └─────│    DONE      │◄───────┘
                    └──────────────┘
```

---

## 三、关键数据结构

### 3.1 OpsContextMemory 扩展

```python
# agent/common/ops_context.py

class OpsContextMemory(TypedDict):
    # ======================== 现有字段（保持不变） ========================
    trace_id: str
    user_id: str
    session_id: str
    agent_id: int
    trigger_type: Literal["manual", "webhook", "cron"]
    command_or_query: str
    llm_model: str
    embedding_model: str
    instance_id: str
    db_type: str
    version_code: int
    db_role: Literal["primary", "standby", "cluster_node"]
    environment: Literal["prod", "stg", "dev"]
    monitor_type: str
    prometheus_instance_label: str | None
    alert_context: dict[str, Any] | None
    runtime_plan: ExecutionPlan | None
    current_step_index: int
    current_execution: SkillExecutionContext | None
    execution_history: list[SkillExecutionContext]
    approval_context: dict[str, Any] | None
    variables: dict[str, Any]
    metric_results: list[dict[str, Any]]
    monitor_results: list[dict[str, Any]]
    os_log_snapshots: list[str]
    doc_results: list[dict[str, Any]]
    temp: dict[str, Any]

    # ======================== HITL 新增字段 ========================
    is_resuming: bool
    # ↑ 恢复标志: Skill 检测到此字段为 True 时跳过充分性检查

    hitl_history: list[dict[str, Any]]
    # ↑ 多轮排查 Timeline（追加而非覆盖）:
    # [
    #   {
    #     "round": 1,
    #     "request_id": "req_xxx",
    #     "reason": "需要查锁等待详情",
    #     "sql_to_run": "SELECT ...",
    #     "user_data": { "rows": [...] },
    #     "user_error": null,
    #     "submitted_at": "2026-07-02T10:30:00Z"
    #   },
    #   {
    #     "round": 2,
    #     "request_id": "req_yyy",
    #     "reason": "上次权限不足，尝试替代方案",
    #     "sql_to_run": "SELECT ... FROM dba_locks WHERE ...",
    #     "user_data": { "rows": [...] },
    #     "user_error": null,
    #     "submitted_at": "2026-07-02T10:35:00Z"
    #   }
    # ]
```

### 3.2 数据库新表：`kbot_ops_pending_request`

```sql
-- ==========================================
-- AIOps HITL 挂起请求表
-- ==========================================
CREATE TABLE kbot_ops_pending_request (
    id                  NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    request_id          VARCHAR2(36) NOT NULL,       -- 挂起请求 UUID v7
    session_id          VARCHAR2(64) NOT NULL,        -- 关联会话 ID
    user_id             VARCHAR2(256) NOT NULL,       -- 用户 ID
    agent_id            NUMBER NOT NULL,              -- Agent ID
    instance_id         VARCHAR2(36) NOT NULL,        -- 目标实例 ID
    entry_id            VARCHAR2(36) NOT NULL,        -- 原始 entry_id

    -- 挂起上下文 (给用户看的)
    suspend_reason      CLOB NOT NULL,                -- LLM 给出的挂起原因
    user_prompt         CLOB NOT NULL,                -- 展示给用户的操作指引 (Markdown)
    sql_to_run          CLOB,                         -- 需要用户执行的 SQL
    expected_fields     CLOB,                         -- JSON: 期望的数据字段定义
                                                     -- [{"field":"lock_count","type":"number","desc":"锁数量"}]

    -- 执行状态快照 (用于恢复)
    suspended_by_skill  VARCHAR2(128) NOT NULL,       -- 哪个 Skill 触发的挂起
    current_step_index  NUMBER DEFAULT 0,             -- 当前步骤索引 (不是 +1)
    completed_steps     CLOB,                         -- JSON: execution_history
    accumulated_results CLOB,                         -- JSON: metric/monitor/doc_results
    pending_variables   CLOB,                         -- JSON: ctx["variables"] 快照
    hitl_history        CLOB,                         -- JSON: 完整的 hitl_history Timeline
    runtime_plan        CLOB,                         -- JSON: ctx["runtime_plan"] 快照

    -- 生命周期管理
    status              VARCHAR2(16) DEFAULT 'pending', -- pending / answered / timeout / cancelled
    requested_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    responded_at        TIMESTAMP,
    timeout_at          TIMESTAMP,                     -- 默认 +30 分钟
    reminder_count      NUMBER DEFAULT 0,              -- 催办次数

    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_pend_session FOREIGN KEY (session_id)
        REFERENCES kbot_md_conv_context(session_id),
    CONSTRAINT fk_pend_agent FOREIGN KEY (agent_id)
        REFERENCES kbot_md_agent(agent_id),
    CONSTRAINT fk_pend_instance FOREIGN KEY (instance_id)
        REFERENCES kbot_ops_db_instance(instance_id),
    CONSTRAINT ck_pend_status CHECK (status IN ('pending', 'answered', 'timeout', 'cancelled'))
);

CREATE INDEX idx_pend_session ON kbot_ops_pending_request(session_id, status);
CREATE INDEX idx_pend_timeout ON kbot_ops_pending_request(status, timeout_at);
CREATE INDEX idx_pend_request_id ON kbot_ops_pending_request(request_id);

COMMENT ON TABLE kbot_ops_pending_request IS 'AIOps HITL 挂起请求快照表: 存储 Agent 等待用户输入时的完整状态';
COMMENT ON COLUMN kbot_ops_pending_request.request_id IS '挂起请求唯一标识 (UUID v7)';
COMMENT ON COLUMN kbot_ops_pending_request.sql_to_run IS 'LLM 生成的用户需执行的 SQL (可一键复制)';
COMMENT ON COLUMN kbot_ops_pending_request.expected_fields IS '期望用户返回的字段列表, 前端据此动态渲染输入表单';
COMMENT ON COLUMN kbot_ops_pending_request.suspended_by_skill IS '触发挂起的 Skill 名称, 用于恢复时路由';
COMMENT ON COLUMN kbot_ops_pending_request.current_step_index IS '挂起时正在执行的步骤索引, 恢复时从此索引重新驱动 Skill';
COMMENT ON COLUMN kbot_ops_pending_request.hitl_history IS '完整的 HITL 多轮交互 Timeline, 恢复时注入 ctx';
COMMENT ON COLUMN kbot_ops_pending_request.runtime_plan IS 'Runtime Plan 快照, 恢复时重建 ctx';
```

### 3.3 现有表扩展

```sql
-- kbot_md_conv_context 新增挂起追踪字段
ALTER TABLE kbot_md_conv_context ADD (
    pending_request_id VARCHAR2(36),
    is_suspended       NUMBER(1) DEFAULT 0
);

COMMENT ON COLUMN kbot_md_conv_context.pending_request_id IS '当前活跃的挂起请求 ID, NULL = 无挂起';
COMMENT ON COLUMN kbot_md_conv_context.is_suspended IS '会话挂起标记: 0=正常运行, 1=等待用户输入中';
```

### 3.4 PacketType 扩展

```python
# core/dictionary.py 的 PacketType 枚举新增:

WAIT_FOR_USER = "wait_for_user"
# Agent 需要用户输入时才发出。content 包含:
#   - request_id: 恢复时需要回传
#   - reason: 为什么需要更多数据
#   - sql_to_run: 用户需执行的 SQL
#   - expected_fields: 期望的字段列表
#   - timeout_at: 超时时间
# 前端收到此事件后应渲染用户输入表单
```

---

## 四、核心代码实现

### 4.1 DBAnalysisSkill 改造（最关键的部分）

```python
# skills/skill_libs/db-analysis-skill/db_analysis_skill_core.py

import json
import uuid
from typing import Any, AsyncGenerator
from loguru import logger

from skills import BaseSkill, SkillMeta, SkillDomain, SkillRunMode
from agent.common.ops_context import OpsContextMemory
from core.dictionary import PacketType
from utils.clients import AIModelClient
from agent.prompt import default_prompt
from core.config import get_prompt_config


class DBAnalysisSkill(BaseSkill):
    """
    AIOps RCA 引擎 v3 — 支持 HITL 多轮人机协同。

    新增能力:
      - 数据充分性动态检查: LLM 判断现有证据是否足以定位根因
      - 恢复模式: 检测 ctx["is_resuming"] 后跳过检查，直接融合用户数据
      - 多轮 Timeline: 通过 ctx["hitl_history"] 维护完整排查历史
      - 用户错误容错: 识别用户返回的 SQL 报错并生成替代方案
    """
    meta = SkillMeta(
        name="db-analysis-skill",
        description="融合监控+诊断+用户补充数据进行 RCA，支持多轮 HITL 交互",
        domain=SkillDomain.OPS,
        run_mode=SkillRunMode.READ_ONLY,
    )

    def __init__(self):
        super().__init__()
        self.model_client = AIModelClient()

    async def run_stream(
        self, context: OpsContextMemory, **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        trace_id = context.get("trace_id")
        query_text = context["command_or_query"]
        instance_id = context["instance_id"]
        db_type = context["db_type"]
        environment = context["environment"]
        llm_model = context["llm_model"]

        metric_results = context.get("metric_results", [])
        monitor_results = context.get("monitor_results", [])
        doc_results = context.get("doc_results", [])

        # ---- HITL: 恢复模式检测 ----
        is_resuming = context.get("is_resuming", False)
        hitl_history: list[dict] = context.get("hitl_history", [])

        logger.info(
            f"[{trace_id}] DBAnalysisSkill v3 激活 | 实例: {instance_id} | "
            f"恢复模式: {is_resuming} | HITL轮次: {len(hitl_history)} | "
            f"监控数据: {len(monitor_results)} | 诊断数据: {len(metric_results)}"
        )

        # ---- HITL: 检查用户最近一次提交是否包含错误 ----
        # 如果用户执行 SQL 报错，应该在下一轮自动修正 SQL
        if is_resuming and hitl_history:
            last_round = hitl_history[-1]
            user_error = last_round.get("user_error")
            if user_error:
                logger.info(
                    f"[{trace_id}] 检测到用户上次执行 SQL 报错: "
                    f"{str(user_error)[:200]}, 将触发 SQL 自愈修正"
                )

        # ---- 数据空检测 (硬编码规则，不消耗 LLM Token) ----
        if not metric_results and not monitor_results and not is_resuming:
            logger.warning(f"[{trace_id}] 无任何数据，拒绝盲目猜测")
            yield {
                "type": PacketType.WARNING,
                "content": (
                    "⚠️ 未采集到任何监控或诊断数据，诊断大脑拒绝盲目猜测。"
                    "请检查 Prometheus 连接和数据库连接是否正常。"
                )
            }
            return

        # ---- HITL: 数据充分性前置检查 (仅在非恢复模式下) ----
        if not is_resuming:
            sufficiency = await self._check_data_sufficiency(
                query_text=query_text,
                metric_results=metric_results,
                monitor_results=monitor_results,
                doc_results=doc_results,
                hitl_history=hitl_history,
                db_type=db_type,
                environment=environment,
                llm_model=llm_model,
            )

            if sufficiency["verdict"] == "insufficient":
                request_id = f"req_{uuid.uuid4().hex[:12]}"
                suspend_payload = {
                    "request_id": request_id,
                    "reason": sufficiency["reason"],
                    "sql_to_run": sufficiency.get("sql_to_run", ""),
                    "expected_fields": sufficiency.get("expected_fields", []),
                    "suspended_by": "db-analysis-skill",
                    "timeout_at": sufficiency.get("timeout_at"),
                }
                yield {
                    "type": PacketType.WAIT_FOR_USER,
                    "content": suspend_payload,
                }
                return  # 🔴 中断，正常返回

        # ---- 构建融合证据链（同现有逻辑） ----
        knowledge_context = self._build_knowledge_context(doc_results)
        monitor_context = self._build_monitor_context(monitor_results)
        metric_context = self._build_metric_context(metric_results)

        # ---- HITL: 构建多轮排查 Timeline 上下文 ----
        hitl_context = self._build_hitl_context(hitl_history)

        # ---- 渲染最终诊断 Prompt ----
        system_prompt = await default_prompt.generate(
            get_prompt_config().ops_diagnosis,
            environment=environment,
            db_type=db_type,
            version_code=context.get("version_code", 0),
            db_role=context.get("db_role", "primary"),
            variables=json.dumps(
                {k: v for k, v in context.get("variables", {}).items()
                 if not k.startswith("_")},
                ensure_ascii=False,
            ),
            metric_results=metric_context,
            monitor_results=monitor_context,
            os_log_snapshots=json.dumps(
                context.get("os_log_snapshots", []), ensure_ascii=False, indent=2
            ),
            knowledge_context=knowledge_context,
            standalone_query=query_text,
            # HITL 新增注入
            hitl_context=hitl_context,
        )

        yield {
            "type": PacketType.THOUGHT,
            "content": "正在召集 DBA 专家大脑融合多路数据进行 RCA 根因推演...\n"
        }

        # ---- 流式输出 RCA 报告 (同现有逻辑) ----
        is_thinking = False
        output_buffer = ""
        try:
            async for chunk in self.model_client.get_llm_stream_parsed(
                model_name=llm_model,
                prompt=[{"role": "user", "content": system_prompt}],
                temperature=0.1
            ):
                if not chunk:
                    continue

                if hasattr(chunk, "reasoning_content") and chunk.reasoning_content:
                    yield {"type": PacketType.THOUGHT,
                           "content": chunk.reasoning_content}
                    continue

                if not chunk.content:
                    continue

                output_buffer += chunk.content

                while output_buffer:
                    if not is_thinking:
                        if "<thought>" in output_buffer:
                            parts = output_buffer.split("<thought>", 1)
                            if parts[0].strip():
                                yield {"type": PacketType.ANSWER,
                                       "content": parts[0]}
                            is_thinking = True
                            output_buffer = parts[1]
                        elif "<" in output_buffer and any(
                            tag.startswith(output_buffer.rsplit("<", 1)[1])
                            for tag in ["thought>", "/thought>"]
                        ):
                            break
                        else:
                            yield {"type": PacketType.ANSWER,
                                   "content": output_buffer}
                            output_buffer = ""
                    else:
                        if "</thought>" in output_buffer:
                            parts = output_buffer.split("</thought>", 1)
                            if parts[0].strip():
                                yield {"type": PacketType.THOUGHT,
                                       "content": parts[0]}
                            is_thinking = False
                            output_buffer = parts[1]
                        elif "<" in output_buffer and "/thought>".startswith(
                            output_buffer.rsplit("<", 1)[1]
                        ):
                            break
                        else:
                            yield {"type": PacketType.THOUGHT,
                                   "content": output_buffer}
                            output_buffer = ""

            if output_buffer.strip():
                m_type = PacketType.THOUGHT if is_thinking else PacketType.ANSWER
                yield {"type": m_type, "content": output_buffer}

        except Exception as e:
            logger.error(f"DBAnalysisSkill 运行异常: {e}")
            yield {"type": PacketType.ERROR,
                   "content": f"⚠️ 分析中断: {str(e)}\n"}

    # ==================================================================
    # HITL: 数据充分性前置检查
    # ==================================================================

    async def _check_data_sufficiency(
        self,
        query_text: str,
        metric_results: list[dict],
        monitor_results: list[dict],
        doc_results: list[dict],
        hitl_history: list[dict],
        db_type: str,
        environment: str,
        llm_model: str,
    ) -> dict[str, Any]:
        """
        前置检查: LLM 判断现有证据是否足以给出确定性 RCA。

        优化: 严格控制 max_tokens (~500), 只输出 JSON 判定。
        规则前置: 硬编码可判断的情况先处理，减少 LLM 调用。
        """

        # ---- 规则前置 (无需 LLM): 数据完全为空 ----
        if not metric_results and not monitor_results:
            return {
                "verdict": "insufficient",
                "reason": (
                    f"Prometheus 和数据库诊断工具均未返回任何数据。"
                    f"请检查 {environment} 环境 {db_type} 实例的监控连接和数据库连接。"
                ),
                "sql_to_run": "",
                "expected_fields": [],
            }

        # ---- 规则前置 (无需 LLM): 已有多轮交互但无进展 ----
        if len(hitl_history) >= 5:
            return {
                "verdict": "sufficient",
                "reason": "已进行 5 轮排查，建议基于现有证据给出初步结论",
                "sql_to_run": "",
                "expected_fields": [],
            }

        # ---- LLM 充分性检查 ----
        evidence = self._build_evidence_summary(
            metric_results, monitor_results, doc_results
        )
        hitl_summary = self._build_hitl_context(hitl_history)

        prompt = await default_prompt.generate(
            get_prompt_config().ops_sufficiency_check,
            query_text=query_text,
            db_type=db_type,
            environment=environment,
            evidence_summary=evidence,
            hitl_context=hitl_summary,
        )

        try:
            result = await self.model_client.get_llm_json(
                model_name=llm_model,
                prompt=prompt,
                temperature=0,
                max_tokens=500,  # 严格控制: 只输出 JSON
            )
            return {
                "verdict": result.get("verdict", "sufficient"),
                "reason": result.get("reason", ""),
                "sql_to_run": result.get("sql_to_run", ""),
                "expected_fields": result.get("expected_fields", []),
            }
        except Exception as e:
            logger.error(f"充分性检查 LLM 调用失败: {e}, 降级为直接分析")
            return {"verdict": "sufficient"}

    # ==================================================================
    # HITL: 构建多轮排查 Timeline 上下文
    # ==================================================================

    def _build_hitl_context(self, hitl_history: list[dict]) -> str:
        """将多轮 HITL 交互渲染为 LLM 可读的排查历史"""
        if not hitl_history:
            return "（这是第一轮排查，暂无人工补充数据）"

        parts = ["## 📋 多轮排查 Timeline"]
        for entry in hitl_history:
            round_num = entry.get("round", "?")
            reason = entry.get("reason", "")
            sql = entry.get("sql_to_run", "")
            user_error = entry.get("user_error")
            user_data = entry.get("user_data")

            parts.append(f"\n### 第 {round_num} 轮")
            parts.append(f"- **Agent 请求原因**: {reason}")
            if sql:
                parts.append(f"- **Agent 让用户执行的 SQL**:\n```sql\n{sql}\n```")

            if user_error:
                parts.append(
                    f"- **⚠️ 用户执行报错**: {user_error}\n"
                    f"  注意: 用户没有执行此 SQL 的权限或 SQL 语法不兼容，"
                    f"  请尝试替代方案或使用更通用的系统视图。"
                )
            elif user_data:
                data_str = json.dumps(user_data, ensure_ascii=False,
                                      default=str, indent=2)
                # Token 控制: 最多保留 2000 字符
                if len(data_str) > 2000:
                    data_str = data_str[:2000] + "\n... (数据已截断)"
                parts.append(f"- **用户回填数据**:\n```json\n{data_str}\n```")

        return "\n".join(parts)

    # ==================================================================
    # 已有辅助方法 (保持不变)
    # ==================================================================

    def _build_knowledge_context(self, doc_results: list[dict]) -> str:
        if not doc_results:
            return "当前无匹配的专家 SOP 手册。"
        return "\n".join(
            f"- 《{d.get('file_name', '?')}》: {d.get('text_content', '')}"
            for d in doc_results
        )

    def _build_evidence_summary(self, metric_results, monitor_results,
                                 doc_results) -> str:
        """构建证据摘要（用于充分性检查）"""
        parts = []
        if monitor_results:
            parts.append(f"- Prometheus 监控数据: {len(monitor_results)} 条")
        if metric_results:
            parts.append(f"- 数据库诊断结果: {len(metric_results)} 条")
        if doc_results:
            parts.append(f"- 运维 SOP 手册: {len(doc_results)} 篇")
        if not parts:
            parts.append("- 无任何证据")
        return "\n".join(parts)

    def _build_monitor_context(self, monitor_results) -> str:
        """同现有实现，略"""
        # ... 现有 _build_monitor_context 逻辑 ...
        return ""

    def _build_metric_context(self, metric_results) -> str:
        """同现有实现，略"""
        # ... 现有 _build_metric_context 逻辑 ...
        return ""
```

### 4.2 OpsOrchestrator 改造

```python
# agent/orchestrator/ops_orchestrator.py (关键改动部分)

async def execute_ops_stream_pipeline(self, ...):
    # ... 现有初始化代码 (L72-L178) 保持不变 ...

    # ---- 注入基础设施引用 ----
    ctx["variables"]["_prometheus_client"] = self.prometheus_client
    ctx["variables"]["_metric_registry"] = self.metric_registry
    ctx["variables"]["_ops_db_executor"] = self.ops_db_executor

    # ---- HITL: 初始化 Timeline ----
    ctx["hitl_history"] = []
    ctx["is_resuming"] = False

    # ---- 步骤执行循环 (增加中断检测) ----
    plan_steps = ctx["runtime_plan"]["steps"] if ctx["runtime_plan"] else []
    final_answer_accumulator = ""

    for idx, step in enumerate(plan_steps):
        ctx["current_step_index"] = idx

        runtime = SkillRuntime(context=ctx)
        exec_info = runtime.create_execution_context(step_config=step)
        skill_name = exec_info["skill"]

        ctx["current_execution"] = cast(Any, exec_info)

        yield {
            "type": PacketType.CALL,
            "content": {"skill": skill_name,
                        "description": exec_info["resolved_input"]}
        }

        skill_instance = self.skill_manager.get_skill_instance(skill_name)
        if not skill_instance:
            exec_info.update({"status": "failed",
                              "error": f"组件 {skill_name} 未激活"})
            ctx["execution_history"].append(cast(Any, exec_info))
            yield {"type": PacketType.ERROR,
                   "content": f"⚠️ 关键自愈组件 [{skill_name}] 离线, 本步骤跳过。"}
            continue

        # 安全熔断门禁
        gate_result = self._check_safety_gate(ctx, skill_instance, skill_name)
        if not gate_result["allowed"]:
            exec_info.update({"status": "blocked",
                              "error": gate_result["reason"]})
            ctx["execution_history"].append(cast(Any, exec_info))
            yield {"type": PacketType.ERROR,
                   "content": f"🚫 安全熔断: {gate_result['reason']}"}
            continue

        try:
            _monitor_snapshot = len(ctx.get("monitor_results", []))
            _metric_snapshot = len(ctx.get("metric_results", []))

            async for packet in runtime.execute_skill(skill_instance, exec_info):
                p_type = packet.get("type")
                content = packet.get("content")

                # ──── HITL: 中断检测 ────
                if p_type == PacketType.WAIT_FOR_USER:
                    suspend_ctx = content
                    request_id = suspend_ctx["request_id"]

                    logger.info(
                        f"[{ctx['trace_id']}] 🔴 HITL 中断触发 | "
                        f"Skill: {skill_name} | Step: {idx} | "
                        f"RequestID: {request_id}"
                    )

                    # 持久化完整快照
                    await self._suspend_execution(
                        ctx=ctx,
                        suspend_ctx=suspend_ctx,
                        request_id=request_id,
                        current_step_index=idx,  # ← 不是 idx+1
                        entry_id=entry_id,
                        start_time=start_time,
                    )

                    # 更新会话挂起状态
                    await self._mark_session_suspended(
                        ctx["session_id"], request_id
                    )

                    # 推送中断包
                    yield packet

                    # 添加 timeout 信息
                    timeout_at = datetime.now(timezone.utc) + timedelta(minutes=30)
                    suspend_ctx["timeout_at"] = timeout_at.isoformat()

                    yield {
                        "type": PacketType.DONE,
                        "content": {
                            "entry_id": entry_id,
                            "status": "suspended",
                            "request_id": request_id,
                        }
                    }
                    return  # 🔴 正常结束

                # ──── 原有数据沉淀逻辑 ────
                if p_type == PacketType.ANSWER:
                    final_answer_accumulator += (content or "")

                if p_type == PacketType.MONITOR_RESULTS:
                    if isinstance(content, dict) and "data" in content:
                        ctx["monitor_results"].append({...})
                elif p_type == PacketType.METRIC_RESULTS:
                    if isinstance(content, dict) and "data" in content:
                        ctx["metric_results"].append({...})

                if p_type in DISPLAY_PACKET_TYPES:
                    yield packet

            # 步骤正常完成
            exec_info.update({"status": "success"})
            output_var = exec_info.get("output_var")
            if output_var:
                new_monitor = ctx.get("monitor_results", [])[_monitor_snapshot:]
                new_metric = ctx.get("metric_results", [])[_metric_snapshot:]
                step_data = {"monitor": new_monitor, "metric": new_metric}
                ctx["variables"][output_var] = json.dumps(
                    step_data, ensure_ascii=False, default=str
                )
            ctx["execution_history"].append(cast(Any, exec_info))
            ctx["current_execution"] = None

        except Exception as e:
            logger.error(f"[Orchestrator] Skill [{skill_name}] 异常: {e}")
            exec_info.update({"status": "failed", "error": str(e)})
            ctx["execution_history"].append(cast(Any, exec_info))
            ctx["current_execution"] = None
            continue

    # ... 现有持久化逻辑 ...


# ==================================================================
# HITL: 恢复执行
# ==================================================================

async def resume_ops_stream_pipeline(
    self,
    background_tasks: BackgroundTasks,
    request_id: str,
    user_data: dict[str, Any] | None,
    user_note: str | None,
    user_error: str | None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    从挂起状态恢复执行。

    Args:
        request_id: 挂起请求 ID (来自 WAIT_FOR_USER 包)
        user_data: 用户回填的数据
        user_note: 用户备注
        user_error: 用户执行 SQL 时的报错信息 (如 ORA-00942)
    """
    # 1. 从数据库恢复挂起状态
    pending = await self._pending_repo.get_by_request_id(request_id)
    if not pending:
        yield {"type": PacketType.ERROR,
               "content": f"❌ 挂起请求 {request_id} 不存在或已过期"}
        yield {"type": PacketType.DONE,
               "content": {"entry_id": "N/A", "status": "error"}}
        return

    if pending["status"] != "pending":
        yield {"type": PacketType.ERROR,
               "content": f"❌ 挂起请求 {request_id} 状态为 {pending['status']}，不可恢复"}
        yield {"type": PacketType.DONE,
               "content": {"entry_id": pending.get("entry_id", "N/A"),
                           "status": "already_handled"}}
        return

    logger.info(
        f"[HITL Resume] request_id={request_id} | "
        f"session={pending['session_id']} | "
        f"has_data={user_data is not None} | "
        f"has_error={user_error is not None}"
    )

    # 2. 重建 OpsContextMemory
    ctx = self._rebuild_context_from_pending(pending)

    # 3. 恢复基础设施引用
    ctx["variables"]["_prometheus_client"] = self.prometheus_client
    ctx["variables"]["_metric_registry"] = self.metric_registry
    ctx["variables"]["_ops_db_executor"] = self.ops_db_executor

    # 4. HITL: 追加本轮到 Timeline (不是覆盖!)
    hitl_history: list[dict] = ctx.get("hitl_history", [])
    round_num = len(hitl_history) + 1

    hitl_history.append({
        "round": round_num,
        "request_id": request_id,
        "reason": pending.get("suspend_reason", ""),
        "sql_to_run": pending.get("sql_to_run", ""),
        "user_data": user_data,
        "user_error": user_error,
        "user_note": user_note,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    })
    ctx["hitl_history"] = hitl_history

    # 5. 标记为恢复模式
    ctx["is_resuming"] = True

    # 6. 从断点继续执行
    plan_steps = ctx["runtime_plan"]["steps"]
    current_step_index = pending["current_step_index"]
    entry_id = pending["entry_id"]
    start_time = pending.get("requested_at", datetime.now(timezone.utc))

    logger.info(
        f"[HITL Resume] 从 Step {current_step_index} ({plan_steps[current_step_index].get('skill')}) 恢复 | "
        f"总步骤: {len(plan_steps)} | HITL 轮次: {len(hitl_history)}"
    )

    # 7. 标记挂起为已处理
    await self._pending_repo.mark_answered(request_id)
    await self._clear_session_suspended(pending["session_id"])

    # 8. 继续执行步骤循环 (从中断的 skill 开始)
    #  当前 skill 会检测到 is_resuming=True, 走恢复分支
    final_answer_accumulator = ""
    for idx in range(current_step_index, len(plan_steps)):
        ctx["current_step_index"] = idx
        step = plan_steps[idx]

        # ... 同 execute_ops_stream_pipeline 的步骤执行逻辑 ...
        # 关键: 支持再次中断 (多轮 HITL)

    # 9. 闭环落库
    # ... 同 execute_ops_stream_pipeline 的持久化逻辑 ...
    yield {"type": PacketType.DONE, "content": {"entry_id": entry_id}}


# ==================================================================
# HITL: 挂起辅助方法
# ==================================================================

async def _suspend_execution(
    self, ctx, suspend_ctx, request_id, current_step_index,
    entry_id, start_time
):
    """持久化完整执行快照"""
    pending_data = {
        "request_id": request_id,
        "session_id": ctx["session_id"],
        "user_id": ctx["user_id"],
        "agent_id": ctx["agent_id"],
        "instance_id": ctx["instance_id"],
        "entry_id": entry_id,
        "suspend_reason": suspend_ctx.get("reason", ""),
        "user_prompt": suspend_ctx.get("sql_to_run", ""),
        "sql_to_run": suspend_ctx.get("sql_to_run", ""),
        "expected_fields": json.dumps(
            suspend_ctx.get("expected_fields", []), ensure_ascii=False
        ),
        "suspended_by_skill": suspend_ctx.get("suspended_by", "unknown"),
        "current_step_index": current_step_index,
        "completed_steps": json.dumps(
            ctx.get("execution_history", []), default=str, ensure_ascii=False
        ),
        "accumulated_results": json.dumps({
            "metric_results": ctx.get("metric_results", []),
            "monitor_results": ctx.get("monitor_results", []),
            "doc_results": ctx.get("doc_results", []),
        }, default=str, ensure_ascii=False),
        "pending_variables": json.dumps({
            k: v for k, v in ctx["variables"].items()
            if not k.startswith("_")
        }, default=str, ensure_ascii=False),
        "hitl_history": json.dumps(
            ctx.get("hitl_history", []), default=str, ensure_ascii=False
        ),
        "runtime_plan": json.dumps(
            ctx.get("runtime_plan"), default=str, ensure_ascii=False
        ),
        "status": "pending",
        "timeout_at": datetime.now(timezone.utc) + timedelta(minutes=30),
    }
    await self._pending_repo.create(pending_data)
    logger.info(
        f"[HITL Suspend] request_id={request_id} | "
        f"step={current_step_index} | 已持久化快照"
    )


def _rebuild_context_from_pending(self, pending: dict) -> OpsContextMemory:
    """从持久化快照重建 OpsContextMemory"""
    import json as _json

    runtime_plan = _json.loads(pending.get("runtime_plan", "{}"))
    variables = _json.loads(pending.get("pending_variables", "{}"))
    accumulated = _json.loads(pending.get("accumulated_results", "{}"))
    hitl_history = _json.loads(pending.get("hitl_history", "[]"))

    ctx: OpsContextMemory = {
        "trace_id": f"trace-resume-{uuid.uuid4().hex[:12]}",
        "user_id": pending["user_id"],
        "session_id": pending["session_id"],
        "agent_id": pending["agent_id"],
        "trigger_type": cast(Any, "manual"),
        "command_or_query": runtime_plan.get("inputs", {}).get("user_query", ""),
        "llm_model": runtime_plan.get("inputs", {}).get("model_name", ""),
        "embedding_model": "",
        "instance_id": pending["instance_id"],
        "db_type": "",
        "version_code": 0,
        "db_role": "primary",
        "environment": "dev",
        "monitor_type": "prometheus",
        "prometheus_instance_label": None,
        "alert_context": None,
        "runtime_plan": runtime_plan,
        "current_step_index": pending["current_step_index"],
        "current_execution": None,
        "execution_history": _json.loads(pending.get("completed_steps", "[]")),
        "approval_context": None,
        "variables": variables,
        "metric_results": accumulated.get("metric_results", []),
        "monitor_results": accumulated.get("monitor_results", []),
        "os_log_snapshots": [],
        "doc_results": accumulated.get("doc_results", []),
        "temp": {},
        # HITL
        "is_resuming": True,
        "hitl_history": hitl_history,
    }
    return ctx
```

### 4.3 API 层新增

```python
# api/schemas/ops_schema.py 新增

class OpsResumeRequest(BaseModel):
    """HITL 恢复执行请求体"""
    request_id: str = Field(
        ..., description="挂起请求 ID（来自 WAIT_FOR_USER 包的 request_id）"
    )
    user_data: dict[str, Any] | None = Field(
        None, description="用户回填的数据，key-value 形式"
    )
    user_note: str | None = Field(
        None, description="用户备注"
    )
    user_error: str | None = Field(
        None,
        description="用户执行 SQL 时的报错信息，如 ORA-00942: table or view does not exist"
    )


# api/routers/ops_router.py 新增

@router.post(
    "/chat/resume",
    summary="【HITL】提交用户采集的数据并恢复诊断",
    description="用户在收到 WAIT_FOR_USER 事件后，执行 SQL 并将结果通过此接口提交，Agent 从断点恢复分析。",
    response_class=StreamingResponse
)
async def resume_ops_chat(
    auth: UserAuth,
    request: OpsResumeRequest,
    background_tasks: BackgroundTasks
):
    return await ops_controller.resume_chat(
        request=request,
        background_tasks=background_tasks
    )
```

### 4.4 Prompt 模板新增

```python
# agent/prompt/default_prompt.py 新增 ops_sufficiency_check 模板

ops_sufficiency_check = """
你是一个 {db_type} 数据库诊断专家。评估现有证据是否足以定位根因。

## 用户问题
{query_text}

## 当前环境
- 数据库类型: {db_type}
- 环境: {environment}

## 已采集的证据
{evidence_summary}

## 历史 HITL 交互
{hitl_context}

## 评估规则
1. 如果 Prometheus + 诊断 SQL + SOP 手册已经构成完整证据链 → verdict: "sufficient"
2. 如果证据指向某个方向但缺少关键数据来确认 → verdict: "insufficient"
3. 当 verdict="insufficient" 时，你必须:
   - 生成一条用户可在 {db_type} 数据库执行的 SELECT SQL
   - SQL 必须包含行数限制（ROWNUM <= 20 / LIMIT 20 / TOP 20）
   - 使用通用系统视图（避免需要 DBA 权限的视图）
   - 告知用户期望返回哪些关键字段

## 输出格式（严格 JSON）
{{
  "verdict": "sufficient" 或 "insufficient",
  "reason": "用中文解释你做此判断的原因",
  "sql_to_run": "需要用户执行的完整 SQL（仅在 insufficient 时输出）",
  "expected_fields": ["关键字段1", "关键字段2"]
}}

请严格按 JSON 输出，不要输出其他内容。
"""
```

---

## 五、实施步骤

### Phase 0: 基础设施 (1-2 天)

| 步骤 | 文件 | 说明 |
|------|------|------|
| 0.1 | `docs/kbot_db_change_ddl_3.4_hitl.sql` | 创建 DDL 脚本 |
| 0.2 | `dao/entities/ops_pending.py` | 新增 `PendingRequestEntity` |
| 0.3 | `dao/repositories/ops_pending_repo.py` | 新增 `PendingRequestRepository` |
| 0.4 | `core/dictionary.py` | `PacketType` 追加 `WAIT_FOR_USER` |
| 0.5 | `agent/common/ops_context.py` | `OpsContextMemory` 新增 `is_resuming`, `hitl_history` |

### Phase 1: 核心引擎 (3-4 天)

| 步骤 | 文件 | 说明 |
|------|------|------|
| 1.1 | `agent/prompt/default_prompt.py` | 新增 `ops_sufficiency_check` 模板 |
| 1.2 | `core/config/prompt_config.py` | 注册新 prompt key |
| 1.3 | `skills/.../db_analysis_skill_core.py` | 增加 `_check_data_sufficiency()` + 恢复模式分支 |
| 1.4 | `agent/orchestrator/ops_orchestrator.py` | 步骤循环增加中断检测 + `resume_ops_stream_pipeline()` |
| 1.5 | 单元测试 | 测试中断-恢复循环 |

### Phase 2: API 与前端 (2-3 天)

| 步骤 | 文件 | 说明 |
|------|------|------|
| 2.1 | `api/schemas/ops_schema.py` | 新增 `OpsResumeRequest` |
| 2.2 | `api/routers/ops_router.py` | 新增 `POST /ops/chat/resume` |
| 2.3 | `api/controllers/ops_controller.py` | 新增 `resume_chat()` |
| 2.4 | 前端 SSE 消费者 | 处理 `wait_for_user` 事件 + 渲染输入表单 |
| 2.5 | 集成测试 | 端到端测试：请求 → 中断 → 提交 → 恢复 → 完成 |

### Phase 3: 鲁棒性 (1-2 天)

| 步骤 | 说明 |
|------|------|
| 3.1 | 超时检测：定时任务扫描 `timeout_at < now` 的 pending 记录 |
| 3.2 | 重复提交保护：同一 `request_id` 第二次提交返回 "已处理" |
| 3.3 | 并发控制：同一 session 同时只有一个活跃挂起 |
| 3.4 | 取消支持：`POST /ops/chat/cancel-pending` 让用户主动放弃等待 |

---

## 六、关键设计要点

### 6.1 hitl_history Timeline（不丢数据）

每次恢复时将本轮交互 **追加** 到 `ctx["hitl_history"]`，而不是覆盖 `user_response_data`。这样 LLM 在最终分析时看到完整的排查路径：

```
第1轮: Agent 要锁数据 → 用户提供了锁详情 (15 行)
第2轮: Agent 要执行计划 → 用户提供了 explain plan
第3轮: Agent 确认根因为索引缺失 → 输出 RCA
```

### 6.2 充分性检查的 Token 控制

- `max_tokens=500`：只输出 JSON 判定，不允许发散
- 规则前置：`metric_results` 和 `monitor_results` 都为空 → 直接返回 insufficient
- 最大轮次保护：`len(hitl_history) >= 5` → 强制进入分析

### 6.3 SQL 错误容错闭环

`OpsResumeRequest` 的 `user_error` 字段是关键。当用户执行 SQL 报错时：

```
用户 → POST /ops/chat/resume
  { user_error: "ORA-00942: table or view does not exist" }

DBAnalysisSkill 恢复时:
  → hitl_history 中看到 user_error
  → LLM 分析: "用户没有 v$lock 权限"
  → LLM 自动生成替代 SQL: "请尝试查询 dba_locks WHERE ..."
  → 再次 yield WAIT_FOR_USER

如果连续 3 次权限不足:
  → 降级: "抱歉，看来当前环境受限，基于已有数据给出初步分析..."
```

### 6.4 current_step_index = idx 的语义

挂起时记录 `current_step_index = idx`（当前 skill 的索引），恢复时 `for idx in range(current_step_index, len(plan_steps))` 从同一个 skill 重新开始。该 skill 通过 `ctx["is_resuming"]` 检测到自己被重入，走恢复分支而不是从头检查。

### 6.5 前端 UI 渲染

前端收到 `WAIT_FOR_USER` 事件后应渲染：

```
┌────────────────────────────────────────────┐
│ 🔍 Agent 需要你的协助                        │
│                                            │
│ 📋 原因: 排除资源瓶颈后，等待事件指向行锁...    │
│                                            │
│ 📝 请在目标数据库执行以下 SQL:               │
│ ┌──────────────────────────────────────┐   │
│ │ SELECT s.sid, s.serial#, ...        │   │
│ │ FROM v$session s                    │   │
│ │ WHERE ... AND ROWNUM <= 20          │   │
│ └──────────────────────────────────────┘   │
│ [📋 一键复制]                               │
│                                            │
│ 📊 请将查询结果粘贴到下方:                   │
│ ┌──────────────────────────────────────┐   │
│ │ (textarea / 文件上传 / CSV 粘贴)      │   │
│ └──────────────────────────────────────┘   │
│                                            │
│ ⚠️ 如果执行报错，请粘贴错误信息:             │
│ ┌──────────────────────────────────────┐   │
│ │ (error input)                       │   │
│ └──────────────────────────────────────┘   │
│                                            │
│ [提交并继续分析]    [放弃此诊断]             │
│                                            │
│ ⏰ 超时时间: 2026-07-02 11:15              │
└────────────────────────────────────────────┘
```

---

## 七、变更文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| **新增** | `docs/kbot_db_change_ddl_3.4_hitl.sql` | DDL 迁移脚本 |
| **新增** | `dao/entities/ops_pending.py` | PendingRequestEntity |
| **新增** | `dao/repositories/ops_pending_repo.py` | PendingRequestRepository |
| 修改 | `core/dictionary.py` | PacketType 追加 WAIT_FOR_USER |
| 修改 | `agent/common/ops_context.py` | OpsContextMemory 新增 2 个字段 |
| 修改 | `agent/orchestrator/ops_orchestrator.py` | 中断检测 + resume() + _suspend_execution() |
| 修改 | `skills/.../db_analysis_skill_core.py` | _check_data_sufficiency() + 恢复分支 + Timeline |
| 修改 | `agent/prompt/default_prompt.py` | 新增 ops_sufficiency_check 模板 |
| 修改 | `core/config/prompt_config.py` | 注册 ops_sufficiency_check key |
| 修改 | `api/schemas/ops_schema.py` | 新增 OpsResumeRequest |
| 修改 | `api/routers/ops_router.py` | 新增 POST /ops/chat/resume |
| 修改 | `api/controllers/ops_controller.py` | 新增 resume_chat() |
| 修改 | `kbot_md_conv_context` 表 | ALTER TABLE 新增 2 列 |

**不需要变更的**：
- Planner（OpsTaskPlanner）—— 不需要感知 HITL
- DBMetricSkill —— 保持现有两阶段逻辑不变
- SkillRuntime / SkillManager —— 通用框架不需要改动

---

## 八、测试场景

### 场景 1: 单轮交互

```
输入: "数据库 CPU 100%，帮我看看"
→ Prometheus: CPU 使用率 100%, IO 正常
→ 专家 SQL: db_top_cpu_sql() 找到了消耗 CPU 的 SQL
→ 充分性检查: sufficient
→ 直接输出 RCA: "SQL ID abc123 全表扫描导致 CPU 飙升，建议添加索引..."
```

### 场景 2: 单轮 HITL

```
输入: "数据库突然变慢"
→ Prometheus: CPU/IO 正常
→ 专家 SQL: active_session_wait → "enq: TX - row lock contention"
→ 充分性检查: insufficient → 需要锁详情
→ WAIT_FOR_USER: 请执行锁查询 SQL
→ 用户回填: { blocked_sessions: [...], blocking_sid: 102 }
→ 恢复: 注入用户数据
→ 输出 RCA: "SID 102 未提交事务持有排他锁，阻塞了 15 个会话，建议 kill SID 102"
```

### 场景 3: 多轮 HITL

```
输入: "数据库间歇性 hang"
→ Prometheus: CPU spikes at 10:00-10:05
→ 专家 SQL: 无明显异常
→ 充分性检查: insufficient → 要 AWR 报告
→ R1: 用户提供 AWR 片段
→ 充分性检查: insufficient → 要具体执行计划
→ R2: 用户提供 explain plan
→ 充分性检查: sufficient
→ 输出 RCA: "10:00 的 ETL 任务引发执行计划突变，建议固化 baseline"
```

### 场景 4: 用户 SQL 报错容错

```
→ WAIT_FOR_USER: SELECT ... FROM v$lock ...
→ 用户回填: user_error = "ORA-00942: table or view does not exist"
→ 恢复: LLM 识别权限不足
→ LLM 自动修正: "请尝试 SELECT ... FROM dba_locks WHERE ..."
→ WAIT_FOR_USER (第 2 轮)
→ 用户回填: 查询成功，数据正常
→ 输出 RCA
```
