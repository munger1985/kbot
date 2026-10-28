"""未来观察复用既有只读工具；等待由持久化 available_at 承担，不在 Worker 内 sleep。"""

from datetime import datetime, timezone

from aiops_agent.tools.observation_series import (
    assemble_observation_series, observation_gap, point_error, replay_baseline, replay_point,
)


class FutureObservationPointHandler:
    def __init__(self, *, direct_handler, dynamic_handler):
        self._direct = direct_handler
        self._dynamic = dynamic_handler

    async def execute(self, context):
        info = context.plan_snapshot["future_observation"]
        base = replay_baseline(context.input_artifacts, info)
        error = context.plan_snapshot.get("future_observation_access_error")
        if error:
            return observation_gap(base, str(error), "未来观察的来源、目标或当前授权已变化；没有继续连接数据库")
        due = datetime.fromisoformat(context.plan_snapshot["future_observation_due_at"])
        started = datetime.now(timezone.utc)
        if due.tzinfo is None or started < due or (started - due).total_seconds() > 30:
            return observation_gap(base, "FOLLOWUP_TIME_MISSED", "未来观察未在允许的时点范围内领取；不将迟到查询冒充原计划采样")
        previous = [replay_point(item) for item in context.input_artifacts
            if item.get("artifact_key") != "future:baseline" and item.get("schema_version") == "DBA_TOOL_RESULT.v1"]
        if any(item.status != "SUCCEEDED" or len(item.tool_outcomes) != 1
            or item.tool_outcomes[0].status != "SUCCEEDED" or item.tool_outcomes[0].observation is None
            or item.tool_outcomes[0].observation.truncated for item in previous):
            return observation_gap(base, "FOLLOWUP_PREVIOUS_GAP", "前次未来观察出现缺口，停止该组后续查询，保留已经完成的原始时点")
        prior = [base.tool_outcomes[0].observation,
            *(item.tool_outcomes[0].observation for item in previous)]
        prior.sort(key=lambda item: item.captured_at)
        handler = self._direct if info["kind"] == "DIRECT" else self._dynamic
        result = await handler.execute(context)
        error = point_error(result, base=base, target_id=context.target_id, previous=prior,
            now=datetime.now(timezone.utc), interval=info["requested"]["interval_seconds"], earliest=due)
        if error:
            return observation_gap(base, error, "未来观察返回旧时点或身份不匹配的结果，已停止后续查询")
        return result


class FutureObservationSeriesHandler:
    async def execute(self, context):
        return assemble_observation_series(
            artifacts=context.input_artifacts, info=context.plan_snapshot["future_observation"],
            target_id=context.target_id, now=datetime.now(timezone.utc),
        )
