"""非最终模型阶段必须共享工作期限，为回答保留配置预算。"""

from datetime import datetime, timedelta


def work_deadline(deadline: datetime | None, *, answer_reserve_seconds: int) -> datetime | None:
    return deadline - timedelta(seconds=answer_reserve_seconds) if deadline is not None else None
