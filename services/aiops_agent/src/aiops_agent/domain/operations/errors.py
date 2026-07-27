"""确定性运行内核可持久化的稳定错误目录。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeErrorPolicy:
    retryable: bool
    safe_message: str


ERROR_CATALOG: dict[str, RuntimeErrorPolicy] = {
    "HANDLER_NOT_FOUND": RuntimeErrorPolicy(False, "任务处理器不可用"),
    "INPUT_SCHEMA_INVALID": RuntimeErrorPolicy(False, "任务输入格式无效"),
    "OUTPUT_SCHEMA_INVALID": RuntimeErrorPolicy(False, "任务输出格式无效"),
    "HANDLER_TIMEOUT": RuntimeErrorPolicy(True, "任务执行超时"),
    "HANDLER_RETRYABLE_FAILURE": RuntimeErrorPolicy(
        True, "任务执行暂时失败"
    ),
    "HANDLER_TERMINAL_FAILURE": RuntimeErrorPolicy(
        False, "任务执行失败"
    ),
    "WORKER_LEASE_EXPIRED": RuntimeErrorPolicy(
        True, "任务执行租约已过期"
    ),
}
