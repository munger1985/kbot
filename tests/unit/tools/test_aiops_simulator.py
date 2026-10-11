"""独立AIOps流量模拟程序的安全合同测试。"""

from __future__ import annotations

import asyncio
import os
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_ROOT = REPO_ROOT / "tools" / "aiops_simulator"
sys.path.insert(0, str(TOOL_ROOT))

from aiops_simulator.config import load_config, parse_schedule  # noqa: E402
from aiops_simulator.faults.base import (  # noqa: E402
    CONNECTION_SURGE_SIZE,
    FAULT_TTL_SECONDS,
    SUPPORTED_FAULTS,
    normalize_database_name,
    validate_fault_type,
)
from aiops_simulator.faults.state import (  # noqa: E402
    print_fault_status,
    read_state,
    write_state,
)
from aiops_simulator.faults.runner import run_fault  # noqa: E402
from aiops_simulator.modes import load_mode  # noqa: E402
from aiops_simulator.modes.base import OperationKind  # noqa: E402


class AIOpsSimulatorTest(unittest.TestCase):
    def _config_file(self, mode: int = 0o600) -> Path:
        temporary = tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8")
        content = (TOOL_ROOT / "simulator.example.ini").read_text(encoding="utf-8")
        temporary.write(content.replace("CHANGE_ME", "unit-test-password"))
        temporary.close()
        path = Path(temporary.name)
        os.chmod(path, mode)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_example_config_uses_real_time_and_three_distinct_profiles(self) -> None:
        config = load_config(self._config_file())

        self.assertEqual("daily", config.mode)
        self.assertEqual("Asia/Shanghai", config.timezone_name)
        self.assertEqual(3, len([item for item in config.databases if item.enabled]))
        rates = {
            item.engine: item.traffic.rate_at(9 * 60, weekday=0)
            for item in config.databases
        }
        self.assertEqual(
            {"oracle": 3.0, "postgresql": 3.5, "mysql": 3.0}, rates
        )

    def test_schedule_must_cover_complete_day_without_gaps(self) -> None:
        with self.assertRaisesRegex(ValueError, "连续"):
            parse_schedule("00:00-06:00=1, 07:00-24:00=1")
        with self.assertRaisesRegex(ValueError, "24:00"):
            parse_schedule("00:00-23:00=1")

    def test_weekend_rate_uses_database_specific_multiplier(self) -> None:
        config = load_config(self._config_file())
        oracle = next(item for item in config.databases if item.engine == "oracle")
        weekday = oracle.traffic.rate_at(9 * 60, weekday=0)
        weekend = oracle.traffic.rate_at(9 * 60, weekday=6)

        self.assertAlmostEqual(weekday * 0.35, weekend)

    def test_daily_mode_keeps_exact_eight_to_two_ratio_per_window(self) -> None:
        mode = load_mode("daily")
        mix = mode.operation_mix(random.Random(7))

        for _ in range(10):
            window = [mix.next() for _ in range(100)]
            self.assertEqual(80, window.count(OperationKind.READ))
            self.assertEqual(20, window.count(OperationKind.WRITE))

    def test_daily_mode_uses_wall_clock_profile(self) -> None:
        config = load_config(self._config_file())
        postgresql = next(
            item for item in config.databases if item.engine == "postgresql"
        )
        mode = load_mode("daily")
        morning = datetime(2026, 10, 12, 9, 0, tzinfo=config.timezone)
        night = datetime(2026, 10, 12, 2, 0, tzinfo=config.timezone)

        self.assertEqual(3.5, mode.rate_at(postgresql, morning))
        self.assertEqual(0.8, mode.rate_at(postgresql, night))

    def test_unimplemented_risky_modes_are_rejected(self) -> None:
        for name in ("abnormal", "fault"):
            with self.assertRaisesRegex(ValueError, "未实现"):
                load_mode(name)

    def test_secret_config_rejects_group_or_other_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "权限"):
            load_config(self._config_file(mode=0o640))

    def test_daily_adapter_sources_exclude_destructive_database_actions(self) -> None:
        adapter_root = TOOL_ROOT / "aiops_simulator" / "adapters"
        source = "\n".join(
            path.read_text(encoding="utf-8").lower()
            for path in adapter_root.glob("*.py")
        )
        for forbidden in (
            "drop table",
            "truncate table",
            "alter system",
            "shutdown immediate",
            "kill connection",
        ):
            self.assertNotIn(forbidden, source)

    def test_manager_is_plain_python_and_independent_from_kbot_services(self) -> None:
        manager = (TOOL_ROOT / "manage").read_text(encoding="utf-8")

        self.assertFalse((TOOL_ROOT / "Dockerfile").exists())
        self.assertFalse((TOOL_ROOT / "compose.yaml").exists())
        self.assertIn("-m aiops_simulator run", manager)
        self.assertIn("simulator.pid", manager)
        self.assertNotIn("systemctl", manager)
        self.assertNotIn("docker", manager.lower())
        self.assertNotIn("aiops-stack", manager)
        self.assertNotIn("services/", manager)

    def test_fault_command_contract_has_only_database_action_and_type(self) -> None:
        manager = (TOOL_ROOT / "manage").read_text(encoding="utf-8")
        example_config = (TOOL_ROOT / "simulator.example.ini").read_text(
            encoding="utf-8"
        )

        self.assertIn("fault {oracle|postgres|mysql} start", manager)
        self.assertIn("fault {oracle|postgres|mysql} {stop|status}", manager)
        self.assertIn("--fault-type", manager)
        self.assertNotIn("--ttl", manager)
        self.assertNotIn("--connections", manager)
        self.assertNotIn("fault.oracle", manager)
        self.assertNotIn("[fault.", example_config)

    def test_fault_types_and_safety_limits_are_fixed_in_code(self) -> None:
        self.assertEqual(600, FAULT_TTL_SECONDS)
        self.assertEqual(12, CONNECTION_SURGE_SIZE)
        self.assertEqual(
            {
                "slow_query",
                "blocking_lock",
                "deadlock",
                "long_transaction",
                "connection_surge",
                "temp_pressure",
            },
            set(SUPPORTED_FAULTS),
        )
        self.assertEqual("postgresql", normalize_database_name("postgres"))
        self.assertEqual("deadlock", validate_fault_type("deadlock"))
        with self.assertRaisesRegex(ValueError, "故障类型"):
            validate_fault_type("shutdown")

    def test_fault_status_does_not_require_database_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [str(TOOL_ROOT / "manage"), "fault", "oracle", "status"],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "AIOPS_SIMULATOR_RUNTIME_DIR": temporary,
                },
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("当前没有故障模拟记录", result.stdout)

    def test_fault_state_contains_no_connection_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            write_state(
                path,
                database="oracle",
                fault_type="blocking_lock",
                run_id="test-run",
                pid=123,
                ttl_seconds=600,
                status="active",
            )
            state = read_state(path)
            output = StringIO()
            with redirect_stdout(output):
                print_fault_status(path, "oracle", running=True)

        self.assertIsNotNone(state)
        self.assertEqual("active", state["status"])
        self.assertNotIn("password", state)
        self.assertNotIn("host", state)
        self.assertIn("状态=运行中", output.getvalue())

    def test_fault_sources_exclude_destructive_database_control(self) -> None:
        fault_root = TOOL_ROOT / "aiops_simulator" / "faults"
        source = "\n".join(
            path.read_text(encoding="utf-8").lower()
            for path in fault_root.glob("*.py")
        )
        for forbidden in (
            "shutdown immediate",
            "alter system",
            "drop table",
            "truncate table",
            "kill session",
            "pg_terminate_backend",
        ):
            self.assertNotIn(forbidden, source)

    def test_fault_automatically_expires_and_closes_adapter(self) -> None:
        async def exercise(state_path: Path) -> tuple[dict[str, object], bool]:
            class FakeAdapter:
                def __init__(self) -> None:
                    self.ready_event = asyncio.Event()
                    self.closed = False

                async def run(self) -> None:
                    self.ready_event.set()
                    await asyncio.Event().wait()

                async def close(self) -> None:
                    self.closed = True

            adapter = FakeAdapter()
            with patch(
                "aiops_simulator.faults.runner._create_adapter",
                return_value=adapter,
            ), patch(
                "aiops_simulator.faults.runner.FAULT_TTL_SECONDS",
                0.01,
            ):
                await run_fault(
                    object(), "oracle", "slow_query", state_path  # type: ignore[arg-type]
                )
            state = read_state(state_path)
            assert state is not None
            return state, adapter.closed

        with tempfile.TemporaryDirectory() as temporary:
            state, closed = asyncio.run(
                exercise(Path(temporary) / "state.json")
            )

        self.assertEqual("expired", state["status"])
        self.assertTrue(closed)


if __name__ == "__main__":
    unittest.main()
