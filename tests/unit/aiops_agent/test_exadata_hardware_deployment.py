"""硬件模拟器更新保留隔离、绑定 inode 和漂移保护。"""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest


DIRECTORY = Path(__file__).resolve().parents[3] / "scripts/deployment/aiops_exadata_sim"


def module(name):
    spec = importlib.util.spec_from_file_location("hardware_" + name, DIRECTORY / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def inspection(container, target, destination):
    return {"Name": "/" + container, "State": {"Running": True},
        "HostConfig": {"ReadonlyRootfs": True, "Privileged": False, "CapAdd": None, "PortBindings": {}},
        "Config": {"User": "54321:54321"},
        "Mounts": [{"Type": "bind", "Source": str(target), "Destination": destination, "RW": False}]}


@pytest.mark.parametrize("field,value", [("ReadonlyRootfs", False), ("Privileged", True),
    ("CapAdd", ["NET_ADMIN"]), ("PortBindings", {"80/tcp": [{"HostPort": "80"}]})])
def test_update_rejects_weakened_isolation(field, value):
    deployment = module("update_hardware")
    info = inspection("sim", Path("/fixed.py"), "/sim/fixed.py")
    info["HostConfig"][field] = value
    with pytest.raises(RuntimeError):
        deployment.validate(info, "sim", Path("/fixed.py"), "/sim/fixed.py")


def test_hardware_update_previews_then_backs_up_and_preserves_inode(tmp_path):
    deployment = module("update_hardware")
    sources = tmp_path / "payload"
    sources.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    files, expected, before = {}, {}, {}
    for key, name in (("switch", "switch_simulator.py"), ("pdu", "pdu_simulator.py"), ("pdu_codec", "switch_simulator.py")):
        target = tmp_path / (key + ".py")
        target.write_bytes(b"old = True\n")
        (sources / name).write_bytes(b"new = True\n")
        files[key] = ("sim-" + key, target, "/sim/" + name, name)
        expected[key] = deployment.digest(target.read_bytes())
        before[key] = target.stat().st_ino
    def output(command, **kwargs):
        if command[1] == "inspect":
            info = next(v for v in files.values() if v[0] == command[2])
            return json.dumps([inspection(*info[:3])])
        return deployment.digest(b"new = True\n") + "  file\n"
    with patch.object(deployment, "ROOT", tmp_path), patch.object(deployment, "FILES", files), \
            patch.object(deployment.os, "geteuid", return_value=0), \
            patch.object(deployment.subprocess, "check_output", side_effect=output), \
            patch.object(deployment.subprocess, "run") as restart:
        assert not deployment.update(sources, expected)["applied"]
        restart.assert_not_called()
        assert all(v[1].read_bytes() == b"old = True\n" for v in files.values())
        receipt = deployment.update(sources, expected, apply=True)
        assert receipt["applied"] and len(restart.call_args_list) == 3
        for key, (_, target, _, _) in files.items():
            assert target.stat().st_ino == before[key]
            assert target.read_bytes() == b"new = True\n"
            assert (Path(receipt["backup"]) / (key + ".py")).read_bytes() == b"old = True\n"
        with pytest.raises(RuntimeError, match="漂移"):
            deployment.update(sources, expected, apply=True)


def test_collection_tool_has_only_fixed_sim_targets_and_native_preview():
    deployment = module("enable_hardware_telemetry")
    for item in deployment.SCHEDULES:
        args = deployment.arguments(item, True)
        assert args[0] == "modify_collection_schedule" and args[-1] == "-preview=Y"
        assert "AIOPS_SIM_EXADATA_" in item[1] and "-force" not in args
        assert deployment.arguments(item, False)[-1] == "-preview=N"
