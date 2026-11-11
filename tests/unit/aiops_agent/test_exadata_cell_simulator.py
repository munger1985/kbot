"""独立 Cell 模拟器的原生字段、只读边界与时间窗口合同。"""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET


path = Path(__file__).resolve().parents[3] / "scripts/deployment/aiops_exadata_sim/cell_simulator.py"
spec = importlib.util.spec_from_file_location("exadata_cell_simulator", path)
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


class CellSimulatorTests(unittest.TestCase):
    def test_native_status_field_order(self):
        root = ET.fromstring(sim.render("list cell attributes cellsrvStatus, msStatus, rsStatus"))
        self.assertEqual([(item.tag, item.text) for item in root.find("cell")],
                         [("cellsrvStatus", "running"), ("msStatus", "running"), ("rsStatus", "running")])

    def test_current_cpu_memory_and_identity(self):
        root = ET.fromstring(sim.render("list metriccurrent where objectType='CELL'", now=1800000000))
        self.assertEqual(root.find("context").get("cell"), "sim-cell01")
        metrics = {item.findtext("name"): item for item in root.findall("metriccurrent")}
        self.assertIn("CL_CPUT", metrics)
        self.assertIn("CL_MEMUT", metrics)
        self.assertEqual(metrics["CL_CPUT"].findtext("objectType"), "CELL")
        self.assertEqual(list(metrics["CL_CPUT"])[-1].tag, "objectType")
        self.assertGreater(float(metrics["CL_CPUT"].findtext("metricValue")), 0)

    def test_history_respects_names_window_and_millisecond_order(self):
        command = ("list metrichistory CL_CPUT, CL_MEMUT where objectType != 'GRIDDISK' and "
                   "collectionTime>'2026-11-11T09:00:00+00:00' and "
                   "collectionTime<'2026-11-11T09:05:01+00:00' attributes name, metricValue")
        rows = ET.fromstring(sim.render(command)).findall("metrichistory")
        self.assertEqual(len(rows), 10)
        self.assertEqual({row.findtext("name") for row in rows}, {"CL_CPUT", "CL_MEMUT"})
        self.assertTrue(all(int(row.findtext("collectionTime")) > 10 ** 12 for row in rows))
        self.assertEqual([item.tag for item in rows[0]][-2:], ["collectionTime", "objectType"])

    def test_mutations_unknown_objects_attributes_and_unbounded_queries_fail(self):
        for command in ("alter cell shutdown services all", "list cell; alter cell shutdown services all",
                        "list gridbogus", "list cell attributes unknown", "list metrichistory CL_CPUT",
                        "list celldisk attributes unknown", "list cell detail;" * 9):
            with self.subTest(command=command):
                self.assertEqual(ET.fromstring(sim.render(command)).tag, "fault")

    def test_disk_types_remain_distinct(self):
        root = ET.fromstring(sim.render("list celldisk attributes name,diskType,size"))
        self.assertEqual([row.findtext("diskType") for row in root], [None, "HardDisk", "FlashDisk"])

    def test_forced_ssh_command_rejects_shell_and_mutations(self):
        entrypoint = path.with_name("cellcli_entrypoint.py")
        for command in ("", "sh", "cat /etc/passwd", "cellcli -xml -e 'list cell'; id",
                        "cellcli -xml -e 'alter cell shutdown services all'", "cellcli -xml -e '"):
            with self.subTest(command=command):
                result = subprocess.run([sys.executable, "-B", str(entrypoint), "--command"],
                    env={"SSH_ORIGINAL_COMMAND": command}, capture_output=True, check=True)
                self.assertEqual(ET.fromstring(result.stdout).tag, "fault")

    def test_forced_ssh_command_accepts_readonly_cellcli(self):
        result = subprocess.run([sys.executable, "-B", str(path.with_name("cellcli_entrypoint.py")), "--command"],
            env={"SSH_ORIGINAL_COMMAND": "cellcli -xml -e 'list cell attributes cellsrvStatus'"},
            capture_output=True, check=True)
        self.assertEqual(ET.fromstring(result.stdout).findtext("cell/cellsrvStatus"), "running")

    def test_native_registration_preserves_grid_hierarchy(self):
        registration = importlib.util.spec_from_file_location("sim_registration", path.with_name("register_targets.py"))
        module = importlib.util.module_from_spec(registration)
        registration.loader.exec_module(module)
        self.assertIn("SupportExacli:0", " ".join(module.arguments_for("cell")))
        self.assertIn("-add_members=AIOPS_SIM_EXADATA_CELL01:oracle_exadata", module.arguments_for("grid"))
        self.assertIn("-add_members=AIOPS_SIM_EXADATA_GRID01:oracle_exadata_grid", module.arguments_for("system"))
        self.assertTrue(module.native_success(SimpleNamespace(returncode=0, stdout="System created successfully")))
        for code, output in ((0, "Warning: member cannot be added"), (1, ""), (0, "Error: invalid member")):
            self.assertFalse(module.native_success(SimpleNamespace(returncode=code, stdout=output)))


if __name__ == "__main__":
    unittest.main()
