"""交换机夹具保持真实指标维度、只读 SNMP 边界与原始计数语义。"""

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace


path = Path(__file__).resolve().parents[3] / "scripts/deployment/aiops_exadata_sim/switch_simulator.py"
spec = importlib.util.spec_from_file_location("exadata_switch_simulator", path)
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)
COMMUNITY = b"unit-test-private-community"


def packet(operation, names, *, version=1, community=COMMUNITY, repeat=4):
    bindings = b"".join(sim.tlv(0x30, sim.oid(name) + sim.tlv(5, b"")) for name in names)
    pdu = sim.integer(123) + sim.integer(0) + sim.integer(repeat if operation == 0xA5 else 0) + sim.tlv(0x30, bindings)
    return sim.tlv(0x30, sim.integer(version) + sim.tlv(4, community) + sim.tlv(operation, pdu))


def decoded(response):
    message = sim.fields(sim.read_tlv(response)[1])
    pdu = sim.fields(message[2][1])
    bindings = [sim.fields(raw) for tag, raw in sim.fields(pdu[3][1])]
    return int.from_bytes(pdu[1][1], "big"), [(sim.decode_oid(row[0][1]), row[1]) for row in bindings]


class SwitchSimulatorTests(unittest.TestCase):
    def test_native_registration_uses_access_point_and_exact_credential_option(self):
        registration = importlib.util.spec_from_file_location("switch_registration", path.with_name("register_switch.py"))
        module = importlib.util.module_from_spec(registration)
        registration.loader.exec_module(module)
        community = "ab" * 24
        arguments = module.arguments_for(community)
        self.assertIn("-type=oracle_si_netswitch", arguments)
        self.assertIn("-access_point_type=oracle_si_switch_cisco_ios_snmp", arguments)
        self.assertIn("-subseparator=properties=->", arguments)
        self.assertIn("-monitoring_cred=snmp_v1v2_v3;oracle_si_switch_cisco_ios_snmp;SNMPV1Creds;COMMUNITY:" + community, arguments)
        self.assertFalse(any(arg.startswith("-monitoring_creds=") for arg in arguments))
        self.assertNotIn("-force", arguments)
        self.assertFalse(any(arg.startswith("-input_file=") for arg in arguments))
        self.assertIn("SNMPVersion->v2c", " ".join(arguments))
        command_file = module.command_file_for(community)
        self.assertEqual(command_file.count("\n"), 1)
        self.assertIn('-display_name="SIM - Exadata Switch 01"', command_file)
        with self.assertRaises(ValueError):
            module.command_file_for('bad"\nadd_target')

    def test_deployment_creates_private_community_and_unpublished_nonroot_container(self):
        deployment = importlib.util.spec_from_file_location("switch_deployment", path.with_name("deploy_switch.py"))
        module = importlib.util.module_from_spec(deployment)
        deployment.loader.exec_module(module)
        def output(*arguments):
            if arguments[1:3] == ("network", "inspect"):
                return '[{"Containers": {}}]'
            return "container-test-id\n" if arguments[1] == "run" else ""
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            with patch.object(module, "STATE", state), patch.object(module.os, "geteuid", return_value=0), \
                    patch.object(module.os, "chown"), patch.object(module, "checked", side_effect=output) as calls, \
                    patch("builtins.print"):
                module.main()
                run = calls.call_args.args
                self.assertNotIn("-p", run)
                self.assertNotIn("--publish", run)
                self.assertIn("--read-only", run)
                self.assertIn("54321:54321", run)
                secret = (state / "snmp-community").read_text().strip()
                self.assertNotIn(secret, " ".join(run))
                self.assertEqual((state / "snmp-community").stat().st_mode & 0o777, 0o400)
                self.assertEqual(json.loads((state / "created.json").read_text())["published_ports"], [])
                with self.assertRaises(SystemExit):
                    module.main()

    def test_registration_uses_native_private_argfile_and_keeps_unverified_receipt(self):
        registration = importlib.util.spec_from_file_location("switch_registration_private", path.with_name("register_switch.py"))
        module = importlib.util.module_from_spec(registration)
        registration.loader.exec_module(module)
        community = "ab" * 24
        inspection = [{"State": {"Running": True}, "HostConfig": {"PortBindings": {}},
            "Config": {"Labels": {"aiops.simulation": "exadata"}},
            "NetworkSettings": {"Networks": {"aiops-oem-lab": {"IPAddress": "172.22.0.9"}}}}]
        def native(*arguments):
            output = ""
            if arguments[0] == "get_plugin_deployment_status":
                output = "Destination : Management Agent - aiops-exadata-monitor.lab:3872\nStatus               : Success"
            elif arguments[0] == "argfile":
                output = "Target added successfully"
            return SimpleNamespace(returncode=0, stdout=output, stderr="")
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            (state / "created.json").write_text(json.dumps({"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
            (state / "snmp-community").write_text(community)
            with patch.object(module, "STATE", state), patch.object(module.os, "geteuid", return_value=0), \
                    patch.object(module, "cli", side_effect=native) as calls, \
                    patch.object(module.subprocess, "check_output", return_value=json.dumps(inspection)), \
                    patch.object(module.subprocess, "run") as run, patch("builtins.print"):
                module.main()
                self.assertEqual(calls.call_args.args, ("argfile", module.COMMAND_FILE))
                self.assertIn(community, run.call_args_list[0].kwargs["input"])
                for call in run.call_args_list:
                    self.assertNotIn(community, " ".join(call.args[0]))
                receipt = json.loads((state / "target-result.json").read_text())
                self.assertTrue(receipt["command_accepted"])
                self.assertFalse(receipt["inventory_verified"])
                with self.assertRaises(SystemExit):
                    module.main()

    def test_oid_and_ber_round_trip_with_multibyte_subidentifier(self):
        for name in ("0.0", "1.3.6.1.4.1.9.9.813.1.2.1.2.1.7", "2.999.16384"):
            self.assertEqual(sim.decode_oid(sim.read_tlv(sim.oid(name))[1]), tuple(map(int, name.split("."))))
        raw = b"x" * 256
        self.assertEqual(sim.read_tlv(sim.tlv(4, raw)), (4, raw, 260))

    def test_native_identity_is_marked_sim(self):
        _, rows = decoded(sim.response(packet(0xA0, ["1.3.6.1.2.1.1.5.0"]), COMMUNITY))
        self.assertEqual(rows[0][1], (4, b"AIOPS_SIM_EXADATA_SWITCH01"))

    def test_native_entity_identity_has_sim_serial_and_chassis_class(self):
        status, rows = decoded(sim.response(packet(0xA0, [
            "1.3.6.1.2.1.47.1.1.1.1.5.1", "1.3.6.1.2.1.47.1.1.1.1.11.1"]), COMMUNITY))
        self.assertEqual(status, 0)
        self.assertEqual(rows[0][1], (2, b"\x03"))
        self.assertEqual(rows[1][1], (4, b"SIM-EXADATA-SWITCH-0001"))

    def test_vtp_has_only_one_explicit_sim_ethernet_vlan(self):
        status, rows = decoded(sim.response(packet(0xA0, [
            "1.3.6.1.4.1.9.9.46.1.3.1.1.2.1.1",
            "1.3.6.1.4.1.9.9.46.1.3.1.1.3.1.1",
            "1.3.6.1.4.1.9.9.46.1.3.1.1.4.1.1",
            "1.3.6.1.4.1.9.9.46.1.3.1.1.17.1.1",
            "1.3.6.1.4.1.9.9.46.1.3.1.1.7.1.1"]), COMMUNITY))
        self.assertEqual(status, 0)
        self.assertEqual([value for _, value in rows],
                         [(2, b"\x01"), (2, b"\x01"), (4, b"SIM_VLAN1"), (4, b"\x00"), (0x80, b"")])

    def test_wrong_community_unsupported_versions_and_oversize_are_rejected(self):
        for request in (packet(0xA0, ["1.3.6.1.2.1.1.5.0"], community=b"public"),
                        packet(0xA0, [], version=3), b"x" * (sim.MAX_PACKET + 1)):
            self.assertIsNone(sim.response(request, COMMUNITY))

    def test_vlan_context_requires_same_secret_and_only_declared_vlan(self):
        for wrong in (b"public@1", COMMUNITY + b"@2", COMMUNITY + b"@01", COMMUNITY + b"@1@1"):
            self.assertIsNone(sim.response(packet(0xA0, ["1.3.6.1.2.1.1.5.0"], community=wrong), COMMUNITY))
        response = sim.response(packet(0xA0, ["1.3.6.1.2.1.1.5.0"], community=COMMUNITY+b"@1"), COMMUNITY)
        self.assertEqual(sim.fields(sim.read_tlv(response)[1])[1], (4, COMMUNITY+b"@1"))
        # FDB 没有远端邻居，不捏造 MAC 地址或拓扑。
        _, rows = decoded(sim.response(packet(0xA0, ["1.3.6.1.2.1.17.4.3.1.1.1"], community=COMMUNITY+b"@1"), COMMUNITY))
        self.assertEqual(rows[0][1], (0x80, b""))

    def test_set_cannot_mutate_mib(self):
        before = sim.mib(sim.EPOCH + 100)
        status, _ = decoded(sim.response(packet(0xA3, ["1.3.6.1.2.1.2.2.1.7.1"]), COMMUNITY, now=sim.EPOCH + 100))
        self.assertEqual(status, 6)
        self.assertEqual(sim.mib(sim.EPOCH + 100), before)

    def test_unknown_oid_does_not_fabricate_healthy_zero(self):
        status, rows = decoded(sim.response(packet(0xA0, ["1.3.6.1.4.1.99999.0"]), COMMUNITY))
        self.assertEqual(status, 0)
        self.assertEqual(rows[0][1], (0x80, b""))
        status, _ = decoded(sim.response(packet(0xA0, ["1.3.6.1.4.1.99999.0"], version=0), COMMUNITY))
        self.assertEqual(status, 2)

    def test_next_and_bulk_are_ordered_and_bounded(self):
        status, rows = decoded(sim.response(packet(0xA5, ["1.3.6.1.2.1.2.2.1.2"], repeat=9999), COMMUNITY))
        self.assertEqual(status, 0)
        self.assertEqual(len(rows), sim.MAX_REPETITIONS)
        self.assertEqual([key for key, value in rows], sorted(key for key, value in rows))
        self.assertEqual(rows[0][1], (4, b"Ethernet1/1"))
        self.assertEqual(rows[1][1], (4, b"Ethernet1/2"))

    def test_counters_grow_but_rates_and_deltas_are_not_fabricated(self):
        before, after = sim.mib(sim.EPOCH + 100), sim.mib(sim.EPOCH + 160)
        for name in ("1.3.6.1.2.1.31.1.1.1.6.1", "1.3.6.1.4.1.9.9.813.1.2.1.2.1.7"):
            key = tuple(map(int, name.split(".")))
            tag, old, _ = sim.read_tlv(before[key])
            _, new, _ = sim.read_tlv(after[key])
            self.assertEqual(tag, 0x46)
            self.assertGreater(int.from_bytes(new, "big"), int.from_bytes(old, "big"))

    def test_pfc_two_dimensional_priority_instances_are_complete(self):
        data = sim.mib(sim.EPOCH + 100)
        prefix = tuple(map(int, "1.3.6.1.4.1.9.9.813.1.2.1.2".split(".")))
        self.assertEqual({key[len(prefix):] for key in data if key[:len(prefix)] == prefix},
                         {(port, priority) for port in sim.PORTS for priority in range(8)})

    def test_native_pfc_walk_can_leave_last_table_without_end_of_mib_exception(self):
        status, rows = decoded(sim.response(packet(0xA1, ["1.3.6.1.4.1.9.9.813.1.2.1.3.2.7"]), COMMUNITY))
        self.assertEqual(status, 0)
        self.assertEqual(rows[0][0], tuple(map(int, "1.3.6.1.6.3.10.2.1.1.0".split("."))))
        self.assertEqual(rows[0][1][0], 4)

    def test_truncation_indefinite_length_and_duplicate_trailing_message_fail(self):
        valid = packet(0xA0, ["1.3.6.1.2.1.1.5.0"])
        for request in (valid[:-1], valid + b"x", b"\x30\x80\x00\x00", b"\x30\x84\xff\xff\xff\xff"):
            with self.assertRaises(ValueError):
                sim.response(request, COMMUNITY)


if __name__ == "__main__":
    unittest.main()
