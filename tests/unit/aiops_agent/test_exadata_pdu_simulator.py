"""PDU 管理与功率协议明确为 SIM，不把遥测当作硬件健康。"""

from http.client import HTTPConnection
from http.server import HTTPServer
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3] / "scripts/deployment/aiops_exadata_sim"


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"switch_simulator": codec} if name == "pdu_simulator" else {}):
        spec.loader.exec_module(result)
    return result


codec = module("switch_simulator")
sim = module("pdu_simulator")
COMMUNITY = b"unit-test-private-community"


def packet(operation, name, community=COMMUNITY):
    bindings = codec.tlv(0x30, codec.oid(name) + codec.tlv(5, b""))
    pdu = codec.integer(1) + codec.integer(0) + codec.integer(0) + codec.tlv(0x30, bindings)
    return codec.tlv(0x30, codec.integer(1) + codec.tlv(4, community) + codec.tlv(operation, pdu))


class PduSimulatorTests(unittest.TestCase):
    def test_identity_is_explicit_sim_with_native_protocol_and_power_telemetry(self):
        data = sim.mib()
        self.assertIn(b"AIOPS_SIM", data[(1, 3, 6, 1, 2, 1, 1, 5, 0)])
        self.assertEqual(data[(1, 3, 6, 1, 2, 1, 1, 2, 0)], codec.oid("1.3.6.1.4.1.2769.1.4"))
        self.assertIn(b"SIM", data[(1, 3, 6, 1, 4, 1, 2769, 1, 4, 1, 11, 0)])
        before, after = sim.mib(sim.STARTED), sim.mib(sim.STARTED + 3600)
        energy = (1, 3, 6, 1, 4, 1, 2769, 1, 4, 5, 1, 5, 0)
        self.assertEqual(int.from_bytes(codec.read_tlv(before[energy])[1], "big"), 13046)
        self.assertGreater(int.from_bytes(codec.read_tlv(after[energy])[1], "big"), 13046)

    def test_snmp_rejects_wrong_community_and_set_is_read_only(self):
        self.assertIsNone(sim.response(packet(0xA0, "1.3.6.1.2.1.1.5.0", b"public"), COMMUNITY))
        raw = sim.response(packet(0xA3, "1.3.6.1.2.1.1.5.0"), COMMUNITY)
        pdu = codec.fields(codec.fields(codec.read_tlv(raw)[1])[2][1])
        self.assertEqual(int.from_bytes(pdu[1][1], "big"), 6)

    def test_unknown_hardware_oid_is_missing_not_healthy_zero(self):
        raw = sim.response(packet(0xA0, "1.3.6.1.4.1.99999.0"), COMMUNITY)
        pdu = codec.fields(codec.fields(codec.read_tlv(raw)[1])[2][1])
        binding = codec.fields(codec.fields(pdu[3][1])[0][1])
        self.assertEqual(binding[1], (0x80, b""))

    def test_http_is_read_only_and_does_not_claim_power_health(self):
        with HTTPServer(("127.0.0.1", 0), sim.ManagementHandler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                client = HTTPConnection(*server.server_address, timeout=3)
                client.request("GET", "/")
                response = client.getresponse()
                self.assertEqual(response.status, 200)
                body = json.loads(response.read())
                self.assertTrue(body["simulation"])
                self.assertIsNone(body["power_health"])
                for method, path, status in (("POST", "/", 501), ("PUT", "/", 501), ("GET", "/power", 404)):
                    client.request(method, path)
                    response = client.getresponse()
                    self.assertEqual(response.status, status)
                    response.read()
                client.close()
            finally:
                server.shutdown()
                thread.join(3)

    def test_registration_uses_nonmap_native_target_and_does_not_override_response_probe(self):
        register = module("register_pdu")
        secret = "ab" * 24
        args = register.arguments_for(secret)
        self.assertIn("-type=oracle_si_pdu", args)
        self.assertFalse(any("access_point" in arg or "HttpCommunicationProtocol" in arg for arg in args))
        self.assertNotIn("-force", args)
        self.assertIn("SkipSnmpSubscription->true", " ".join(args))
        self.assertIn("-monitoring_cred=snmp_v1v2_v3;oracle_si_pdu;SNMPV1Creds;COMMUNITY:" + secret, args)
        self.assertEqual(register.command_file_for(secret).count("\n"), 1)
        for value in ('bad"\nadd_target', "public", "AB" * 24):
            with self.assertRaises(ValueError):
                register.command_file_for(value)

    def test_deployment_is_isolated_nonroot_and_refuses_existing_state(self):
        deployment = module("deploy_pdu")
        def output(*arguments):
            if arguments[1:3] == ("network", "inspect"):
                return '[{"Containers": {}}]'
            return "container-test-id\n" if arguments[1] == "run" else ""
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            with patch.object(deployment, "STATE", state), patch.object(deployment.os, "geteuid", return_value=0), \
                    patch.object(deployment.os, "chown"), patch.object(deployment, "checked", side_effect=output) as calls, \
                    patch("builtins.print"):
                deployment.main()
                run = calls.call_args.args
                self.assertNotIn("-p", run)
                self.assertNotIn("--privileged", run)
                self.assertIn("--read-only", run)
                self.assertIn("54321:54321", run)
                secret = (state / "snmp-community").read_text().strip()
                self.assertNotIn(secret, " ".join(run))
                self.assertEqual((state / "snmp-community").stat().st_mode & 0o777, 0o400)
                receipt = json.loads((state / "created.json").read_text())
                self.assertEqual(receipt["scope"], "MANAGEMENT_AND_POWER_TELEMETRY")
                self.assertEqual(set(receipt["source_sha256"]), {"pdu_simulator.py", "switch_simulator.py"})
                with self.assertRaises(SystemExit):
                    deployment.main()

    def test_deployment_refuses_occupied_address(self):
        deployment = module("deploy_pdu")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(deployment, "STATE", Path(directory) / "state"), \
                    patch.object(deployment.os, "geteuid", return_value=0), \
                    patch.object(deployment, "checked", side_effect=["", json.dumps([
                        {"Containers": {"existing": {"IPv4Address": deployment.ADDRESS + "/16"}}}])]) as calls:
                with self.assertRaises(SystemExit):
                    deployment.main()
                self.assertEqual(calls.call_count, 2)

    def test_member_attachment_appends_only_fixed_sim_members_and_verifies_native_relation(self):
        from types import SimpleNamespace
        attach = module("attach_hardware")
        before = ("Source Target Name Member Target Name Member Target Type Level Key Member\n"
                  "AIOPS_SIM_EXADATA_DBM01 AIOPS_SIM_EXADATA_GRID01 oracle_exadata_grid 1\n"
                  "AIOPS_SIM_EXADATA_GRID01 AIOPS_SIM_EXADATA_CELL01 oracle_exadata 2\n")
        after = before + "".join(f"{attach.SYSTEM} {name} {kind} 1\n" for name, kind in attach.MEMBERS)
        def native(*args):
            if args[0] == "get_targets":
                name, kind = args[1].removeprefix("-targets=").split(":")
                output = f"1 Up {kind} {name}\n"
            elif args[0] == "get_system_members":
                output = after if len([call for call in calls.call_args_list
                    if call.args[0] == "get_system_members"]) > 1 else before
            else:
                output = "System modified successfully"
            return SimpleNamespace(returncode=0, stdout=output, stderr="")
        inspection = json.dumps([{"State": {"Running": True},
            "Config": {"Labels": {"aiops.simulation": "exadata"}}}])
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            with patch.object(attach, "STATE", state), patch.object(attach.os, "geteuid", return_value=0), \
                    patch.object(attach.subprocess, "check_output", return_value=inspection), \
                    patch.object(attach, "cli", side_effect=native) as calls, patch("builtins.print"):
                attach.main()
                modification = next(call.args for call in calls.call_args_list if call.args[0] == "modify_system")
                self.assertIn("-type=oracle_dbmachine", modification)
                self.assertNotIn("-force", modification)
                self.assertFalse(any("delete" in arg or "propag" in arg for arg in modification))
                receipt = json.loads((state / "result.json").read_text())
                self.assertTrue(receipt["native_members_verified"])
                self.assertFalse(receipt["application_bindings_confirmed"])
                with self.assertRaises(SystemExit):
                    attach.main()

    def test_native_zero_exit_with_warning_is_not_success(self):
        from types import SimpleNamespace
        attach = module("attach_hardware")
        for output in ("Warning: target not added", "Error: target type not allowed", "Failed"):
            self.assertFalse(attach.accepted(SimpleNamespace(returncode=0, stdout=output, stderr="")))

    def test_rack_registration_uses_only_native_allowed_composite_relation(self):
        rack = module("register_rack")
        create = rack.arguments_for("create")
        self.assertEqual(create[0], "add_target")
        self.assertIn("-type=oracle_si_rack", create)
        self.assertTrue(any("EngineeredSystemId:" + rack.SYSTEM in arg for arg in create))
        members = rack.arguments_for("members")
        self.assertEqual(members[0], "create_assoc")
        self.assertIn("-assoc_type=app_composite_contains", members)
        self.assertFalse(any("internal_contains" in arg for arg in members))
        system = rack.arguments_for("system")
        self.assertEqual(system[0], "modify_system")
        self.assertTrue(any(rack.RACK + ":oracle_si_rack" in arg for arg in system))
        self.assertFalse(any("-force" in arg or "delete" in arg for arg in create + members + system))

    def test_db_server_attachment_does_not_bypass_rack_constraint(self):
        attach = module("attach_hardware")
        self.assertEqual(attach.MEMBERS, (("aiops-exadata-monitor.lab", "host"),))

    def test_switch_response_schedule_is_native_previewed_and_scoped_to_sim(self):
        schedule = module("enable_switch_response")
        preview, apply = schedule.arguments("Y"), schedule.arguments("N")
        self.assertEqual(preview[:-1], apply[:-1])
        self.assertEqual(preview[-1], "-preview=Y")
        self.assertEqual(apply[-1], "-preview=N")
        self.assertIn("-targetNames=AIOPS_SIM_EXADATA_SWITCH01", apply)
        self.assertIn("-collectionName=AccessPointResponse", apply)
        self.assertIn("-freqValue=5", apply)
        self.assertNotIn("-force", apply)


if __name__ == "__main__":
    unittest.main()
