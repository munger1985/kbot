"""分阶段调用未修改的原厂 本地监控 Agent 安装器，保留前置失败及非空卷。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import zipfile

import sys

sys.path.insert(0, "/opt/aiops-oem-lab/scripts")
from oem_install_status import redact


ROOT = Path("/opt/aiops-oem-lab")
STATE = ROOT / "state/exadata-sim-monitor"
CONTAINER = "aiops-exadata-sim-monitor"
PARENT = "/opt/oem-remote"
BASE = PARENT + "/agent"
HOME = BASE + "/agent_24.1.0.0.0"
SOFTWARE = PARENT + "/software"
VALUES = []


def command(*arguments, root=False):
    return ["docker", "exec", "--user", "root" if root else "oracle", CONTAINER, *map(str, arguments)]


def checked(*arguments, root=False):
    result = subprocess.run(command(*arguments, root=root), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        print(redact(result.stdout, VALUES))
        raise SystemExit("本地监控 Agent 文件操作失败；保留现场。")
    return result.stdout


def copy(source, destination):
    subprocess.run(["docker", "cp", str(source), CONTAINER + ":" + destination], check=True)


def native(arguments, phase):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = STATE / (phase + "-" + stamp + ".log")
    process = subprocess.Popen(arguments, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, errors="replace")
    collected = []
    with log.open("x") as output:
        log.chmod(0o600)
        for line in process.stdout:
            line = redact(line, VALUES) + "\n"
            print(line, end="", flush=True)
            output.write(line)
            output.flush()
            collected.append(line)
    code = process.wait()
    process.stdout.close()
    (STATE / (phase + "-result.json")).write_text(json.dumps({
        "native_exit_code": code, "phase": phase, "log": log.name,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }) + "\n")
    if code:
        raise SystemExit(f"本地监控 Agent {phase} 原生返回码 {code}；保留现场，不忽略或重装。")
    return "".join(collected)


def deploy(phase):
    arguments = [SOFTWARE + "/agentDeploy.sh", "AGENT_BASE_DIR=" + BASE,
                 "RESPONSE_FILE=" + PARENT + "/agent.rsp", "REMOTE_AGENT=false",
                 "-invPtrLoc", PARENT + "/oraInst.loc"]
    if phase == "prereq":
        arguments.append("-prereqOnly")
    output = native(command("/usr/bin/env", "-u", "ORACLE_HOME", "-u", "ORACLE_SID", "-u", "LD_LIBRARY_PATH",
                            "/bin/bash", "-c", "ulimit -Sn 65536 && exec " + shlex.join(arguments)), phase)
    if phase == "prereq":
        candidates = checked("find", HOME + "/cfgtoollogs/prereqlogs", "-type", "f",
                             "-name", "oraInstall*.out", "-printf", "%T@ %p\n").splitlines()
        latest = max(candidates, key=lambda value: float(value.split(" ", 1)[0])).split(" ", 1)[1]
        checks = parse_prereq(checked("cat", latest))
        (STATE / "prereq-checks.json").write_text(json.dumps({
            "report": latest, "checks": checks,
            "all_passed": len(checks) == 11 and all(v == "Passed" for v in checks.values()),
        }) + "\n")
        if len(checks) != 11 or any(value != "Passed" for value in checks.values()):
            raise SystemExit("原厂检查含失败、未执行或缺项；即使脚本返回 0 也不允许正式安装。")
    return output


def parse_prereq(text):
    checks = {}
    for block in text.split("Performing check for ")[1:]:
        name = block.splitlines()[0].strip()
        match = re.search(r"overall result of this check is:\s*([^\n]+)", block)
        if match is None:
            match = re.search(r"Check complete:\s*([^\n]+)", block)
        checks[name] = match[1].strip() if match else "Unknown"
    return checks


def main():
    global VALUES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "prereq", "install", "root", "status"))
    parser.add_argument("--resume-empty-preparation", action="store_true",
                        help="仅恢复本次新卷中已核对的两个空准备目录，不覆盖任何软件")
    parser.add_argument("--resume-directory-guard", action="store_true",
                        help="仅恢复原厂因前置检查 Home 已存在而在安装前拒绝的情况；归档而不删除")
    arguments = parser.parse_args()
    phase = arguments.phase
    if os.geteuid() != 0:
        raise SystemExit("请以 root 运行以保护注册凭据。")
    VALUES = [p.read_text().strip() for p in (ROOT / "secrets").iterdir() if p.is_file()]
    metadata = json.loads(subprocess.check_output(["docker", "inspect", CONTAINER]))[0]
    if not metadata["State"]["Running"]:
        raise SystemExit("独立 Agent 容器尚未运行。")
    if not any(m.get("Name") == "aiops_exadata_sim_monitor_data" and m["Destination"] == PARENT
               for m in metadata["Mounts"]):
        raise SystemExit("持久卷不匹配，不接管未知容器。")
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    STATE.chmod(0o700)
    prepared = STATE / "prepared.json"
    if phase == "status":
        print(checked(HOME + "/bin/emctl", "status", "agent"))
        return
    if phase == "prepare":
        contents = set(checked("find", PARENT, "-mindepth", "1", "-maxdepth", "1", "-printf", "%f\n").splitlines())
        empty_preparation = (arguments.resume_empty_preparation and contents == {"software", "inventory"}
                             and not checked("find", SOFTWARE, PARENT + "/inventory", "-mindepth", "1").strip()
                             and checked("stat", "-c", "%u", SOFTWARE, PARENT + "/inventory").split() == ["54321", "54321"])
        if prepared.exists() or (contents and not empty_preparation):
            raise SystemExit("已存在准备记录或非空卷，不覆盖、不重解包。")
        archives = list((ROOT / "staging/agent19").glob("24.1.0.0.0_AgentCore_*.zip"))
        if len(archives) != 1:
            raise SystemExit("没有唯一匹配的原版 Agent 介质。")
        archive = archives[0]
        with zipfile.ZipFile(archive) as source:
            if any(Path(i.filename).is_absolute() or ".." in Path(i.filename).parts for i in source.infolist()):
                raise SystemExit("归档存在越界路径。")
            if source.testzip() is not None:
                raise SystemExit("Agent ZIP CRC 检查失败。")
        checked("install", "-d", "-o", "oracle", "-g", "oinstall", "-m", "750",
                SOFTWARE, PARENT + "/inventory", root=True)
        # 原始 staging 保持私有权限，不为容器 oracle 放宽宿主机介质目录。
        copy(archive, PARENT + "/agent-core.zip")
        checked("chown", "oracle:oinstall", PARENT + "/agent-core.zip", root=True)
        checked("chmod", "640", PARENT + "/agent-core.zip", root=True)
        checked("unzip", "-q", PARENT + "/agent-core.zip", "-d", SOFTWARE)
        response = STATE / "agent.rsp"
        descriptor = os.open(response, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w") as output:
            output.write("\n".join((
                "OMS_HOST=aiops-oem.lab", "EM_UPLOAD_PORT=4903",
                "AGENT_REGISTRATION_PASSWORD=" + (ROOT / "secrets/agent_registration_password").read_text().strip(),
                "AGENT_PORT=3872", "ORACLE_HOSTNAME=aiops-exadata-monitor.lab", "REMOTE_AGENT=false",
                "START_AGENT=true", "AGENT_INSTANCE_HOME=" + BASE + "/agent_inst",
            )) + "\n")
        pointer = STATE / "oraInst.loc"
        pointer.write_text("inventory_loc=" + PARENT + "/inventory\ninst_group=oinstall\n")
        pointer.chmod(0o600)
        for source, destination in ((response, PARENT + "/agent.rsp"), (pointer, PARENT + "/oraInst.loc")):
            copy(source, destination)
            checked("chown", "oracle:oinstall", destination, root=True)
            checked("chmod", "600", destination, root=True)
        with archive.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        prepared.write_text(json.dumps({"archive": archive.name, "sha256": digest,
                                        "agent_home": HOME, "local_agent": True}) + "\n")
        deploy("prereq")
        return
    if not prepared.is_file() or json.loads(prepared.read_text()).get("agent_home") != HOME:
        raise SystemExit("没有匹配的准备凭证，不接管未知安装。")
    if phase == "prereq":
        if (STATE / ".install-started").exists():
            raise SystemExit("已有正式安装记录，不重跑准备入口。")
        deploy("prereq")
    elif phase == "install":
        if (STATE / ".install-started").exists():
            previous = json.loads((STATE / "install-result.json").read_text())
            expected = "Directory " + HOME + " already exist, pass -force option to overwrite"
            if (not arguments.resume_directory_guard or previous.get("native_exit_code") != 1
                    or (STATE / previous["log"]).read_text().strip() != expected):
                raise SystemExit("已有正式安装记录，不自动重装或覆盖。")
            (STATE / "install-result.before-directory-guard-recovery.json").write_text(json.dumps(previous) + "\n")
        if json.loads((STATE / "prereq-result.json").read_text()).get("native_exit_code") != 0:
            raise SystemExit("原厂前置检查未通过。")
        report = json.loads((STATE / "prereq-checks.json").read_text())
        if (report.get("all_passed") is not True or len(report.get("checks", {})) != 11
                or any(v != "Passed" for v in report["checks"].values())):
            raise SystemExit("原厂完整检查未全部通过，不按脚本退出码放行。")
        # 原厂 -prereqOnly 在目标 Home 留下自身的 OUI/JDK，正式入口拒绝非空 Home。
        # 只归档本次独立卷、空 inventory、没有 Agent 二进制/实例的前置检查现场。
        entries = set(checked("find", HOME, "-mindepth", "1", "-maxdepth", "1", "-printf", "%f\n").splitlines())
        if (entries != {"oui", "jdk", "modules", "cfgtoollogs", "jlib"}
                or checked("find", PARENT + "/inventory", "-mindepth", "1").strip()
                or subprocess.run(command("test", "-e", BASE + "/agent_inst")).returncode == 0):
            raise SystemExit("Home 不符合本次仅前置检查的精确目录形态，禁止移动或覆盖。")
        destination = PARENT + "/prereq-home-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        checked("mv", "-T", "--", HOME, destination, root=True)
        report["archived_report"] = report["report"].replace(HOME + "/", destination + "/", 1)
        (STATE / "prereq-home-archive.json").write_text(json.dumps(report) + "\n")
        (STATE / ".install-started").write_text(datetime.now(timezone.utc).isoformat() + "\n")
        deploy("install")
    elif phase == "root":
        result = STATE / "install-result.json"
        if not result.is_file() or json.loads(result.read_text()).get("native_exit_code") != 0:
            raise SystemExit("原厂安装未成功，不执行 root 收尾。")
        receipt = STATE / "root-result.json"
        if not receipt.exists():
            native(command("/bin/bash", HOME + "/root.sh", root=True), "root")
        elif json.loads(receipt.read_text()).get("native_exit_code") != 0:
            raise SystemExit("上次 root 执行失败，不自动重跑。")
        uid, mode = checked("stat", "-c", "%u %a", HOME + "/sbin/nmo", root=True).split()
        if uid != "0" or not int(mode, 8) & 0o4000:
            raise SystemExit("原厂 nmo root 配置未完成。")
        output = native(command(HOME + "/bin/emctl", "status", "agent"), "verify")
        if ("Agent is Running and Ready" not in output
                or not re.search(r"Heartbeat Status\s*:\s*Ok", output)):
            raise SystemExit("Agent 尚未就绪或心跳失败；不写成功标记。")
        configured = STATE / "configured.json"
        configured.write_text(json.dumps({"local_agent": True, "agent_home": HOME,
                                         "configured_at": datetime.now(timezone.utc).isoformat()}) + "\n")
        copy(configured, PARENT + "/.configured")
        checked("chmod", "600", PARENT + "/.configured", root=True)
        checked("rm", "-f", "--", PARENT + "/agent.rsp", root=True)
        (STATE / "agent.rsp").unlink(missing_ok=True)
        print("标准 Agent 安装、root 与心跳成功；仍需验证模拟 Cell 的真实 OEM 采样。")


if __name__ == "__main__":
    main()
