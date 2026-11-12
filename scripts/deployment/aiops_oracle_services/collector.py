"""可选的 Oracle 基础服务只读采集器；每次现采，不重放缓存或连接成功健康值。"""

import argparse
from datetime import UTC, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


ASM_CAPACITY = """SELECT name,total_mb,free_mb,usable_file_mb,
    CASE WHEN total_mb>0 THEN (total_mb-free_mb)*100/total_mb END
    FROM SYS.V_$ASM_DISKGROUP_STAT WHERE state='MOUNTED'"""
ASM_DISKS = """SELECT g.name,d.name,d.mode_status FROM SYS.V_$ASM_DISK_STAT d
    JOIN SYS.V_$ASM_DISKGROUP_STAT g ON g.group_number=d.group_number"""
ASM_COUNTERS = """SELECT g.name,TO_CHAR(d.inst_id),SUM(d.reads),SUM(d.writes),
    SUM(d.bytes_read)/1048576,SUM(d.bytes_written)/1048576,
    SUM(d.read_time)*1000,SUM(d.write_time)*1000
    FROM SYS.GV_$ASM_DISK_STAT d JOIN SYS.GV_$ASM_DISKGROUP_STAT g
    ON g.group_number=d.group_number AND g.inst_id=d.inst_id
    WHERE g.state='MOUNTED' GROUP BY g.name,d.inst_id"""
RAC_INSTANCES = 'SELECT instance_name,status FROM SYS.GV_$INSTANCE'
RAC_SERVICES = 'SELECT DISTINCT name,inst_id FROM SYS.GV_$ACTIVE_SERVICES'
GAUGES = {'asm.free_mb': 'MB', 'asm.total_mb': 'MB', 'asm.usable_file_mb': 'MB', 'asm.used_percent': '%'}
COUNTERS = ('asm.reads_total', 'asm.writes_total', 'asm.read_mb_total', 'asm.written_mb_total',
    'asm.read_time_ms_total', 'asm.write_time_ms_total')
NAMES = {code: 'aiops_oracle_' + code.replace('.', '_') for code in (*GAUGES, *COUNTERS,
    'asm.disk_mode', 'rac.instances', 'rac.service_status', 'listener.availability', 'listener.tns_ping_ms', 'listener.registration')}


def config(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.geteuid():
        raise ValueError('配置必须是当前运行用户持有的0600普通文件，不接受符号链接')
    value = json.loads(path.read_text())
    if value.get('resource_type') not in {'ORACLE_ASM', 'ORACLE_RAC', 'ORACLE_LISTENER'}:
        raise ValueError('资源类型无效')
    if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', value.get('target_key', '')):
        raise ValueError('必须显式配置稳定 target_key')
    return value


def row(code, value, dimensions=None):
    if value is None:
        return None
    numeric = Decimal(str(value))
    if not numeric.is_finite():
        raise ValueError('采样必须是有限数值')
    return {'code': code, 'value': str(numeric), 'dimensions': dimensions or {}}


def sql_rows(connection, query):
    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchmany(1001)
        if len(rows) > 1000:
            raise ValueError('采集结果超出预算，拒绝发布截断值')
        return rows


def collect_sql(cfg, connect=None):
    if connect is None:
        import oracledb
        connect = oracledb.connect
    # 不开放 SQL、数据库写入、SYSDBA/SYSASM 或远程命令入口。
    connection = connect(user=cfg['username'], password=cfg['password'], dsn=cfg['dsn'], tcp_connect_timeout=5)
    result = []
    with connection:
        connection.call_timeout = 10000
        if cfg['resource_type'] == 'ORACLE_ASM':
            identity = sql_rows(connection, 'SELECT instance_name FROM SYS.V_$INSTANCE')
            if identity != [(cfg['instance_name'],)]:
                raise ValueError('ASM 实例身份与配置不匹配')
            instance_type = sql_rows(connection, "SELECT value FROM SYS.V_$PARAMETER WHERE name='instance_type'")
            if len(instance_type) != 1 or str(instance_type[0][0]).upper() != 'ASM':
                raise ValueError('目标不是 ASM 实例，不以普通数据库冒充 ASM 服务')
            for group, total, free, usable, used in sql_rows(connection, ASM_CAPACITY):
                result.extend(row(code, number, {'diskgroup': group}) for code, number in (
                    ('asm.total_mb', total), ('asm.free_mb', free), ('asm.usable_file_mb', usable), ('asm.used_percent', used)))
            for group, disk, mode in sql_rows(connection, ASM_DISKS):
                if mode in {'ONLINE', 'OFFLINE'}:
                    result.append(row('asm.disk_mode', 1 if mode == 'ONLINE' else 0, {'diskgroup': group, 'disk': disk}))
            for group, instance, *values in sql_rows(connection, ASM_COUNTERS):
                result.extend(row(code, number, {'diskgroup': group, 'inst_id': instance}) for code, number in zip(COUNTERS, values, strict=True))
        else:
            identity = sql_rows(connection, 'SELECT name FROM SYS.V_$DATABASE')
            if identity != [(cfg['database_name'],)]:
                raise ValueError('RAC 数据库身份与配置不匹配')
            if sql_rows(connection, "SELECT value FROM SYS.V_$PARAMETER WHERE name='cluster_database'") != [('TRUE',)]:
                raise ValueError('目标未声明为 RAC，不以普通数据库冒充集群')
            instances = sql_rows(connection, RAC_INSTANCES)
            for instance, status in instances:
                if status in {'OPEN', 'MOUNTED', 'STARTED'}:
                    result.append(row('rac.instances', {'OPEN': 1, 'MOUNTED': 2, 'STARTED': 3}[status], {'instance': instance}))
            configured = {item[0] for item in sql_rows(connection, 'SELECT name FROM SYS.DBA_SERVICES')}
            active = {item[0] for item in sql_rows(connection, RAC_SERVICES)}
            for service in cfg.get('services', []):
                if service not in configured:
                    raise ValueError('所选服务不在数据库配置中，不把未核验名称当停服')
                result.append(row('rac.service_status', int(service in active), {'database': cfg['database_name'], 'service': service}))
    return [item for item in result if item is not None]


def local_command(home, binary, arguments):
    executable = Path(home) / 'bin' / binary
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError('必须显式提供已安装的 Oracle Home')
    environment = {key: os.environ[key] for key in ('PATH', 'LD_LIBRARY_PATH', 'TNS_ADMIN') if key in os.environ}
    environment.update(ORACLE_HOME=str(Path(home)), NLS_LANG='AMERICAN_AMERICA.AL32UTF8', LC_ALL='C')
    result = subprocess.run([str(executable), *arguments], env=environment, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10, check=False)
    if len(result.stdout) > 65536:
        raise ValueError('原厂工具输出超出预算')
    return result.returncode, result.stdout.decode('utf-8', errors='strict')


def parse_listener_services(text, expected):
    """只解释真实 READY 实例；静态 UNKNOWN 不冒充动态注册。"""
    services, current = {}, None
    for line in text.splitlines():
        match = re.fullmatch(r'\s*Service "([^"]+)" has \d+ instance\(s\)\.\s*', line)
        if match:
            current = match[1]
            services.setdefault(current, [])
        match = re.fullmatch(r'\s*Instance "([^"]+)", status ([A-Z]+), has \d+ handler\(s\) for this service\.\.\.\s*', line)
        if match and current:
            services[current].append(match[2])
    result = []
    if 'The command completed successfully' not in text or not services:
        raise ValueError('无法解析原厂 Listener 动态服务列表')
    for name in expected:
        states = services.get(name, [])
        # 原厂成功返回的完整列表可以证明所选服务不在列表；UNKNOWN 仍不作判断。
        if not states or 'READY' in states or all(state in {'BLOCKED', 'RESTRICTED'} for state in states):
            result.append(row('listener.registration', int('READY' in states), {'service': name}))
    return result


def collect_listener(cfg, command=local_command):
    alias, listener = cfg['tns_alias'], cfg['listener_name']
    if not all(re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', name) for name in (alias, listener)):
        raise ValueError('只接受本地配置的 TNS 别名与 Listener 名称')
    code, text = command(cfg['oracle_home'], 'tnsping', [alias, '1'])
    matches = re.findall(r'^OK \((\d+) msec\)$', text, flags=re.MULTILINE)
    result = []
    if code == 0 and len(matches) == 1:
        result = [row('listener.availability', 1), row('listener.tns_ping_ms', matches[0])]
    elif re.search(r'\bTNS-125(?:41|43|45|35)\b', text):
        result = [row('listener.availability', 0)]
    else:
        raise ValueError('TNS 返回未审核错误，不能判为服务宕机')
    code, text = command(cfg['oracle_home'], 'lsnrctl', ['services', listener])
    if code == 0:
        result.extend(parse_listener_services(text, cfg.get('services', [])))
    return result


def collect(cfg):
    samples = collect_listener(cfg) if cfg['resource_type'] == 'ORACLE_LISTENER' else collect_sql(cfg)
    return {'target_key': cfg['target_key'], 'resource_type': cfg['resource_type'],
        'observed_at': datetime.now(UTC).isoformat(), 'samples': samples}


def prometheus(document):
    def quoted(value):
        value = str(value)
        if any(ord(char) < 32 and char != '\n' or ord(char) == 127 for char in value):
            raise ValueError('指标标签含有协议不支持的控制字符')
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'
    lines, emitted = [], set()
    for sample in document['samples']:
        name = NAMES[sample['code']]
        if name not in emitted:
            lines.extend(['# HELP ' + name + ' Oracle 原厂只读现采，未知不补零',
                '# TYPE ' + name + (' counter' if sample['code'] in COUNTERS else ' gauge')])
            emitted.add(name)
        labels = {'target_key': document['target_key'], 'resource_scope': document['resource_type'], **sample['dimensions']}
        lines.append(name + '{' + ','.join(key + '=' + quoted(value)
            for key, value in sorted(labels.items())) + '} ' + sample['value'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--format', choices=['json', 'prometheus'], default='json')
    parser.add_argument('--listen', type=int, help='仅绑定127.0.0.1，供本机Prometheus或受控反代采集')
    args = parser.parse_args()
    if args.listen:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != '/metrics':
                    self.send_error(404)
                    return
                try:
                    body = prometheus(collect(config(args.config))).encode()
                except Exception:
                    self.send_error(503, 'collector unavailable')
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *_):
                pass
        HTTPServer(('127.0.0.1', args.listen), Handler).serve_forever()
    else:
        document = collect(config(args.config))
        print(prometheus(document) if args.format == 'prometheus' else json.dumps(document, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('Oracle 服务采集失败；未输出健康零值，连接和原厂正文不写日志。', file=sys.stderr)
        raise SystemExit(1)
