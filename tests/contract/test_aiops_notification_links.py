"""真实共享通知脚本的资源定位与空间隔离回归，不依赖浏览器或 Oracle。"""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 Node.js 执行真实前端定位器")
@pytest.mark.parametrize("kind,page", [("run", "runs"), ("situation", "situations")])
def test_exact_resource_links_never_switch_domain_or_use_untrusted_locators(kind, page):
    resource_id = "019ffff0-0000-7000-8000-000000000013"
    script = """
const fs = require('node:fs');
eval(fs.readFileSync(process.argv[1], 'utf8'));
const kind = process.argv[2], id = process.argv[3];
const item = {domain_id:73, route_key:`aiops.${kind}.detail`,
  route_params:{domain_id:'73', [kind+'_id']:id}};
const href = KBotNotificationCenter.href;
console.log(JSON.stringify([
  href(item,73), href(item,74), href(item,undefined),
  href({...item,route_params:{...item.route_params,[kind+'_id']:'javascript:alert(1)'}},73),
  href({...item,route_params:{[kind+'_id']:id}},73),
  href({route_key:item.route_key,route_params:{[kind+'_id']:id}},undefined),
]));
"""
    result = subprocess.run(
        ["node", "-e", script, str(ROOT / "ui/shared/kbot-notifications.js"), kind, resource_id],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == [
        f"./{page}.html?{kind}Id={resource_id}", None, None, None, None, None,
    ]
