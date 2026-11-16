#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DESTINATION="${1:?请指定独立部署目录，例如 /data/gcj_data/fast/baseline/tempo_full}"
python3 - "$ROOT" "$DESTINATION" <<'PY'
from pathlib import Path
import shutil
import sys

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
if source == destination:
    raise ValueError('完整模型部署需要使用独立目录，避免覆盖仓库原有启动入口')
for relative in ('tempo/src', 'tempo/tools', 'tempo/packages/openpi-client/src'):
    shutil.copytree(source / relative, destination / relative, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
files = {
    'trained_model/scripts/manage_model_service.py': 'trained_model/scripts/manage_model_service.py',
    'trained_model/config_use/deployment_full_all393.json': 'trained_model/deployment_all393.json',
    'trained_model/scripts/serve_tempo_full.sh': 'trained_model/scripts/serve_wired.sh',
    'trained_model/scripts/start_tempo_full_service.sh': 'trained_model/scripts/start_service.sh',
    'trained_model/config_use/deployment_full.json': 'trained_model/deployment.json',
    'trained_model/config_use/deployment_full_rtc_paper.json': 'trained_model/deployment_rtc_paper.json',
    'trained_model/config_use/deployment_full_step4000.json': 'trained_model/deployment_step4000.json',
    'tempo/tests/test_full_deployment.py': 'tests/test_full_deployment.py',
    'tempo/tests/test_exact_optimizations.py': 'tests/test_exact_optimizations.py',
    'tempo/tests/test_deployment_runtime.py': 'tests/test_deployment_runtime.py',
}
for original, target in files.items():
    path = destination / target
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / original, path)
(destination / 'trained_model/logs').mkdir(parents=True, exist_ok=True)
(destination / 'trained_model/runtime').mkdir(parents=True, exist_ok=True)
shutil.copytree(source / 'docs/tempo_full', destination / 'docs', dirs_exist_ok=True)
print(f'已安装完整模型部署源码与配置到 {destination}')
PY
