#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$ROOT/trained_model/cache/uv"
export UV_LINK_MODE=copy
export GIT_LFS_SKIP_SMUDGE=1
cd "$ROOT/tempo"
uv sync --python 3.11 --frozen
# 直接核对实际导入目录，避免补丁被复制到另一个 Python 环境。
uv run --no-sync python - <<'PY'
import pathlib
import shutil
import transformers
source = pathlib.Path('src/openpi/models_pytorch/transformers_replace')
target = pathlib.Path(transformers.__file__).parent
assert transformers.__version__ == '4.53.2'
shutil.copytree(source, target, dirs_exist_ok=True)
print('Transformers 补丁目录', target)
PY
SAM2_BUILD_CUDA=0 uv pip install --no-deps --no-build-isolation -e "$ROOT/trained_model/cache/sam2"
uv run --no-sync python -c 'from openpi.tempo_lora.model import require_patch; require_patch(); import sam2, peft; print("依赖及补丁检查通过", peft.__version__)'
