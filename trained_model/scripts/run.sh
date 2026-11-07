#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$ROOT/trained_model/cache/uv"
export JAX_PLATFORMS=cpu
export OPENPI_DATA_HOME="$ROOT/trained_model/cache/openpi"
export HF_HOME="$ROOT/trained_model/cache/huggingface"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
# 复用服务器已有登录，密钥只进入进程环境，不写入配置、日志或仓库。
if [[ -z "${WANDB_API_KEY:-}" && -f "$ROOT/../openpi/.wandb_env" ]]; then
    source "$ROOT/../openpi/.wandb_env"
fi
cd "$ROOT/tempo"
exec uv run --no-sync python scripts/piperx.py "$@"
