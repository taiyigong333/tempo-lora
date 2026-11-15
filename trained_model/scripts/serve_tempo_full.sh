#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPO_PYTHON="${TEMPO_PYTHON:-/data/gcj_data/fast/baseline/tempo/tempo/.venv/bin/python}"
export PYTHONPATH="$ROOT/tempo/src:$ROOT/tempo/packages/openpi-client/src:$ROOT/tempo${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$ROOT/trained_model/cache/inductor"
cd "$ROOT"
exec "$TEMPO_PYTHON" -m openpi.tempo_full.inference --config "$ROOT/trained_model/deployment.json" "$@"
