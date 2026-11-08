#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT="$ROOT/trained_model/checkpoints/pi05_piperx_tempo_ut_lora/formal_20261108_b16_g1/step_00010000.pt"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/trained_model/cache/inductor_inference"
# 显式覆盖部署地址，保留checkpoint及其训练配置。
exec bash "$ROOT/trained_model/scripts/run.sh" serve --checkpoint "$CHECKPOINT" --host 192.168.50.2 --port 8000 "$@"
