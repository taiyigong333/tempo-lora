#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f trained_model/runtime/status.json ]]; then
    cat trained_model/runtime/status.json
fi
for stage in mvae decoder train; do
    file="trained_model/logs/${stage}_alltrain.log"
    if [[ -f "$file" ]]; then
        printf '\n%s\n' "$file"
        tail -n 3 "$file"
    fi
done
nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv
