#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/run_gui.sh" --config "$ROOT/client/configs/piperx_full_vlash.yaml" "$@"
