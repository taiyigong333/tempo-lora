#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec "$ROOT/client/.venv/bin/python" -m tempo_client.gui "$@"
