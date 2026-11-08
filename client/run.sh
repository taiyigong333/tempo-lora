#!/usr/bin/env bash
set -euo pipefail
CLIENT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$CLIENT/.venv/bin/python" -m tempo_client.cli "$@"
