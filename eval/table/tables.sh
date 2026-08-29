#!/usr/bin/env bash
# Build table JSON and Markdown/LaTeX reports from saved metrics.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

exec python -m eval.table.run_tables "$@"
