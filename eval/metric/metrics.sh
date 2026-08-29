#!/usr/bin/env bash
# Aggregate saved rollout JSONL files; no GPU/model loading is required.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

exec python -m eval.metric.run_metrics "$@"
