#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_dir="$repo_root/.upstream/nanochat"

if [[ ! -d "$upstream_dir/nanochat" ]]; then
    echo "nanochat source is missing; run: bash scripts/setup_env.sh" >&2
    exit 1
fi

export PYTHONPATH="$upstream_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"
exec uv run --locked --extra dev "$@"
