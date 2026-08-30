#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_dir="$repo_root/.upstream/nanochat"
upstream_url="https://github.com/karpathy/nanochat.git"
upstream_commit="92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"

cd "$repo_root"
uv sync --extra dev --locked

if [[ ! -d "$upstream_dir/.git" ]]; then
    mkdir -p "$(dirname -- "$upstream_dir")"
    git clone --filter=blob:none "$upstream_url" "$upstream_dir"
fi

if [[ -n "$(git -C "$upstream_dir" status --short)" ]]; then
    echo "Refusing to update dirty nanochat checkout: $upstream_dir" >&2
    exit 1
fi

if [[ "$(git -C "$upstream_dir" remote get-url origin)" != "$upstream_url" ]]; then
    echo "Unexpected nanochat origin in $upstream_dir" >&2
    exit 1
fi

git -C "$upstream_dir" fetch --depth 1 origin "$upstream_commit"
git -C "$upstream_dir" checkout --detach "$upstream_commit"

echo "Environment ready: $repo_root/.venv"
echo "nanochat source: $upstream_dir ($upstream_commit)"
