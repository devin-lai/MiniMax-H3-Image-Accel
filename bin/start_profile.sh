#!/usr/bin/env bash
set -euo pipefail
base=${H3_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
export H3_ROOT="$base"
exec python3 "$base/bin/launch_profile.py" "$@"
