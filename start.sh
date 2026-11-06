#!/usr/bin/env bash
# Run the latest source tree directly (no reinstall / no build step).
#
#   ./start.sh                 # serve on the default host/port
#   ./start.sh --port 9000     # options are forwarded to `serve`
#   ./start.sh check           # plugin topology check
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "error: $PY not found; create the .venv first (see README)." >&2
  exit 1
fi

# No subcommand -> serve; leading options -> serve with those options.
if [[ $# -eq 0 ]]; then
  set -- serve
elif [[ "${1#-}" != "$1" ]]; then
  set -- serve "$@"
fi

# Put the source tree ahead of any installed copy so edits take effect at once.
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m src "$@"
