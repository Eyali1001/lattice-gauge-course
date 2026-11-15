#!/usr/bin/env bash
# Build and execute chapter notebooks from their jupytext sources.
# Usage: ./build.sh            (all chapters)
#        ./build.sh ch02       (chapters whose file name contains "ch02")
set -euo pipefail
cd "$(dirname "$0")"
uv run python -m ipykernel install --sys-prefix --name lgt --display-name "Python (lgt course)" >/dev/null
for src in chapters/ch*.py; do
  [[ -n "${1:-}" && "$src" != *"$1"* ]] && continue
  echo "== $src"
  time uv run jupytext --to ipynb --set-kernel lgt --execute "$src"
done
# Fail loudly if any executed cell raised an error (jupytext --execute does not).
uv run python - <<'PY'
import glob, json, sys
bad = [(p, i, o["ename"]) for p in sorted(glob.glob("chapters/ch*.ipynb"))
       for i, c in enumerate(json.load(open(p))["cells"])
       for o in c.get("outputs", []) if o.get("output_type") == "error"]
for b in bad: print("CELL ERROR:", *b)
sys.exit(1 if bad else 0)
PY
# Trust the freshly executed notebooks so their animations and 3D figures display.
uv run jupyter trust chapters/ch*.ipynb >/dev/null
