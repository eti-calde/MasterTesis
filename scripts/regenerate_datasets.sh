#!/usr/bin/env bash
# Regenerate the per-experiment ground-truth .npz files that are
# gitignored (rebuildable from each `ground_truth.py` / `generate_and_plot.py`).
#
# Run once after `git clone` (or inside the docker container) before
# launching any sweep. Each Exp's generator is deterministic so the
# resulting .npz are identical across machines.
#
# Skips Exp 6 (Angel) — its processed data ships in the repo under
# Experiments/datasets/angel2024/processed/ (kept tracked despite the
# global datasets/ ignore — see .gitignore exception).
#
# Usage:
#   bash scripts/regenerate_datasets.sh
#   PY=/path/to/python bash scripts/regenerate_datasets.sh   # custom interpreter

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  # Inside the docker image the venv lives at /opt/venv.
  if [[ -x /opt/venv/bin/python ]]; then
    PY=/opt/venv/bin/python
  else
    echo "venv python not found. Run 'uv venv && uv sync' or set PY explicitly." >&2
    exit 1
  fi
fi

echo "Using interpreter: $PY"
echo ""

generate_exp() {
  local exp_dir="$1"
  local exp_name
  exp_name="$(basename "$exp_dir")"
  echo "=== $exp_name ==="
  if [[ ! -f "$exp_dir/generate_and_plot.py" ]]; then
    echo "  (skip: no generate_and_plot.py)"
    return 0
  fi
  ( cd "$exp_dir" && "$PY" generate_and_plot.py )
  echo "  data:"
  ls -lh "$exp_dir"/data/*.npz 2>/dev/null | awk '{printf "    %s  %s\n", $5, $NF}'
  echo ""
}

for d in \
  Experiments/01-subcritical-bump-1d \
  Experiments/02-thacker-basin-1d \
  Experiments/03-two-cylinders-2d \
  Experiments/04-tidal-oscillatory-2d \
  Experiments/05-thacker-paraboloid-3d; do
  generate_exp "$d"
done

echo "===================================================================="
echo "All 1-5 datasets regenerated."
echo "Exp 6 uses the pre-shipped Experiments/datasets/angel2024/processed/."
echo "===================================================================="
