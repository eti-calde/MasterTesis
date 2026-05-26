#!/usr/bin/env bash
# Run EVERYTHING end-to-end on a single GPU host (>=6 GB VRAM).
# Intended for a fresh clone inside the docker container:
#
#   docker compose run --rm pinn bash scripts/run_all.sh
#
# Steps (each is idempotent and resumable — re-run picks up where it left off):
#   1. Regenerate the Exp 1-5 ground-truth datasets if missing.
#   2. Run the 4 light sweeps (ablation_forms, exp6, exp1_sensitivity,
#      arch_scaling 1D).
#   3. Run the 3 heavy 2D sweeps (arch_scaling 2D, exp2_n_t, exp5_n_t).
#   4. Tar up runs/ into runs.tar.gz for easy transfer.
#
# Total wall-time on an RTX 4060: ~6 hrs. On a GTX 1650: skip the heavy
# step (4 GB VRAM is too tight); run the script with HEAVY=0.
#
# Env overrides:
#   DEV=cuda|cpu        (default cuda)
#   HEAVY=1|0           (default 1; set 0 on <6 GB VRAM)
#   SKIP_REGEN=1        (skip dataset regen, e.g. if you already ran it)
#   SKIP_TAR=1          (skip the final tar)

set -uo pipefail

DEV="${DEV:-cuda}"
HEAVY="${HEAVY:-1}"
SKIP_REGEN="${SKIP_REGEN:-0}"
SKIP_TAR="${SKIP_TAR:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p runs

# Resolve interpreter (container path first, then host venv).
PY="${PY:-}"
if [[ -z "$PY" ]]; then
  if [[ -x /opt/venv/bin/python ]]; then
    PY=/opt/venv/bin/python
  elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"
  else
    echo "No python venv found. Run inside the docker container or 'uv sync' locally." >&2
    exit 1
  fi
fi
export PY

# CUDA sanity check.
if [[ "$DEV" == "cuda" ]]; then
  "$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available; set DEV=cpu'" \
    || { echo "CUDA check failed. Aborting."; exit 1; }
fi

echo "===================================================================="
echo "Full overnight pipeline — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Device: $DEV  |  HEAVY=$HEAVY  |  python: $PY"
echo "Repo: $(git rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
echo "===================================================================="

# ---------- 1) Regenerate datasets ----------------------------------------
if [[ "$SKIP_REGEN" == "0" ]]; then
  echo ""
  echo "### Step 1/4: Regenerate ground-truth datasets ###"
  bash scripts/regenerate_datasets.sh
else
  echo "### Step 1/4: dataset regen skipped (SKIP_REGEN=1) ###"
fi

# ---------- helper for timed, logged sweep runs ---------------------------
run() {
  local name="$1"; shift
  local log="runs/log_${name}.txt"
  local t0
  t0=$(date +%s)
  echo ""
  echo "[$(date '+%H:%M:%S')] >>> $name >>>" | tee -a "$log"
  "$@" 2>&1 | tee -a "$log"
  local elapsed=$(( $(date +%s) - t0 ))
  printf "[$(date '+%H:%M:%S')] <<< %s done in %dm%02ds <<<\n" \
    "$name" $((elapsed/60)) $((elapsed%60)) | tee -a "$log"
  return 0
}

# ---------- 2) Light sweeps ----------------------------------------------
echo ""
echo "### Step 2/4: Light sweeps (~4 hrs on a 4060) ###"

run ablation_forms "$PY" -m studies.ablation_forms \
  --device "$DEV" --study-dir runs/ablation_forms

run exp6_run_matrix "$PY" -m studies.exp6_run_matrix \
  --device "$DEV" --study-dir runs/exp6

run exp1_sensitivity "$PY" -m studies.exp1_sensitivity \
  --device "$DEV" --study-dir runs/exp1_sensitivity

run arch_scaling_1d "$PY" -m studies.arch_scaling \
  --device "$DEV" --study-dir runs/arch_scaling --cases exp1,exp2

# ---------- 3) Heavy 2D sweeps -------------------------------------------
if [[ "$HEAVY" == "1" ]]; then
  echo ""
  echo "### Step 3/4: Heavy 2D sweeps (~2-3 hrs on a 4060) ###"

  run arch_scaling_2d "$PY" -m studies.arch_scaling \
    --device "$DEV" --study-dir runs/arch_scaling --cases exp3,exp5

  run exp2_n_t_sweep "$PY" -m studies.exp2_n_t_sweep \
    --device "$DEV" --study-dir runs/exp2_n_t

  run exp5_n_t_sweep "$PY" -m studies.exp5_n_t_sweep \
    --device "$DEV" --study-dir runs/exp5_n_t
else
  echo "### Step 3/4: heavy 2D sweeps skipped (HEAVY=0) ###"
fi

# ---------- 4) Package results -------------------------------------------
if [[ "$SKIP_TAR" == "0" ]]; then
  echo ""
  echo "### Step 4/4: package runs/ for transfer ###"
  tar -czf runs.tar.gz runs/
  echo "Wrote runs.tar.gz ($(du -h runs.tar.gz | awk '{print $1}'))"
else
  echo "### Step 4/4: tar skipped (SKIP_TAR=1) ###"
fi

echo ""
echo "===================================================================="
echo "All done at $(date '+%Y-%m-%d %H:%M:%S')."
echo "Per-sweep logs:    runs/log_*.txt"
echo "Aggregated tarball: runs.tar.gz"
echo "===================================================================="
