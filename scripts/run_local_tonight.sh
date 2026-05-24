#!/usr/bin/env bash
# Run the 4 lightweight sweeps on the local machine (GTX 1650, CUDA).
# Total estimated wall-time: ~10 hours on CUDA, fits an overnight run.
#
# Heavy 2D stuff is reserved for azirafel:
#   - arch_scaling --cases exp3,exp5
#   - exp2_n_t_sweep
#   - exp5_n_t_sweep
#
# Each command tees its stdout to runs/log_<study>.txt so you can tail
# them in another terminal: tail -f runs/log_*.txt
#
# Usage:
#   tmux new -s local-tonight
#   cd /mnt/datos/Code/MasterTesis
#   source .venv/bin/activate
#   bash scripts/run_local_tonight.sh
#   # Ctrl-b d to detach; re-attach with: tmux attach -t local-tonight

set -uo pipefail   # NOT -e: per-sweep failure must not kill the rest.
                    # M8 already makes each sweep tolerate per-run errors.

DEV="${DEV:-cuda}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p runs

# Sanity: CUDA must be available unless DEV=cpu was passed explicitly.
if [[ "$DEV" == "cuda" ]]; then
  python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available; set DEV=cpu'" \
    || { echo "CUDA check failed. Aborting."; exit 1; }
fi

echo "===================================================================="
echo "Local overnight sweep — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Device: $DEV"
echo "Repo: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"
echo "===================================================================="

run() {
  local name="$1"; shift
  local log="runs/log_${name}.txt"
  local t0
  t0=$(date +%s)
  echo ""
  echo "[$(date '+%H:%M:%S')] >>> $name >>>" | tee -a "$log"
  "$@" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  local elapsed=$(( $(date +%s) - t0 ))
  printf "[$(date '+%H:%M:%S')] <<< %s done in %dm%02ds (rc=%d) <<<\n" \
    "$name" $((elapsed/60)) $((elapsed%60)) "$rc" | tee -a "$log"
  return 0   # always proceed to the next sweep
}

# Fast (~45 min)
run ablation_forms python -m studies.ablation_forms \
  --device "$DEV" --study-dir runs/ablation_forms

# Fast (~1.25 h)
run exp6_run_matrix python -m studies.exp6_run_matrix \
  --device "$DEV" --study-dir runs/exp6

# Medium (~2.5 h)
run exp1_sensitivity python -m studies.exp1_sensitivity \
  --device "$DEV" --study-dir runs/exp1_sensitivity

# Heaviest local sweep (~5-7 h). arch_scaling restricted to 1D cases
# so it fits in the GTX 1650 budget; exp3 + exp5 (2D) go to azirafel.
run arch_scaling_1d python -m studies.arch_scaling \
  --device "$DEV" --study-dir runs/arch_scaling --cases exp1,exp2

echo ""
echo "===================================================================="
echo "All local sweeps finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Per-sweep logs in runs/log_*.txt"
echo "Tomorrow on azirafel run:"
echo "  python -m studies.arch_scaling --device cuda --study-dir runs/arch_scaling --cases exp3,exp5"
echo "  python -m studies.exp2_n_t_sweep --device cuda --study-dir runs/exp2_n_t"
echo "  python -m studies.exp5_n_t_sweep --device cuda --study-dir runs/exp5_n_t"
echo "===================================================================="
