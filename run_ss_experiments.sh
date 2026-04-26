#!/usr/bin/env bash
# run_ss_experiments.sh — Steady-state thesis experiments across all environments.
#
# Execution order:
#   All maps are run once in full mode using the fixed planner setup:
#   CA* for charger/idle phases, RHCR for task phases.
#   Group 2 only starts after Group 1 fully completes.
#
# Usage:
#   bash run_ss_experiments.sh                  # full thesis run
#   bash run_ss_experiments.sh --workers 8      # override worker count (default: all cores)
#   bash run_ss_experiments.sh --ss-initial-tasks 0

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
WORKERS=12
SS_INITIAL_TASKS=0

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        --ss-initial-tasks) SS_INITIAL_TASKS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Resolve project root ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Activate venv ────────────────────────────────────────────────────────────
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f ".venv/Scripts/activate" ]]; then
    source .venv/Scripts/activate
else
    echo "ERROR: .venv not found. Run setup first (see SETUP.md)."
    exit 1
fi

# ── Verify install ───────────────────────────────────────────────────────────
echo "=== Verifying install ==="
python -c "from integration.orchestrator import IntegrationOrchestrator; print('OK')"

# ── Helper ───────────────────────────────────────────────────────────────────
run_exp() {
    local map="$1"
    local config="$2"
    local planner="$3"
    local label="$4"
    local methods="${5:-}"   # optional: space-separated method names (e.g. "gcbba dmchba")
    echo ""
    echo "============================================================"
    echo "  MAP: $map | config: $config | fixed planner setup"
    echo "  $label"
    echo "  ss_initial_tasks=$SS_INITIAL_TASKS"
    echo "============================================================"
    local methods_args=()
    if [[ -n "$methods" ]]; then
        methods_args=(--methods $methods)
    fi
    python experiments/run_experiments.py \
        --map        "$map" \
        --mode       full \
        --config     "$config" \
        --workers    "$WORKERS" \
        --ss-initial-tasks "$SS_INITIAL_TASKS" \
        "${methods_args[@]}"
}

# ════════════════════════════════════════════════════════════════════════════
#  GROUP 1 — Small maps  (warehouse_small, crossdock, kiva)
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "████████████████████████████████████████████████████████████████████"
echo "  GROUP 1 — Small maps"
echo "████████████████████████████████████████████████████████████████████"

run_exp gridworld_warehouse_small  ss_only  fixed  "N=6   — GCBBA + CBBA + SGA + DMCHBA"
run_exp gridworld_crossdock        ss_only  fixed  "N=50  — GCBBA + CBBA + SGA + DMCHBA"
run_exp gridworld_kiva             ss_only  fixed  "N=100 — GCBBA + CBBA + SGA + DMCHBA"

echo ""
echo "============================================================"
echo "  GROUP 1 complete."
echo "============================================================"

# ════════════════════════════════════════════════════════════════════════════
#  GROUP 2 — Large maps  (warehouse_large, kiva_large, shelf_aisle)
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "████████████████████████████████████████████████████████████████████"
echo "  GROUP 2 — Large maps"
echo "████████████████████████████████████████████████████████████████████"

run_exp gridworld_warehouse_large  ss_only  fixed  "N=18  — GCBBA + CBBA + SGA + DMCHBA"
run_exp gridworld_kiva_large       ss_only  fixed  "N=200 — GCBBA + DMCHBA"  "gcbba dmchba"
run_exp gridworld_shelf_aisle      ss_only  fixed  "N=470 — GCBBA + DMCHBA"  "gcbba dmchba"

echo ""
echo "============================================================"
echo "  GROUP 2 complete."
echo "  All steady-state experiments done."
echo "  Results written to results/experiments/<map>/<timestamp>/"
echo "============================================================"