#!/usr/bin/env bash
# Score all four experiments once the runs for all eight models are in place (CPU only).
# Usage: bash scripts/score_all.sh
set -euo pipefail
source "$(dirname "$0")/models.sh"

# Experiment 1 — necessity, site specificity, value specificity
for s in score_necessity_B score_necessity_mp score_occlusion_B score_occlusion_mp; do
  run "$CODE/occlusion" "$PY $s.py"
done
run "$CODE" "$PY score_valuespec_sp.py"
run "$CODE" "$PY score_valuespec_mp.py"

# Experiment 2 — Lookup vs Compute endpoints (full-generation endpoint and log-prob injection curve)
run "$CODE" "$PY score_cf_pairs_unified_rev.py"   # single-page 739 + multi-page 190
run "$CODE" "$PY score_paired_injection.py"
run "$CODE" "$PY score_cfmp2.py"                  # multi-page 821
run "$CODE" "$PY score_cfmp2_ka2.py"              # both metrics on the same statistics

# Experiment 3 — layer necessity by operation x page structure
run "$CODE/mpreq" "$PY score_c0_opxpage.py"

# Experiment 4 — probes and recomputation
for s in probes_summary probes_summary_mp821 score_recompute_common score_recompute_common8; do
  run "$CODE/common" "$PY $s.py"
done
run "$CODE/common" "$PY score_probe_after_endpoint.py"         # multi-page set = 190
run "$CODE/common" "$PY score_probe_after_endpoint.py mp821"   # multi-page set = 821
