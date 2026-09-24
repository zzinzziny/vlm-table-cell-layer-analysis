#!/usr/bin/env bash
# Experiment 2 — Lookup vs Compute: the last layer at which the B-cell state still drives the answer.
# For every item of the paired pools:
#   gate  : 5 greedy generations (base Lookup-A / Lookup-B / Compute, donor Lookup-B' / Compute'); keep all-correct items
#   patch : copy the donor (B') run's B-cell visual state into the base run, one layer at a time,
#           and record the generated answer and the donor-answer log-prob
# Pools: single-page 739 (cf2+cf3+cf4+cf5) and multi-page 821 (cfmp 190 + cfmp2 631).
# Output file names are the ones the scorers read.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/exp2_lookup_compute.sh <model_tag>
set -euo pipefail
source "$(dirname "$0")/models.sh"
tag=${1:?model tag}; model_info "$tag"
C=$CODE
M="--model $MODEL $NT"

sp_pair() {  # sp_pair <gate runner> <patch runner> <pool> <gates out> <pass prefix> <patch out>
  run "$C" "$PYBIN $1 $M --cands $3 --out $4 --pass-prefix $5 --resume $(lim)"
  run "$C" "$PYBIN $2 $M --plan $3 --qids $5_pass_patch.txt --out $6 $(lim)"
}
mp_pair() {  # mp_pair <pool> <set name>
  local be=""; [[ $BACKEND != qwen ]] && be="--backend $BACKEND"
  run "$C" "$PYBIN run_cf_mp_gates.py $M $be --cands $1 --out $2_gates_$tag.jsonl --pass-prefix $2_$tag --resume $(lim)"
  run "$C" "$PYBIN run_cf_mp_patch.py $M $be --plan $1 --qids $2_${tag}_pass_patch.txt --out $2_patch_$tag.jsonl $(lim)"
}

# ---- single-page pool (739) ----
case $tag in
  qwen35_9b)   # the 9B was run pool by pool; run_cf_gates.py / run_cf_patch.py are the 9B-only first versions
    run "$C" "$PYBIN run_cf_gates.py --cands cf2_candidates.jsonl --out cf2_gates.jsonl --pass-prefix cf2 --resume $(lim)"
    run "$C" "$PYBIN run_cf_patch.py --plan cf2_candidates.jsonl --qids cf2_pass_patch.txt --out cf_patch_cf2_qwen35_9b.jsonl $(lim)"
    run "$C" "$PYBIN run_cf_gates.py --cands cf3_candidates.jsonl --out cf3_gates.jsonl --pass-prefix cf3 --resume $(lim)"
    run "$C" "$PYBIN run_cf_patch.py --plan cf3_candidates.jsonl --qids cf3_pass_patch.txt --out cf_patch_cf3_qwen35_9b.jsonl $(lim)"
    sp_pair run_cf_gates_m.py run_cf_patch_m.py common/cf4_pool_qwen35.jsonl cf4_gates_qwen35_9b.jsonl cf4_qwen35_9b cf4_patch_qwen35_9b.jsonl
    sp_pair run_cf_gates_m.py run_cf_patch_m.py cf5_pool_qwen35.jsonl cf5_gates_qwen35_9b.jsonl cf5_qwen35_9b cf5_patch_qwen35_9b.jsonl ;;
  qwen3vl_8b|qwen25_7b|qwen3vl_32b|gemma4_12b|gemma4_31b)
    g=run_cf_gates_m.py; p=run_cf_patch_m.py
    [[ $BACKEND == gemma ]] && { g=run_cf_gates_gemma.py; p=run_cf_patch_gemma.py; }
    sp_pair $g $p cf_pool_$GEO.jsonl  cf_gates_$tag.jsonl  cfm_$tag cf_patch_$tag.jsonl     # cf2+cf3+cf4 (410)
    sp_pair $g $p cf5_pool_$GEO.jsonl cf5_gates_$tag.jsonl cf5_$tag cf5_patch_$tag.jsonl    # cf5 (329)
    # the original runs split these two models' patching into a first batch and an "_ext" batch;
    # the scorers list both files, so keep an empty _ext file
    [[ $tag == qwen25_7b || $tag == gemma4_12b ]] && run "$C" "touch cf_patch_${tag}_ext.jsonl" ;;
  ministral3_8b|llava_ov2_8b)   # these pools already hold all 739 items; run with the multi-page runner
    mp_pair cf_pool_$GEO.jsonl cf ;;
esac

# ---- multi-page pools (190 + 631) ----
mp_pair cfmp_pool_$GEO.jsonl  cfmp
mp_pair cfmp2_pool_$GEO.jsonl cfmp2
