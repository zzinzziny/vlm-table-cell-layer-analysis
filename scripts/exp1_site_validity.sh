#!/usr/bin/env bash
# Experiment 1 — is the cell's visual-token site the right place to intervene?
#   necessity : replace the B-cell tokens (or a matched control cell) with the mean image token at layer 0
#               and check whether the answer collapses                      (single-page 739 / multi-page 821)
#   site specificity : ablate every cell of the page one at a time and rank the answer-cell B
#               (50 items per model, single-page / multi-page)
# Value specificity (layer-0 donor patching) comes from the Experiment 2 runs; see scripts/score_all.sh.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/exp1_site_validity.sh <model_tag>
set -euo pipefail
source "$(dirname "$0")/models.sh"
tag=${1:?model tag}; model_info "$tag"
D=$CODE/occlusion
ARGS="--tag $tag --model $MODEL --geo $GEO --backend $BACKEND $NT"

run "$D" "$PYBIN run_necessity_B.py  $ARGS $(lim)"            # -> necessity_<tag>.jsonl
run "$D" "$PYBIN run_necessity_mp.py $ARGS $(lim)"            # -> necessity_mp_<tag>.jsonl
run "$D" "$PYBIN run_cell_occlusion_batch.py $ARGS --bs $BS"  # -> occl_main_<tag>.jsonl
run "$D" "$PYBIN run_cell_occlusion_mp.py    $ARGS --bs $BS"  # -> occl_mp_<tag>.jsonl
