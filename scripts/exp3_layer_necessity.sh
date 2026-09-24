#!/usr/bin/env bash
# Experiment 3 — layer necessity of the evidence cells by question type and page structure.
# The evidence-cell state of a corrupted run is written into the clean run at one layer at a time;
# D(l) = drop in the gold-answer log-prob caused by corrupting that single layer.
# Item sets (918 in total, + 175 extension):
#   lookup58 (single-page lookup) · compute (single-page two-operand arithmetic, --sp-only) ·
#   mpreq (items that need several pages) · l3big (387 table-reasoning items) ·
#   v2natfull (368 benchmark items, all evidence pages) · v2natext (175 extension)
# Output names follow what mpreq/score_c0_opxpage.py reads.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/exp3_layer_necessity.sh <model_tag>
set -euo pipefail
source "$(dirname "$0")/models.sh"
tag=${1:?model tag}; model_info "$tag"
D=$CODE/mpreq

if [[ $BACKEND == qwen ]]; then R="run_layer_necessity.py"; else R="run_layer_necessity_generic.py --backend $BACKEND"; fi
small=( qwen35_9b qwen3vl_8b qwen25_7b )    # first three models: rerun with the discriminative-token metric (_v2 files)
is_small() { [[ " ${small[*]} " == *" $tag "* ]]; }

for s in lookup58 compute mpreq l3big v2natfull v2natext; do
  items=items_${s}_$GEO.jsonl; out=$s; sp=""
  [[ $s == compute ]] && { out=compute_sp; sp="--sp-only"; }
  # Qwen2.5-VL / Qwen3.5-geometry versions of the three small sets are the *_remap files
  [[ $s =~ ^(lookup58|compute|mpreq)$ && ( $GEO == qwen35 || $GEO == qwen25 ) ]] && items=items_${s}_${GEO}_remap.jsonl
  suffix=""
  { [[ $s =~ ^(lookup58|compute|mpreq)$ ]] && is_small; } && suffix=_v2
  [[ $s == l3big && $tag == qwen35_9b ]] && suffix=_v2
  run "$D" "$PYBIN $R --model $MODEL $NT $sp --items $items --out layer_necessity_${out}_${tag}${suffix}.jsonl --resume $(lim)"
done
