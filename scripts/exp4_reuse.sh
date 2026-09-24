#!/usr/bin/env bash
# Experiment 4 — is a still-decodable cell value reused as an operand?
#   probes    : store per-layer hidden states at the B-cell tokens and text positions (no intervention),
#               then train document-level 5-fold linear probes (single-page 739; multi-page 190 and 190+631)
#   recompute : inject the B->B' activation difference at the B-cell tokens or at all text positions,
#               one layer at a time, and check whether the answer becomes f(A, B')
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/exp4_reuse.sh <model_tag>
set -euo pipefail
source "$(dirname "$0")/models.sh"
tag=${1:?model tag}; model_info "$tag"
D=$CODE/common
M="--model $MODEL --backend $BACKEND $NT"

# probes (training is CPU only)
run "$D" "$PYBIN run_probe_capture_common.py --items items_sp_$GEO.jsonl $M --outdir probe_feats_sp_$tag --resume $(lim)"
run "$D" "$PY train_probes.py --featdir probe_feats_sp_$tag --out probes_sp_$tag --layers step2 --seed 0"
run "$D" "$PYBIN run_probe_capture_common.py --items items_mp_$GEO.jsonl $M --outdir probe_feats_mp_$tag --resume $(lim)"
run "$D" "$PY train_probes.py --featdir probe_feats_mp_$tag --out probes_mp_$tag --layers step2 --seed 0"
run "$D" "$PYBIN run_probe_capture_common.py --items items_mp2_$GEO.jsonl $M --outdir probe_feats_mp_$tag --resume $(lim)"   # adds the 631 extension
run "$D" "$PY train_probes.py --featdir probe_feats_mp_$tag --out probes_mp821_$tag --layers step2 --seed 0"

# recomputation control
for set in sp mp; do
  run "$D" "$PYBIN run_recompute_common.py --items items_${set}_$GEO.jsonl --geometry $GEO $M --out recompute_${set}_$tag.jsonl --resume $(lim)"
done
