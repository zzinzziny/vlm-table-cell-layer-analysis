# Layer-by-layer use of table-cell representations in VLMs

Layer-by-layer analysis of how vision-language models use the visual tokens of a single table cell when answering questions about document-page images.

## Research summary

We locate each table cell's visual tokens inside the decoder and intervene on them layer by layer.
For one and the same cell value we separate three properties:

1. **Decodability**: whether the value can be read linearly from the hidden state.
2. **Causal influence**: whether editing that site changes the prediction.
3. **Operand reuse**: whether the value is picked up again as an operand by a different task.

### Models

| tag | model | token geometry | backend |
|---|---|---|---|
| `qwen35_9b` | Qwen/Qwen3.5-9B | `qwen35` (32 px) | `qwen` |
| `qwen3vl_8b` | Qwen/Qwen3-VL-8B-Instruct | `qwen35` (32 px) | `qwen` |
| `qwen25_7b` | Qwen/Qwen2.5-VL-7B-Instruct | `qwen25` (28 px) | `qwen` |
| `ministral3_8b` | mistralai/Ministral-3-8B-Instruct-2512-BF16 | `ministral3` (28 px) | `mistral3` |
| `llava_ov2_8b` | lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct | `llava_ov2` (28 px) | `llava_ov2` |
| `gemma4_12b` | google/gemma-4-12B-it | `gemma4` (48 px soft tokens) | `gemma` |
| `qwen3vl_32b` | Qwen/Qwen3-VL-32B-Instruct | `qwen35` (32 px) | `qwen` |
| `gemma4_31b` | google/gemma-4-31B-it | `gemma4` (48 px soft tokens) | `gemma` |

## Repository layout

```
code/                 experiment and scoring code
  *.py                  paired-pool builders, Lookup/Compute gate + patch runners, Experiment 2 scorers
  occlusion/            Experiment 1: necessity and site-specificity runners and scorers
  mpreq/                Experiment 3: layer-necessity runners and scorers
  optype/               Experiment 3: operation-type labels for the 918 questions
  common/               shared item registry; Experiment 4: probes and recomputation
  handoff2/             second-operand (A2) selection for the recomputation control
  external/qa_enrichment/   builds cell -> visual-token labels from PubTables-v2 annotations
data/                 question items and the page images they point to (no results)
scripts/              one script per experiment, plus scoring
link_data.sh          links data/ into code/ so the scripts find their inputs
requirements.txt
```

The code keeps relative paths from the original working tree (for example `common/items_sp_qwen35.jsonl`). `link_data.sh` symlinks every file in `data/` to the same path under `code/`, so no code has to change. Run outputs are written next to the scripts under `code/`.

## Setup

```bash
pip install -r requirements.txt                 # Qwen, Ministral, LLaVA runs and all scoring
# Gemma-4 needs transformers 5.17 in a separate environment (see requirements.txt)

export PY=$(which python)
export PY_GEMMA=/path/to/gemma-env/bin/python
export HF_HOME=~/.cache/huggingface             # download the eight models first; runs use HF_HUB_OFFLINE=1

# Donor / base renders (1.8 GB), hosted on OSF:
#   https://osf.io/2r9vs/overview?view_only=9baf05510c2d402db3bbdc4ad56f6ced
wget -O cf2_images.tar "https://osf.io/download/6ab5721bcecfa55301d3296c/?view_only=9baf05510c2d402db3bbdc4ad56f6ced"
echo "de976de49082d1fc5073599ccff143008830ed0fba1a39a752d6a5f5caf53b25  cf2_images.tar" | sha256sum -c
tar xf cf2_images.tar -C data/                  # -> data/cf2_images/ (2,851 PNGs)

bash link_data.sh                               # bash link_data.sh --clean removes the links
```

## Running the experiments

```bash
export CUDA_VISIBLE_DEVICES=0
for m in qwen35_9b qwen3vl_8b qwen25_7b ministral3_8b llava_ov2_8b gemma4_12b qwen3vl_32b gemma4_31b; do
  bash scripts/exp1_site_validity.sh   $m
  bash scripts/exp2_lookup_compute.sh  $m
  bash scripts/exp3_layer_necessity.sh $m
  bash scripts/exp4_reuse.sh           $m
done
bash scripts/score_all.sh                        # CPU only, after all models have finished
```

GPU memory used in the original runs: about 72 GB for Qwen3-VL-32B and Gemma-4-31B, 32 GB for Gemma-4-12B, and 24 GB for the other models.

### Experiment 1: site validity (`scripts/exp1_site_validity.sh`)

| step | code | output |
|---|---|---|
| necessity (single-page 739 / multi-page 821) | `occlusion/run_necessity_B.py`, `run_necessity_mp.py` | `necessity[_mp]_<tag>.jsonl` |
| site specificity (50 items per model) | `occlusion/run_cell_occlusion_batch.py`, `run_cell_occlusion_mp.py` | `occl_main_<tag>.jsonl`, `occl_mp_<tag>.jsonl` |
| scoring | `occlusion/score_necessity_{B,mp}.py`, `occlusion/score_occlusion_{B,mp}.py` | collapse rates vs control cell (McNemar); rank of B among all cells |
| value specificity | `score_valuespec_sp.py`, `score_valuespec_mp.py` (read the Experiment 2 patch files) | share of answers that follow the donor value after a layer-0 swap |

The target items and cells per model are fixed in `data/occlusion/*_items.json` and `*_cells.json` (made by `occlusion/prep_*.py` from the Experiment 2 gates).

### Experiment 2: Lookup vs Compute endpoints (`scripts/exp2_lookup_compute.sh`)

| step | code | output |
|---|---|---|
| gate: 5 greedy generations; keep items answered correctly on base and donor | `run_cf_gates_m.py` (Qwen), `run_cf_gates_gemma.py` (Gemma), `run_cf_mp_gates.py` (multi-page pools and Ministral / LLaVA), `run_cf_gates.py` (first Qwen3.5-9B batches) | `*_gates_<tag>.jsonl`, `*_pass_patch.txt` |
| layer-wise donor patching of the B-cell state | `run_cf_patch_m.py`, `run_cf_patch_gemma.py`, `run_cf_mp_patch.py`, `run_cf_patch.py` | `*_patch_<tag>.jsonl` (answer and donor log-prob per layer) |
| endpoint difference ΔL = L(Lookup) − L(Compute), document-clustered bootstrap and permutation tests | `score_cf_pairs_unified_rev.py` (full-generation endpoint), `score_paired_injection.py` (log-prob injection curve D_p(l)) | `paired_unified_stats_rev.md`, `paired_injection_stats.md` |
| multi-page pool 821, both metrics on the same items | `score_cfmp2.py`, `score_cfmp2_ka2.py` | `paired_cfmp2_stats.md`, `paired_cfmp2_ka2.md` |

Shared model loading and input construction for all backends is in `cf_mp_common.py`.

### Experiment 3: layer necessity by question type (`scripts/exp3_layer_necessity.sh`)

| step | code | output |
|---|---|---|
| layer necessity D(l) of the evidence cells | `mpreq/run_layer_necessity.py` (Qwen), `mpreq/run_layer_necessity_generic.py` (Gemma, Ministral, LLaVA) | `layer_necessity_<set>_<tag>.jsonl` |
| operation labels (rules + GPT-5, already provided) | `optype/build_c0_items.py`, `label_rules.py`, `label_gpt5.py`, `label_c0_gpt5.py`, `page_structure_c0.py` | `data/optype/c0_labels_final.jsonl` |
| endpoint layer per operation × page structure | `mpreq/score_c0_opxpage.py` (uses `mpreq/score_l3big.py`, `score_ablation.py`) | `score_c0_opxpage.{md,json}` |

Item sets: `lookup58` (single-page lookup), `compute` (single-page arithmetic), `mpreq` (multi-page required), `l3big` (387 table-reasoning items), `v2natfull` (368 benchmark items), and `v2natext` (175 extension). Together the first five sets hold the 918 questions.

### Experiment 4: decodability and reuse (`scripts/exp4_reuse.sh`)

| step | code | output |
|---|---|---|
| hidden-state capture at cell and text positions (no intervention) | `common/run_probe_capture_common.py` | `probe_feats_{sp,mp}_<tag>/` |
| document-level 5-fold linear probes with shuffled-label control | `common/train_probes.py` | `probes_{sp,mp,mp821}_<tag>.{md,npz}` |
| probe summaries; accuracy after the Compute endpoint | `common/probes_summary.py`, `probes_summary_mp821.py`, `score_probe_after_endpoint.py` (reads the endpoints from `score_recompute_common.md`) | `probes_summary*.md`, `probe_after_endpoint*.md` |
| recomputation control | `common/run_recompute_common.py` | `recompute_{sp,mp}_<tag>.jsonl` |
| scoring (6-run gate; 8-run gate shared by the three difference sources) | `common/score_recompute_common.py`, `score_recompute_common8.py` | `score_recompute_common.md`, `recompute_common8.{md,json}` |

Run the scripts in `code/common/` from that directory. `scripts/*.sh` already does this.

## Data

Every file in `data/` is a question item or an image that an item points to. An item holds the question strings (`lookup_a`, `lookup_b`, `compute`), the gold answer, base and donor image paths, and the answer-cell bounding box with its visual-token indices for each model geometry (`<geo>` in file names).

| files | contents |
|---|---|
| `cf{2,3,4,5}_candidates.jsonl`, `cf_pool_<geo>.jsonl`, `cf5_pool_<geo>.jsonl`, `common/cf4_pool_qwen35.jsonl` | single-page paired pool, 739 items (100 + 150 + 160 + 329) |
| `cfmp_candidates.jsonl`, `cfmp_pool_<geo>.jsonl` | multi-page paired pool, 190 items (A and B on different pages) |
| `cfmp2_candidates.jsonl`, `cfmp2_pool_<geo>.jsonl` | multi-page extension, 631 items |
| `common/registry_{sp,mp}.jsonl`, `common/items_{sp,mp,mp2}_<geo>.jsonl` | shared registry and runner inputs for Experiment 4 (includes the second operand A2) |
| `mpreq/items_<set>_<geo>[_remap].jsonl` | Experiment 3 item sets (`_remap` = versions used for the Qwen geometries) |
| `optype/*.jsonl` | operation-type labels for the Experiment 3 questions |
| `occlusion/*_items.json`, `*_cells.json` | Experiment 1 target items and cells per model |
| `handoff2/recompute_items.jsonl` | reference A2 choices for the recomputation control |
| `external/pubtables-v2/Full Documents/test/images/` | the PubTables-v2 test page images the items use (980 pages) |
| `cf2_images/` (separate download, see Setup) | donor and base renders: the B cell redrawn with its leading digit changed |
