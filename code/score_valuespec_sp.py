#!/usr/bin/env python3
"""Value specificity, single-page paired pool.

For gate-passed items, the B-cell visual state is replaced by the donor (B') state at layer 0 and the
answer is generated greedily. Reports the fraction of items whose answer follows the donor value
(Lookup: B', Compute: f(A, B')), plus a sanity check that patching at the last layer keeps the base answer.
Inputs: the single-page patch files written by run_cf_patch*.py / run_cf_mp_patch.py.
Output: score_valuespec_sp.md / .json
(Split out of the original score_c1_table.py; the computation is unchanged.)"""
import json, collections, re, unicodedata
from pathlib import Path
B = Path(__file__).resolve().parent
lines = []; E = lambda s="": (lines.append(s), print(s))
summary = {}
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s)).strip().lower().replace("−", "-").replace(",", "")).rstrip(".")
PATCH = {"qwen35_9b": ["cf_patch_cf2_qwen35_9b.jsonl", "cf_patch_cf3_qwen35_9b.jsonl", "cf5_patch_qwen35_9b.jsonl", "cf4_patch_qwen35_9b.jsonl"], "qwen3vl_8b": ["cf_patch_qwen3vl_8b.jsonl", "cf5_patch_qwen3vl_8b.jsonl"],
         "qwen25_7b": ["cf_patch_qwen25_7b.jsonl", "cf_patch_qwen25_7b_ext.jsonl", "cf5_patch_qwen25_7b.jsonl"], "gemma4_12b": ["cf_patch_gemma4_12b.jsonl", "cf_patch_gemma4_12b_ext.jsonl", "cf5_patch_gemma4_12b.jsonl"],
         "ministral3_8b": ["cf_patch_ministral3_8b.jsonl"], "llava_ov2_8b": ["cf_patch_llava_ov2_8b.jsonl"], "qwen3vl_32b": ["cf_patch_qwen3vl_32b.jsonl", "cf5_patch_qwen3vl_32b.jsonl"], "gemma4_31b": ["cf_patch_gemma4_31b.jsonl", "cf5_patch_gemma4_31b.jsonl"]}
E("\n## layer-0 donor follow (single-page paired, gate-passed, full generation)")
E("| model | n | LOOKUP follow | COMPUTE follow | sanity: base answer kept at last patched layer (LOOKUP) |"); E("|---|---|---|---|---|")
for t, fs in PATCH.items():
    by = collections.defaultdict(dict)
    for f in fs:
        if (B / f).exists():
            for l in open(B / f):
                r = json.loads(l); by[(r["qid"], r["role"])][r["layer"]] = r
    fol = {}
    for role in ("lookup_b", "compute"):
        v = [norm(d[0]["pred"]) == norm(d[0]["ans_donor"]) for (q, ro), d in by.items() if ro == role and 0 in d]
        fol[role] = (len(v), round(100 * sum(v) / len(v)) if v else None)
    last = [norm(d[max(d)]["pred"]) == norm(d[max(d)]["ans_base"]) for (q, ro), d in by.items() if ro == "lookup_b" and d]
    summary.setdefault("donor", {})[t] = dict(n=fol["lookup_b"][0], lookup=fol["lookup_b"][1], compute=fol["compute"][1])
    E(f"| {t} | {fol['lookup_b'][0]} | {fol['lookup_b'][1]}% | {fol['compute'][1]}% | {100*sum(last)/len(last):.0f}% (layer {max(max(d) for d in by.values())}) |")
(B / "score_valuespec_sp.md").write_text("\n".join(lines) + "\n"); (B / "score_valuespec_sp.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
