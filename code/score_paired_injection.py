#!/usr/bin/env python3
import json, re, random, statistics, unicodedata
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
BASE = Path(__file__).resolve().parent
SP_PATCH = {"qwen35_9b": ["cf_patch_cf2_qwen35_9b.jsonl", "cf_patch_cf3_qwen35_9b.jsonl", "cf5_patch_qwen35_9b.jsonl", "cf4_patch_qwen35_9b.jsonl"],
            "qwen3vl_8b": ["cf_patch_qwen3vl_8b.jsonl", "cf5_patch_qwen3vl_8b.jsonl"], "qwen25_7b": ["cf_patch_qwen25_7b.jsonl", "cf_patch_qwen25_7b_ext.jsonl", "cf5_patch_qwen25_7b.jsonl"],
            "gemma4_12b": ["cf_patch_gemma4_12b.jsonl", "cf_patch_gemma4_12b_ext.jsonl", "cf5_patch_gemma4_12b.jsonl"], "ministral3_8b": ["cf_patch_ministral3_8b.jsonl"], "llava_ov2_8b": ["cf_patch_llava_ov2_8b.jsonl"],
            "qwen3vl_32b": ["cf_patch_qwen3vl_32b.jsonl", "cf5_patch_qwen3vl_32b.jsonl"], "gemma4_31b": ["cf_patch_gemma4_31b.jsonl", "cf5_patch_gemma4_31b.jsonl"]}
MP_PATCH = {t: [f"cfmp_patch_{t}.jsonl"] for t in SP_PATCH}
import os as _os
if _os.environ.get("MP821") == "1":
    MP_PATCH = {t: [f"cfmp_patch_{t}.jsonl", f"cfmp2_patch_{t}.jsonl"] for t in SP_PATCH}
import os
RING1 = os.environ.get("SP_RING1") == "1"
if RING1: SP_PATCH = {t: ["ring1_sp/" + f for f in fs] for t, fs in SP_PATCH.items()}
SFX = "_spring1" if RING1 else ("_mp821" if os.environ.get("MP821") == "1" else "")
NAMES = {"qwen35_9b": "Qwen3.5-9B (32)", "qwen3vl_8b": "Qwen3-VL-8B (36)", "qwen25_7b": "Qwen2.5-VL-7B (28)", "gemma4_12b": "Gemma 4 12B (48)", "ministral3_8b": "Ministral 3 8B (34)", "llava_ov2_8b": "LLaVA-OV2 8B (36)", "qwen3vl_32b": "Qwen3-VL-32B (64)", "gemma4_31b": "Gemma 4 31B (60)"}
GAP, THR = 0.5, 0.5
def clip(x): return max(-0.25, min(1.25, x))
def loss_layer(curve, thr=THR):
    o = [l for l, v in enumerate(curve) if v > thr]; return o[-1] if o else None
def boot(vals_by_doc, fn=np.mean, B=5000, seed=0):
    docs = list(vals_by_doc); rng = random.Random(seed); out = []
    for _ in range(B):
        s = [v for d in (rng.choice(docs) for _ in docs) for v in vals_by_doc[d]]
        if s: out.append(fn(s))
    return np.percentile(out, 2.5), np.percentile(out, 97.5)
def analyze(files):
    by = defaultdict(dict)
    for f in files:
        p = BASE / f
        if not p.exists(): continue
        for l in open(p):
            r = json.loads(l); by[(r["qid"], r["role"])][r["layer"]] = r["lp_donor_tok"]
    L = max(l for d in by.values() for l in d) + 1
    D = {}
    for (q, role), d in by.items():
        if -1 not in d or -2 not in d: continue
        gap = d[-2] - d[-1]
        if gap <= GAP: continue
        D[(q, role)] = [clip((d[l] - d[-1]) / gap) if l in d else np.nan for l in range(L)]
    return D, L
md = ['# Paired data rescored with the injection curve (same metric as layer necessity)', "",
      'D_p(l) = fraction of the donor-answer first-token log-prob shift between base and full donor when the B-cell state is replaced by the donor state at layer l (0-1, clipped to -0.25..1.25). usable = gap > 0.5 nat. Vanishing layer = last layer where the curve exceeds 0.5 (relative depth = layer/(L-1)). Per-item dL(0.5) = Lookup - Compute on the same item (sign test excluding 0, document-clustered bootstrap CI).', ""]
md.append('| model | condition | usable Lookup / Compute / both | curve vanishing layer Lookup (rel.) | curve vanishing layer Compute (rel.) | per-item median Lookup [IQR] / Compute [IQR] | dL(0.5) Compute first:same:reversed | mean dL [doc CI] | sign test p | ref: modal full-generation endpoint Lookup / Compute |')
md.append("|---|---|---|---|---|---|---|---|---|---|")
REF = {"qwen35_9b": "23 / 19", "qwen3vl_8b": "24 / 23", "qwen25_7b": "22 / 21", "gemma4_12b": "35 / 35", "ministral3_8b": "24·28 / 25", "llava_ov2_8b": "24 / 23", "qwen3vl_32b": "52 / 52", "gemma4_31b": "41 / 41"}
curves = {}
for tag in SP_PATCH:
    for cond, files in (("SP", SP_PATCH[tag]), ("MP", MP_PATCH[tag])):
        D, L = analyze(files)
        lk = {q: v for (q, r), v in D.items() if r == "lookup_b"}; cp = {q: v for (q, r), v in D.items() if r == "compute"}
        both = sorted(set(lk) & set(cp))
        if not lk or not cp: continue
        cl = [np.nanmean([lk[q][l] for q in lk]) for l in range(L)]; cc = [np.nanmean([cp[q][l] for q in cp]) for l in range(L)]
        ll, lc = loss_layer(cl), loss_layer(cc); curves[(tag, cond)] = (cl, cc, L)
        pl = {q: (loss_layer(lk[q]) if loss_layer(lk[q]) is not None else -1) for q in both}; pc = {q: (loss_layer(cp[q]) if loss_layer(cp[q]) is not None else -1) for q in both}
        dl = {q: pl[q] - pc[q] for q in both}; pos = sum(v > 0 for v in dl.values()); neg = sum(v < 0 for v in dl.values()); zero = sum(v == 0 for v in dl.values())
        p = binomtest(pos, pos + neg).pvalue if pos + neg else float("nan")
        bd = defaultdict(list)
        for q, v in dl.items(): bd[q.split("_")[1]].append(v)
        lo, hi = boot(bd) if bd else (float("nan"), float("nan"))
        def iqr(vals): s = sorted(vals); return f"{statistics.median(s):g} [{s[len(s)//4]}–{s[(3*len(s))//4]}]" if s else "—"
        md.append(f"| {NAMES[tag]} | {cond} | {len(lk)} / {len(cp)} / {len(both)} | {ll} ({(ll or 0)/(L-1):.2f}) | {lc} ({(lc or 0)/(L-1):.2f}) | {iqr(list(pl.values()))} / {iqr(list(pc.values()))} | {pos}:{zero}:{neg} | {np.mean(list(dl.values())):.2f} [{lo:.2f}, {hi:.2f}] | {p:.2g} | {REF[tag] if cond == 'SP' else ''} |")
md += ["", '## Next to the layer-necessity vanishing layers (discriminative token, same threshold 0.5), Qwen models', "", '| model | paired Lookup (injection curve) | SP lookup 58 (layer necessity) | paired Compute (injection curve) | SP compute 42 (layer necessity) | L3big caption lookup 155 | L3big two-page operands 95 |', "|---|---|---|---|---|---|---|"]
NEC = {"qwen35_9b": (19, 15, 19, 19), "qwen3vl_8b": (23, 18, 24, 23), "qwen25_7b": (22, 22, 22, 22)}
for tag in ("qwen35_9b", "qwen3vl_8b", "qwen25_7b"):
    cl, cc, L = curves[(tag, "SP")]; n = NEC[tag]
    md.append(f"| {NAMES[tag]} | {loss_layer(cl)} | {n[0]} | {loss_layer(cc)} | {n[1]} | {n[2]} | {n[3]} |")
md += ["", '## Mean D_p by relative depth (SP; 0.3/0.5/0.6/0.7/0.8/0.9)', "", '| model | role | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |', "|---|---|---|---|---|---|---|---|"]
for tag in SP_PATCH:
    if (tag, "SP") not in curves: continue
    cl, cc, L = curves[(tag, "SP")]
    for role, c in (('Lookup', cl), ('Compute', cc)):
        md.append(f"| {NAMES[tag]} | {role} | " + " | ".join(f"{c[round(x*(L-1))]:.2f}" for x in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)) + " |")
json.dump({f"{t}_{c}": {"lookup": v[0], "compute": v[1], "L": v[2]} for (t, c), v in curves.items()}, open(BASE / f"paired_injection_curves{SFX}.json", "w"))
(BASE / f"paired_injection_stats{SFX}.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
