#!/usr/bin/env python3
import json, random, statistics
from collections import defaultdict
from pathlib import Path
D = Path(__file__).parent
MODELS = ["qwen35_9b", "qwen3vl_8b", "qwen25_7b", "ministral3_8b", "llava_ov2_8b", "gemma4_12b", "qwen3vl_32b", "gemma4_31b"]
def item(r):
    cs = r["cells"]; Bs = [c for c in cs if c["role"] == "B"]
    if len(Bs) != 1: return None
    b = Bs[0]; bt = set(b["tok"])
    el = [c for c in cs if c is b or not (set(c["tok"]) & bt)]
    N = len(el)
    if N < 2: return None
    top = max(el, key=lambda c: c["delta"])
    rank = 1 + sum(1 for c in el if c["delta"] > b["delta"])
    tot = sum(max(c["delta"], 0) for c in el)
    return {"top1": float(top is b), "pct": 1 - (rank - 1) / (N - 1), "share": (max(b["delta"], 0) / tot) if tot > 0 else None, "chance": 1 / N, "N": N}
def boot(byd, B=10000, seed=0):
    rng = random.Random(seed); docs = list(byd); est = []
    for _ in range(B):
        s = [v for d in (rng.choice(docs) for _ in docs) for v in byd[d]]; est.append(sum(s) / len(s))
    est.sort(); return est[int(.025 * B)], est[int(.975 * B) - 1]
out = {}; lines = ['# Site specificity, multi-page: target cell B (score_occlusion_mp.py)', "",
         'Pool = 50 items per model chosen from MP 821 (20 shared by all 8 models + per-model fill). Candidate cells = the cells on **all pages** the item uses.', "", '| model | task | items | top1 [CI] | B percentile [CI] | share [CI] | random 1/N | mean N |', "|---|---|---|---|---|---|---|---|"]
for t in MODELS:
    f = D / f"occl_mp_{t}.jsonl"
    if not f.exists(): continue
    rows = [json.loads(l) for l in open(f)]
    for task in ("lookup_b", "compute"):
        ms = [(r["qid"].split("_")[1], item(r)) for r in rows if r["task"] == task]; ms = [(d, m) for d, m in ms if m]
        if not ms: continue
        res = {"n": len(ms), "chance": statistics.mean(m["chance"] for _, m in ms), "N_mean": statistics.mean(m["N"] for _, m in ms)}
        for k in ("top1", "pct", "share"):
            byd = defaultdict(list)
            for d, m in ms:
                if m[k] is not None: byd[d].append(m[k])
            vals = [v for vs in byd.values() for v in vs]; res[k] = [statistics.mean(vals), *boot(byd)]
        out[f"{t}/{task}"] = res
        lines.append(f"| {t} | {task} | {len(ms)} | {res['top1'][0]:.2f} [{res['top1'][1]:.2f}, {res['top1'][2]:.2f}] | {res['pct'][0]:.2f} [{res['pct'][1]:.2f}, {res['pct'][2]:.2f}] | {res['share'][0]:.2f} [{res['share'][1]:.2f}, {res['share'][2]:.2f}] | {res['chance']:.3f} | {res['N_mean']:.0f} |")
(D / "score_occlusion_mp.md").write_text("\n".join(lines) + "\n"); json.dump(out, open(D / "score_occlusion_mp.json", "w"), indent=1); print("\n".join(lines))
