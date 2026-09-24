#!/usr/bin/env python3
import json, sys, glob, statistics, collections
files = sys.argv[1:] or sorted(glob.glob("recompute_*.jsonl"))
SITES = ("b_cell", "text_all", "last"); COMBOS = ("d2c", "d2d", "lkd", "sac", "d3c", "d3d")
lines = []
def emit(x=""): lines.append(x); print(x)
for f in files:
    rows = [json.loads(l) for l in open(f)]
    n = len(rows); pos = sum(1 for r in rows if str(r.get("skip", "")).startswith("position")); gate = sum(1 for r in rows if r.get("skip") == "gate")
    ok = [r for r in rows if not r.get("skip")]; all8 = sum(1 for r in ok if r.get("gate_all8"))
    L = ok[0]["n_layers"] if ok else 0
    emit(f"## {f}: items {n} - locating failed {pos} - gate (core6) failed {gate} - **passed {len(ok)}** (all 8 runs correct {all8}) - layers {L}")
    if not ok: continue
    emit('| source@site | n | r recomputed in >=1 layer | r layer window (first-last median) | r share L0-60% / 60-80% / 80-100% | d/e donor answer copied in >=1 layer | b value overwritten in >=1 layer | alignment failures |')
    emit("|---|---|---|---|---|---|---|---|")
    for c in COMBOS:
        for s in SITES:
            k = f"{c}@{s}"; items = [r for r in ok if k in r["patch"]]
            if not items: continue
            mis = sum(1 for r in items if r["patch"][k] == "misaligned"); items = [r for r in items if r["patch"][k] != "misaligned"]
            if not items: emit(f"| {k} | 0 | | | | | | {mis} |"); continue
            seqs = [r["patch"][k] for r in items]; m = len(seqs)
            has_r = [s_ for s_ in seqs if "r" in s_]; firsts = [s_.index("r") for s_ in has_r]; lasts = [len(s_) - 1 - s_[::-1].index("r") for s_ in has_r]
            has_de = sum(1 for s_ in seqs if "d" in s_ or "e" in s_); has_b = sum(1 for s_ in seqs if "b" in s_)
            def frac(lo, hi):
                idx = range(int(lo * L), max(int(lo * L) + 1, int(hi * L))); tot = sum(1 for s_ in seqs for i in idx if i < len(s_)); hit = sum(1 for s_ in seqs for i in idx if i < len(s_) and s_[i] == "r")
                return f"{100*hit/tot:.0f}%" if tot else "-"
            win = f"{statistics.median(firsts):.0f}~{statistics.median(lasts):.0f}" if has_r else "-"
            emit(f"| {k} | {m} | {len(has_r)} ({100*len(has_r)/m:.0f}%) | {win} | {frac(0,.6)} / {frac(.6,.8)} / {frac(.8,1.0)} | {has_de} ({100*has_de/m:.0f}%) | {has_b} ({100*has_b/m:.0f}%) | {mis} |")
    emit()
open("score_recompute_common.md", "w").write("\n".join(lines) + "\n")
