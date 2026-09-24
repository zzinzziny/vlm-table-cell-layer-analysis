#!/usr/bin/env python3
import json, sys, ast, collections, statistics
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parent
sys.argv, _a = ["score_l3big"], sys.argv
from score_l3big import make_accessors  # noqa
sys.argv = _a
usable, D, *_ = make_accessors("disc")
MODELS = [("qwen35_9b", 32), ("qwen3vl_8b", 36), ("qwen25_7b", 28), ("gemma4_12b", 48), ("ministral3_8b", 34), ("llava_ov2_8b", 36), ("qwen3vl_32b", 64), ("gemma4_31b", 60)]
QWEN = {"qwen35_9b", "qwen3vl_8b", "qwen25_7b"}; REP_FIXED = {"qwen35_9b": 19, "qwen3vl_8b": 24, "qwen25_7b": 22}
OP = {"lookup": "Lookup", "lookup_cond": "Lookup", "multi_answer": "Lookup", "lookup_v2": "Lookup",
      "join": "Multi-step", "chain": "Multi-step", "multi_hop_steps": "Multi-step",
      "split_pair_ops": "Compute", "two_cell": "Compute", "answer_plus_operands": "Compute",
      "aggregate": "Aggregate", "cross_table_compare": "Compare/Select", "rank_candidates": "Compare/Select", "extremes": "Compare/Select", "split_top2": "Compare/Select"}
def load(p): return [json.loads(l) for l in open(BASE / p)] if (BASE / p).exists() else None
def npg(v):
    e = v.get("evidence_pages"); e = ast.literal_eval(e) if isinstance(e, str) else e
    return "SP" if e and len(e) == 1 else "MP"
L3 = {i["qid"]: i for i in load("items_l3big_qwen35.jsonl")}; V2 = {i["qid"]: i for i in load("items_v2natfull_qwen35.jsonl")}
MQ = {i["qid"]: i for i in load("items_mpreq_qwen35_remap.jsonl")}
def f_sp(s, t): return f"layer_necessity_{s}_{t}_v2.jsonl" if t in QWEN else f"layer_necessity_{s}_{t}.jsonl"
def f_l3(t): return {"qwen35_9b": "layer_necessity_l3big_qwen35_9b_v2.jsonl"}.get(t, f"layer_necessity_l3big_{t}.jsonl")
LAB = {(json.loads(l)["pool"], json.loads(l)["qid"]): json.loads(l) for l in open(BASE.parent / "optype" / "c0_labels_final.jsonl")}
def lab(pool): return lambda r: ((LAB[(pool, r["qid"])]["final_op"], LAB[(pool, r["qid"])]["page"]) if (pool, r["qid"]) in LAB else None)
SOURCES = [("lookup58", lambda t: f_sp("lookup58", t), lab("lookup58")), ("compute_sp", lambda t: f_sp("compute_sp", t), lab("compute_sp")),
           ("l3big", f_l3, lab("l3big")), ("v2natfull", lambda t: f"layer_necessity_v2natfull_{t}.jsonl", lab("v2natfull")), ("mpreq", lambda t: f_sp("mpreq", t), lab("mpreq"))]
CELLS = [(o, p) for o in ("SELECT", "JOIN", "COMPUTE", "AGGREGATE", "COMPARE", "MULTI_CELL", "OTHER") for p in ("SP", "MP")]
def clip(x): return max(-0.25, min(1.25, x))
def last(c): o = [l for l, v in enumerate(c) if v > 0.5]; return o[-1] if o else None
curves = {(c, t): [] for c in CELLS for t, _ in MODELS}; seen = {(c, t): 0 for c in CELLS for t, _ in MODELS}; missing = collections.defaultdict(list)
for t, L in MODELS:
    for name, ff, cls in SOURCES:
        rows = load(ff(t))
        if rows is None: missing[t].append(name); continue
        for r in rows:
            k = cls(r)
            if not k or k not in curves.get((k, t), [None]) and (k, t) not in curves: continue
            seen[(k, t)] += 1
            if usable(r) and "evidence_visual" in r.get("inject_disc", {}):
                curves[(k, t)].append([clip(D(r, "evidence_visual", l)) for l in range(L)])
rep, rep_med, res = {}, {}, {}
for t, L in MODELS:
    for c in CELLS:
        cs = curves[(c, t)]
        if not cs: res[(c, t)] = None; continue
        mc = np.mean(cs, axis=0); cl = last(mc); per = [x for x in (last(v) for v in cs) if x is not None]
        res[(c, t)] = {"usable": len(cs), "items": seen[(c, t)], "vanish": cl, "per_item": per, "tail": float(np.mean(mc[cl + 1:cl + 4])) if cl is not None and cl + 1 < L else None}
    allc = [cv for c in CELLS for cv in curves[(c, t)]]
    rep[t] = last(np.mean(allc, axis=0)) if allc else None
    pooled = [x for c in CELLS if res[(c, t)] for x in res[(c, t)]["per_item"]]
    rep_mode = collections.Counter(pooled).most_common(1)[0][0] if pooled else None
    vs = [res[(c, t)]["vanish"] for c in CELLS if res[(c, t)] and res[(c, t)]["vanish"] is not None and res[(c, t)]["usable"] >= 20]
    rep_med[t] = statistics.median(vs) if vs else None
for (c, t), v in res.items():
    if v: v["conc"] = 100 * sum(abs(x - rep[t]) <= 2 for x in v["per_item"]) / len(v["per_item"]) if v["per_item"] and rep[t] is not None else None; v.pop("per_item")
out = ['# Operation x page structure (score_c0_opxpage.py)', "", 'L_rep (vanishing layer of the overall mean curve): ' + str(rep), 'L_rep (median of row vanishing layers, rows with usable >= 20): ' + str(rep_med), "",
       'cell = curve vanishing layer [usable / labelled items]; missing sources: ' + json.dumps({t: m for t, m in missing.items()}), "", '| operation | page | ' + " | ".join(t for t, _ in MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
for c in CELLS:
    out.append(f"| {c[0]} | {c[1]} | " + " | ".join("—" if not res[(c, t)] else f"{res[(c, t)]['vanish']} [{res[(c, t)]['usable']}/{res[(c, t)]['items']}]" for t, _ in MODELS) + " |")
(BASE / "score_c0_opxpage.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
json.dump({"rep": rep, "rep_median": rep_med, "rep_mode_legacy": {}, "missing": missing, "cells": {f"{c[0]}|{c[1]}|{t}": v for (c, t), v in res.items()}}, open(BASE / "score_c0_opxpage.json", "w"), indent=1)
