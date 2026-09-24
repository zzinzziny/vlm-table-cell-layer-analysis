#!/usr/bin/env python3
import json, glob, random, collections, sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(BASE))
from build_cf_candidates_relaxed import LABELS
MODELS = ["qwen35_9b", "qwen3vl_8b", "qwen25_7b", "gemma4_12b", "ministral3_8b", "llava_ov2_8b", "qwen3vl_32b", "gemma4_31b"]
N, PER_DOC = 50, 3
items = {json.loads(l)["qid"]: json.loads(l) for l in open(BASE / "common/items_sp_qwen35.jsonl")}
ok = {}
for m in MODELS:
    fs = [f for f in glob.glob(str(BASE / f"cf*_gates_{m}.jsonl")) if not any(x in f for x in ("cfmp", "cfsw", "cfpv", "cfrel"))]
    if m == "qwen35_9b": fs += [str(BASE / "cf2_gates.jsonl"), str(BASE / "cf3_gates.jsonl")]
    d = {}
    for f in fs:
        for l in open(f): r = json.loads(l); d[r["qid"]] = r["ok"]
    ok[m] = sorted(q for q, o in d.items() if q in items and o.get("base_lookup_b") and o.get("base_compute"))
need = {(items[q]["doc_id"], str(items[q]["table_index"])) for m in MODELS for q in ok[m]}
lab = {}
for l in open(LABELS):
    d = json.loads(l); k = (d["doc_id"], str(d["table_index"]))
    if k in need: lab[k] = d
def cells_of(q):
    it = items[q]; d = lab.get((it["doc_id"], str(it["table_index"])))
    if not d: return None
    p = [p for p in d["pages"] if p["page_id"] == it["pages"][0]["page_id"]]
    if not p: return None
    cs = [{k: c[k] for k in ("value", "row_idx", "col_idx", "is_header", "bbox")} for c in p[0]["cells"] if c.get("bbox")]
    A, B = it["operands"]
    has = lambda o: any((c["row_idx"], c["col_idx"]) == (o["row_idx"], o["col_idx"]) for c in cs)
    return {"orig_size": p[0]["orig_size"], "cells": cs} if has(A) and has(B) else None
cells = {}
common = set(ok[MODELS[0]])
for m in MODELS: common &= set(ok[m])
sel = {}
for m in MODELS:
    rng = random.Random(0); chosen, per = [], collections.Counter()
    base = sorted(q for q in common if cells_of(q))
    for q in base: chosen.append(q); per[items[q]["doc_id"]] += 1
    rest = [q for q in ok[m] if q not in common]; rng.shuffle(rest)
    for q in rest:
        if len(chosen) >= N: break
        if per[items[q]["doc_id"]] >= PER_DOC or not cells_of(q): continue
        chosen.append(q); per[items[q]["doc_id"]] += 1
    sel[m] = chosen
    for q in chosen: cells[q] = cells_of(q)
    print(m, "correct", len(ok[m]), "common", len(base), "chosen", len(chosen), "docs", len({items[q]['doc_id'] for q in chosen}), "cells mean", round(sum(len(cells[q]['cells']) for q in chosen) / len(chosen)))
out = Path(__file__).parent
json.dump(sel, open(out / "occl_main_items.json", "w")); json.dump(cells, open(out / "occl_main_cells.json", "w"))
print("unique items", len(cells))
