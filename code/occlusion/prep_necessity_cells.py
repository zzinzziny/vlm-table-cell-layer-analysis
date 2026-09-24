#!/usr/bin/env python3
import json, glob, sys
from pathlib import Path
B = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(B))
from build_cf_candidates_relaxed import LABELS
items = {json.loads(l)["qid"]: json.loads(l) for l in open(B / "common/items_sp_qwen35.jsonl")}
need = {(it["doc_id"], str(it["table_index"])) for it in items.values()}
lab = {}
for l in open(LABELS):
    d = json.loads(l); k = (d["doc_id"], str(d["table_index"]))
    if k in need: lab[k] = d
cells = {}
for q, it in items.items():
    d = lab.get((it["doc_id"], str(it["table_index"])))
    p = [p for p in (d or {}).get("pages", []) if p["page_id"] == it["pages"][0]["page_id"]]
    if not p: continue
    cells[q] = {"orig_size": p[0]["orig_size"], "cells": [{k: c[k] for k in ("value", "row_idx", "col_idx", "is_header", "bbox")} for c in p[0]["cells"] if c.get("bbox")]}
MODELS = ["qwen35_9b", "qwen3vl_8b", "qwen25_7b", "gemma4_12b", "ministral3_8b", "llava_ov2_8b", "qwen3vl_32b", "gemma4_31b"]
sel = {}
for m in MODELS:
    fs = [f for f in glob.glob(str(B / f"cf*_gates_{m}.jsonl")) if not any(x in f for x in ("cfmp", "cfsw", "cfpv", "cfrel"))]
    if m == "qwen35_9b": fs += [str(B / "cf2_gates.jsonl"), str(B / "cf3_gates.jsonl")]
    ok = {}
    for f in fs:
        for l in open(f): r = json.loads(l); ok[r["qid"]] = r["ok"]
    sel[m] = {"lookup_b": sorted(q for q, o in ok.items() if q in cells and o.get("base_lookup_b")), "compute": sorted(q for q, o in ok.items() if q in cells and o.get("base_compute"))}
    print(m, {k: len(v) for k, v in sel[m].items()})
json.dump(cells, open(Path(__file__).parent / "necessity_cells.json", "w")); json.dump(sel, open(Path(__file__).parent / "necessity_items.json", "w"))
print("pages with cells", len(cells), "/", len(items))
