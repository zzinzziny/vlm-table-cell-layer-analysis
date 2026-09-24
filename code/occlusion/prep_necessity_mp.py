#!/usr/bin/env python3
import json, sys
from pathlib import Path

B = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(B))
from build_cf_candidates_relaxed import LABELS

items = {}
for f in ("cfmp_pool_qwen35.jsonl", "cfmp2_pool_qwen35.jsonl"):
    for l in open(B / f):
        if l.strip():
            r = json.loads(l); items[r["qid"]] = r
print('MP pool', len(items))

need = {(it["doc_id"], str(it["table_index"])) for it in items.values()}
lab = {}
for l in open(LABELS):
    d = json.loads(l); k = (d["doc_id"], str(d["table_index"]))
    if k in need:
        lab[k] = d

KEEP = ("value", "row_idx", "col_idx", "is_header", "bbox")
cells = {}
for q, it in items.items():
    d = lab.get((it["doc_id"], str(it["table_index"])))
    if not d:
        continue
    bp = int(it["b_page_pos"])
    pid = it["pages"][bp]["page_id"]
    p = [p for p in d.get("pages", []) if p["page_id"] == pid]
    if not p:
        continue
    cells[q] = {"orig_size": p[0]["orig_size"], "b_page_pos": bp,
                "cells": [{k: c[k] for k in KEEP} for c in p[0]["cells"] if c.get("bbox")]}

MODELS = ["qwen35_9b", "qwen3vl_8b", "qwen25_7b", "gemma4_12b",
          "ministral3_8b", "llava_ov2_8b", "qwen3vl_32b", "gemma4_31b"]
sel = {}
for m in MODELS:
    ok = {}
    for f in (f"cfmp_gates_{m}.jsonl", f"cfmp2_gates_{m}.jsonl"):
        p = B / f
        if not p.exists():
            continue
        for l in open(p):
            r = json.loads(l); ok[r["qid"]] = r["ok"]
    sel[m] = {"lookup_b": sorted(q for q, o in ok.items() if q in cells and o.get("base_lookup_b")),
              "compute": sorted(q for q, o in ok.items() if q in cells and o.get("base_compute"))}
    print(f"{m:14s} " + str({k: len(v) for k, v in sel[m].items()}))

out = Path(__file__).parent
json.dump(cells, open(out / "necessity_mp_cells.json", "w"))
json.dump(sel, open(out / "necessity_mp_items.json", "w"))
tot = sum(len(v["lookup_b"]) + len(v["compute"]) for v in sel.values())
print(f"\nitems with cells on the B page {len(cells)} / {len(items)}")
print(f"total jobs over all models {tot:,} x (base + target + control) = 3 generations")
