#!/usr/bin/env python3
import json, random, collections, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from build_cf_candidates_relaxed import LABELS

MODELS = ["qwen35_9b", "qwen3vl_8b", "qwen25_7b", "gemma4_12b",
          "ministral3_8b", "llava_ov2_8b", "qwen3vl_32b", "gemma4_31b"]
N, PER_DOC = 50, 3

items = {}
for f in ("cfmp_pool_qwen35.jsonl", "cfmp2_pool_qwen35.jsonl"):
    for l in open(BASE / f):
        if l.strip():
            r = json.loads(l); items[r["qid"]] = r
print('MP pool', len(items))

ok = {}
for m in MODELS:
    d = {}
    for f in (f"cfmp_gates_{m}.jsonl", f"cfmp2_gates_{m}.jsonl"):
        p = BASE / f
        if not p.exists():
            continue
        for l in open(p):
            r = json.loads(l); d[r["qid"]] = r["ok"]
    ok[m] = sorted(q for q, o in d.items()
                   if q in items and o.get("base_lookup_b") and o.get("base_compute"))

need = {(items[q]["doc_id"], str(items[q]["table_index"])) for m in MODELS for q in ok[m]}
lab = {}
for l in open(LABELS):
    d = json.loads(l); k = (d["doc_id"], str(d["table_index"]))
    if k in need:
        lab[k] = d

KEEP = ("value", "row_idx", "col_idx", "is_header", "bbox")


def cells_of(q):
    it = items[q]
    d = lab.get((it["doc_id"], str(it["table_index"])))
    if not d:
        return None
    bypage = {p["page_id"]: p for p in d["pages"]}
    out = []
    for p in it["pages"]:
        lp = bypage.get(p["page_id"])
        if not lp:
            return None
        out.append({"page_id": p["page_id"], "orig_size": lp["orig_size"],
                    "cells": [{k: c[k] for k in KEEP} for c in lp["cells"] if c.get("bbox")]})
    A, B = it["operands"]
    ap, bp = int(it["a_page_pos"]), int(it["b_page_pos"])
    has = lambda pi, o: any((c["row_idx"], c["col_idx"]) == (o["row_idx"], o["col_idx"])
                            for c in out[pi]["cells"])
    if not (has(ap, A) and has(bp, B)):
        return None
    return {"pages": out}


cells, sel = {}, {}
common = set(ok[MODELS[0]])
for m in MODELS:
    common &= set(ok[m])
print('correct in all 8 models', len(common))

for m in MODELS:
    rng = random.Random(0)
    chosen, per = [], collections.Counter()
    base = sorted(q for q in common if cells_of(q))
    for q in base:
        if len(chosen) >= N:
            break
        if per[items[q]["doc_id"]] >= PER_DOC:
            continue
        chosen.append(q); per[items[q]["doc_id"]] += 1
    rest = [q for q in ok[m] if q not in common]
    rng.shuffle(rest)
    for q in rest:
        if len(chosen) >= N:
            break
        if per[items[q]["doc_id"]] >= PER_DOC or not cells_of(q):
            continue
        chosen.append(q); per[items[q]["doc_id"]] += 1
    sel[m] = chosen
    for q in chosen:
        cells[q] = cells_of(q)
    if chosen:
        ncell = [sum(len(p["cells"]) for p in cells[q]["pages"]) for q in chosen]
        npage = [len(cells[q]["pages"]) for q in chosen]
        print(f"{m:14s} correct {len(ok[m]):4d} - common {len(base):3d} - chosen {len(chosen):3d} - documents {len({items[q]['doc_id'] for q in chosen}):3d} - mean cells {sum(ncell)/len(ncell):5.0f} (max {max(ncell)}) - mean pages {sum(npage)/len(npage):.2f}")
    else:
        print(f"{m:14s} correct {len(ok[m])} - no items chosen")

out = Path(__file__).parent
json.dump(sel, open(out / "occl_mp_items.json", "w"))
json.dump(cells, open(out / "occl_mp_cells.json", "w"))
tot = sum(sum(len(p["cells"]) for p in cells[q]["pages"]) * 2 for m in MODELS for q in sel[m])
print(f"\nitems {len(cells)} - ablations over all models {tot:,} (both tasks)")
