#!/usr/bin/env python3
import json, ast
from pathlib import Path
M = Path(__file__).resolve().parent.parent / "mpreq"
P = lambda x: ast.literal_eval(x) if isinstance(x, str) else x
def cell(c): return {"row": c.get("row_header"), "col": c.get("col_header"), "value": c.get("value")} if c else None
def pid(c): return c.get("page_id") if isinstance(c, dict) else None
cs_q = {json.loads(l)["qid"] for l in open(M / "layer_necessity_compute_sp_qwen35_9b_v2.jsonl")}
SRC = [("lookup58", "items_lookup58_qwen35_remap.jsonl", None), ("compute_sp", "items_compute_qwen35_remap.jsonl", cs_q),
       ("l3big", "items_l3big_qwen35.jsonl", None), ("v2natfull", "items_v2natfull_qwen35.jsonl", None), ("mpreq", "items_mpreq_qwen35_remap.jsonl", None), ("v2natext", "items_v2natext_qwen35.jsonl", None)]
out = []
for pool, fn, keep in SRC:
    for l in open(M / fn):
        r = json.loads(l)
        if keep is not None and r["qid"] not in keep: continue
        ops = P(r.get("operands")) or []; tgt = P(r.get("target")); keys = (P(r.get("key_cells")) or []) + ([P(r["key_cell"])] if P(r.get("key_cell")) else [])
        sweep = P(r.get("sweep_cells")) or {}
        pages = [p["page_id"] for p in P(r["pages"])]
        out.append({"pool": pool, "qid": r["qid"], "question": r["question"], "gold": r["gold"],
                    "n_tables": len(P(r.get("table_ids")) or [r.get("table_index")]), "operand_cells": [cell(o) for o in ops][:12], "answer_cell": cell(tgt),
                    "key_cells": [cell(k) for k in keys][:6], "n_candidate_cells": sum(len(v) for v in sweep.values()) if isinstance(sweep, dict) else 0,
                    "_pages": pages, "_value_pages": sorted({pid(c) for c in ops + ([tgt] if tgt else []) if pid(c)}), "_key_pages": sorted({pid(k) for k in keys if pid(k)}),
                    "_candidate_pages": sorted(sweep.keys()) if isinstance(sweep, dict) else [], "_evidence_pages": P(r.get("evidence_pages")) or [],
                    "_release_evidence_pages": P(r.get("release_evidence_pages")) or [], "_subgroup": r.get("subgroup") or r.get("group") or r.get("case_name")})
json.dump
with open(Path(__file__).parent / "items_c0_all.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
from collections import Counter
print(len(out), Counter(o["pool"] for o in out), 'no value page', sum(1 for o in out if not o["_value_pages"]))
