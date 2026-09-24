#!/usr/bin/env python3
import json, re
from pathlib import Path
M = Path(__file__).resolve().parent.parent / "mpreq"
RANK = re.compile(r"\b(highest|lowest|largest|smallest|greatest|most|least|maximum|minimum|max|min|top|biggest|second[- ](largest|highest|smallest|lowest)|rank|ranked|peak)\b", re.I)
FILT = re.compile(r"\b(where|whose|only|excluding|among (the )?rows|for (the )?rows?|that (have|has|are|is)|with (a|an|the)? ?\w+ (of|equal|above|below|greater|less))\b", re.I)
AGG = re.compile(r"\b(sum|total|add|adding|combined|average|mean|how many|count|number of)\b", re.I)
def label(r):
    s, q, fam, nt = r["subgroup"], r["question"], r.get("family") or "", len(r.get("table_ids") or [])
    if s == "multi_answer": f = "MULTI_CELL"
    elif s == "multi_hop_steps" or (s in ("lookup_cond", "answer_plus_operands") and (nt >= 2 or "multi_hop" in fam)): f = "JOIN"
    elif s in ("rank_candidates", "extremes", "answer_plus_operands"): f = "COMPARE"
    elif s in ("lookup_cond", "lookup_v2"): f = "SELECT"
    elif s == "aggregate": f = "AGGREGATE" if AGG.search(q) else "AGGREGATE_OTHER"
    elif s == "two_cell": f = "COMPUTE"
    else: f = "OTHER"
    pre_rank = bool(RANK.search(q)) and f != "COMPARE"
    pre_filter = bool(r.get("key_cells") or r.get("key_cell")) or bool(FILT.search(q))
    return dict(final_op=f, pre_rank=pre_rank, pre_filter=pre_filter)
out = []
for pool, fn in (("v2natfull", "items_v2natfull_qwen35.jsonl"), ("v2natext", "items_v2natext_qwen35.jsonl")):
    for l in open(M / fn):
        r = json.loads(l)
        ops = [{"row": o.get("row_header"), "col": o.get("col_header"), "value": o.get("value")} for o in (r.get("operands") or [])][:12]
        tgt = r.get("target"); keys = r.get("key_cells") or ([r["key_cell"]] if r.get("key_cell") else [])
        out.append(dict(pool=pool, qid=r["qid"], question=r["question"], gold=r["gold"], subgroup=r["subgroup"], family=r.get("family"), n_tables=len(r.get("table_ids") or []),
                        k_operands=r.get("k_operands"), operand_cells=ops, answer_cell=({"row": tgt.get("row_header"), "col": tgt.get("col_header"), "value": tgt.get("value")} if tgt else None),
                        key_cells=[{"row": k.get("row_header"), "col": k.get("col_header"), "value": k.get("value")} for k in keys][:6], rule=label(r)))
with open(Path(__file__).resolve().parent / "items_optype_rules.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
from collections import Counter
print(len(out), Counter(o["rule"]["final_op"] for o in out), "pre_rank", sum(o["rule"]["pre_rank"] for o in out), "pre_filter", sum(o["rule"]["pre_filter"] for o in out))
