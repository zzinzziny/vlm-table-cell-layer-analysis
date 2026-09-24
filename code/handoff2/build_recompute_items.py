#!/usr/bin/env python3
import json, re
from pathlib import Path
BASE = Path(__file__).resolve().parent; A = BASE.parent
LAB = "../data/external/qa_enrichment/patch_cell_labels/patch_cell_labels_test_qwen35.jsonl"
INT = re.compile(r"^\d{2,4}$")
C = {}
for f in ("cf3_candidates.jsonl", "cf2_candidates.jsonl"):
    for l in open(A / f): c = json.loads(l); C[c["qid"]] = c
Q = [l.strip() for l in open(BASE / "crosstask_pass_qids.txt") if l.strip()]
need = {(C[q]["doc_id"], C[q]["table_index"]) for q in Q}; lab = {}
for l in open(LAB):
    r = json.loads(l)
    if (r["doc_id"], r["table_index"]) in need: lab[(r["doc_id"], r["table_index"])] = r
fd = lambda x: str(abs(int(x)))[0]
out = []
for q in Q:
    c = C[q]; Ao, Bo = c["operands"]; pg = c["pages"][0]["page_id"]; rec = lab[(c["doc_id"], c["table_index"])]
    a, b, b2 = int(Ao["value"]), int(Bo["value"]), int(c["b_new"])
    f = (lambda x, y: x + y) if c["op"] == "sum" else (lambda x, y: abs(x - y))
    R, Rp = f(a, b), f(a, b2)
    assert R == int(c["compute_gold"]) and Rp == int(c["compute_gold_new"]), q
    cells = [cl for p in rec["pages"] if p["page_id"] == pg for cl in p["cells"]
             if cl["col_idx"] == Bo["col_idx"] and INT.match(str(cl["value"]).strip()) and not cl["is_header"]
             and cl["row_idx"] not in (Ao["row_idx"], Bo["row_idx"]) and cl.get("vision_token_indices") and 2 <= len(str(cl["row_header"]).strip()) <= 60]
    good = []
    for cl in cells:
        a2 = int(cl["value"]); D, D0 = f(a2, b2), f(a2, b)
        if D <= 0 or D0 <= 0: continue
        if len({fd(R), fd(Rp), fd(D), fd(b), fd(b2)}) == 5 and fd(D0) != fd(D) and fd(D0) != fd(Rp):
            good.append((len(str(a2)), abs(a2 - a), cl))
    if not good: continue
    good.sort(key=lambda x: (x[0], -x[1]))
    cl = good[0][2]; a2 = int(cl["value"])
    it = dict(c); it["a2"] = {k: cl[k] for k in ("value", "row_header", "row_idx", "col_idx", "bbox", "vision_token_indices")}
    it["D"] = str(f(a2, b2)); it["D0"] = str(f(a2, b)); it["R"] = str(R); it["Rp"] = str(Rp); it["f"] = c["op"]
    out.append(it)
open(BASE / "recompute_items.jsonl", "w").write("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out))
print(f"items {len(out)}/{len(Q)}")
for o in out[:5]: print(" ", o["qid"], o["f"], "A", o["operands"][0]["value"], "B", o["operands"][1]["value"], "B'", o["b_new"], "R", o["R"], "R'", o["Rp"], "A2", o["a2"]["value"], "D", o["D"], "D0", o["D0"])
