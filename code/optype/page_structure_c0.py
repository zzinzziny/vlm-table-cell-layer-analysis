#!/usr/bin/env python3
import json, collections
from pathlib import Path
H = Path(__file__).resolve().parent
items = {(json.loads(l)["pool"], json.loads(l)["qid"]): json.loads(l) for l in open(H / "items_c0_all.jsonl")}
labs = {(json.loads(l)["pool"], json.loads(l)["qid"]): json.loads(l)["gpt5"] for l in open(H / "labels_c0_gpt5.jsonl")}
out = []
for k, it in items.items():
    g = labs[k]; pages = set(it["_value_pages"]) | set(it["_key_pages"])
    if g.get("table_ref") == "CAPTION" and it["n_tables"] == 1 and it["_pages"]: pages.add(it["_pages"][0])
    if g.get("scope") == "SCAN": pages |= set(it["_candidate_pages"] or it["_pages"])
    out.append({"pool": k[0], "qid": k[1], "final_op": g["final_op"], "pre_steps": g.get("pre_steps", []), "scope": g.get("scope"), "table_ref": g.get("table_ref"),
                "evidence_page_set": sorted(pages), "page": "SP" if len(pages) == 1 else "MP", "subgroup": it["_subgroup"], "note": g.get("note")})
with open(H / "c0_labels_final.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
C = collections.Counter((o["final_op"], o["page"]) for o in out)
ops = ["SELECT", "JOIN", "COMPUTE", "AGGREGATE", "COMPARE", "MULTI_CELL", "OTHER"]
print("final_op × page"); [print(f"  {op:<11} SP {C[(op,'SP')]:>4}  MP {C[(op,'MP')]:>4}") for op in ops]
print("pool × page", collections.Counter((o["pool"], o["page"]) for o in out))
print("subgroup → final_op")
for sg, c in sorted(collections.Counter((str(o["subgroup"]), o["final_op"]) for o in out).items()): print("  ", sg, c)
prev = {json.loads(l)["qid"]: json.loads(l)["gpt5"]["final_op"] for l in open(H / "labels_gpt5.jsonl") if json.loads(l)["pool"] == "v2natfull"}
same = [o for o in out if o["pool"] == "v2natfull" and o["qid"] in prev]
print('matches the earlier v2natfull label', sum(prev[o["qid"]] == o["final_op"] for o in same), "/", len(same))
