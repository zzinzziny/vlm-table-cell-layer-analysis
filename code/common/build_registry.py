#!/usr/bin/env python3
import json, re
from collections import defaultdict
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
LABELS = {"qwen35": Path("../data/external/qa_enrichment/patch_cell_labels/patch_cell_labels_test_qwen35.jsonl"),
          "qwen25": Path("../data/external/qa_enrichment/patch_cell_labels/patch_cell_labels_test.jsonl")}
INT_RE = re.compile(r"^\d{1,4}$")
INSTR = "Reply with the answer only. No explanation, no reasoning, no bullet points."
GEOS = {"sp": {"qwen35": ["cf_pool_qwen35.jsonl", "cf5_pool_qwen35.jsonl"], "qwen25": ["cf_pool_qwen25.jsonl", "cf5_pool_qwen25.jsonl"],
               "gemma4": ["cf_pool_gemma4.jsonl", "cf5_pool_gemma4.jsonl"], "ministral3": ["cf_pool_ministral3.jsonl"], "llava_ov2": ["cf_pool_llava_ov2.jsonl"]},
        "mp": {k: [f"cfmp_pool_{k}.jsonl"] for k in ("qwen35", "qwen25", "gemma4", "ministral3", "llava_ov2")}}
GEO_MODELS = {"qwen35": ["qwen35_9b", "qwen3vl_8b", "qwen3vl_32b"], "qwen25": ["qwen25_7b"], "gemma4": ["gemma4_12b", "gemma4_31b"],
              "ministral3": ["ministral3_8b"], "llava_ov2": ["llava_ov2_8b"]}

def load(files):
    d = {}
    for f in files:
        for l in open(BASE / f):
            r = json.loads(l); d[r["qid"]] = r
    return d
def apply(op, x, y): return str(x + y) if op == "sum" else str(abs(x - y))
def other(op): return "sum" if op == "difference" else "difference"
def recompute_prompts(b_row, col, a_row, a2_row, op):
    prefix = f"In the table, find the row '{b_row}' and look at its value in the '{col}' column."
    lookup = prefix + " What is that value?"
    def comp(row, o):
        if o == "sum": return prefix + f" What is the sum of that value and the '{col}' value for the row '{row}'?"
        return prefix + f" What is the difference between the '{col}' value for the row '{row}' and that value?"
    return {"prefix": prefix, "lookup": lookup, "compute_A": comp(a_row, op), "compute_A2_f": comp(a2_row, op) if a2_row else None,
            "compute_A2_g": comp(a2_row, other(op)) if a2_row else None}
def cell_key(c): return (c["row_idx"], c["col_idx"])

def main():
    old_a2 = {json.loads(l)["qid"]: json.loads(l)["a2"]["value"] for l in open(BASE / "handoff2/recompute_items.jsonl")}
    # labels: (doc, table) -> {page_id: cells}
    lab = {}
    for geo, path in LABELS.items():
        m = {}
        for l in open(path):
            d = json.loads(l)
            m[(d["doc_id"], d["table_index"])] = {p["page_id"]: p.get("cells", []) for p in d["pages"]}
        lab[geo] = m
    for setname in ("sp", "mp"):
        pools = {geo: load(files) for geo, files in GEOS[setname].items()}
        base = pools["qwen35"]; rows = []; stats = defaultdict(int)
        for qid, r in base.items():
            A, B = r["operands"]
            op = r["op"]; a_i, b_i, bp_i = int(A["value"]), int(B["value"]), int(r["b_new"])
            assert apply(op, a_i, b_i) == str(r["compute_gold"]) and apply(op, a_i, bp_i) == str(r["compute_gold_new"]), qid
            pages = r["pages"]; b_page = pages[r.get("b_page_pos", 0)]["page_id"]
            reg = {"item_id": qid, "set": setname, "pool": qid.split("_")[0], "doc_id": r["doc_id"], "table_index": r["table_index"],
                   "page_ids": [p["page_id"] for p in pages], "b_page_id": b_page, "a_page_id": pages[r.get("a_page_pos", 0)]["page_id"],
                   "A": {k: A[k] for k in ("value", "row_header", "col_header", "row_idx", "col_idx", "bbox")},
                   "B": {k: B[k] for k in ("value", "row_header", "col_header", "row_idx", "col_idx", "bbox")},
                   "b_new": r["b_new"], "op": op, "R": str(r["compute_gold"]), "R_new": str(r["compute_gold_new"]),
                   "questions": {"lookup_a": r["lookup_a"]["question"], "lookup_b": r["lookup_b_question"], "compute": r["compute_question"], "instr": INSTR},
                   "base_image": r.get("base_image"), "donor_image": r.get("donor_image"),
                   "base_images": r.get("base_images"), "donor_images": r.get("donor_images"), "n_pages": r["n_pages"], "geometry": {}}
            for geo, pool in pools.items():
                g = pool.get(qid)
                if g is None: stats[f"missing_geo_{geo}"] += 1; continue
                assert g["lookup_b_question"] == r["lookup_b_question"] and g["compute_question"] == r["compute_question"] and g["b_new"] == r["b_new"], qid
                reg["geometry"][geo] = {"models": GEO_MODELS[geo], "B_tokens": g["vision_token_indices"], "A_tokens": g["operands"][0].get("vision_token_indices"),
                                        "grid_hw": g["grid_hw"], "pages_grid_hw": [p.get("grid_hw") for p in g["pages"]],
                                        "b_page_pos": g.get("b_page_pos", 0), "a_page_pos": g.get("a_page_pos", 0), "label": g.get("geometry", "qwen35_patch16")}
            # ---- A₂
            tab = lab["qwen35"].get((r["doc_id"], r["table_index"]), {})
            cand = []
            for pid, cells in tab.items():
                if setname == "sp" and pid != b_page: continue
                rows_hdr = [str(c.get("row_header", "")).strip() for c in cells if c.get("col_idx") == B["col_idx"]]
                for c in cells:
                    if c.get("col_idx") != B["col_idx"] or c.get("is_header"): continue
                    if cell_key(c) in (cell_key(A), cell_key(B)): continue
                    v = str(c.get("value", "")).strip()
                    if not INT_RE.match(v) or c.get("match_ratio", 0) < 0.95 or not c.get("bbox"): continue
                    rh = str(c.get("row_header", "")).strip()
                    if not rh or rows_hdr.count(rh) > 1 or rh in (A["row_header"], B["row_header"]): continue
                    cand.append((pid, c))
            if not cand:
                reg["a2"] = None; reg["recompute_ineligible_reason"] = "no_other_int_cell"; stats["no_other_int_cell"] += 1
            else:
                ok = []
                for pid, c in cand:
                    a2 = int(c["value"]); f, g = op, other(op)
                    vals = {"R": reg["R"], "R_new": reg["R_new"], "D": apply(f, a2, bp_i), "D0": apply(f, a2, b_i), "E": apply(g, a2, bp_i), "E0": apply(g, a2, b_i), "B": str(b_i), "B_new": str(bp_i)}
                    if len(set(vals.values())) == 8 and int(vals["D"]) != 0 and int(vals["E"]) != 0 and int(vals["D0"]) != 0 and int(vals["E0"]) != 0:
                        pref = 0 if str(a2) == old_a2.get(qid) else 1
                        ok.append(((pref, abs(len(str(a2)) - len(str(b_i))), len(str(c["row_header"]).strip()), abs(a2 - a_i)), 0, pid, c, vals))
                if not ok:
                    reg["a2"] = None; reg["recompute_ineligible_reason"] = "answers_collide"; stats["answers_collide"] += 1
                else:
                    ok.sort(key=lambda t: t[0]); _, _, pid, c, vals = ok[0]
                    a2 = {"value": str(c["value"]).strip(), "row_header": str(c["row_header"]).strip(), "row_idx": c["row_idx"], "col_idx": c["col_idx"], "bbox": c["bbox"], "page_id": pid,
                          "tokens": {"qwen35": c.get("vision_token_indices")}}
                    q25 = lab["qwen25"].get((r["doc_id"], r["table_index"]), {}).get(pid, [])
                    m25 = next((x for x in q25 if cell_key(x) == cell_key(c)), None)
                    a2["tokens"]["qwen25"] = m25.get("vision_token_indices") if m25 else None
                    a2["tokens_needed_from_bbox"] = ["gemma4", "ministral3", "llava_ov2"]
                    reg["a2"] = a2; reg["recompute_vals"] = vals; reg["recompute_ineligible_reason"] = None; stats["eligible"] += 1
                    if setname == "mp" and pid != b_page: stats["a2_other_page"] += 1
            reg["recompute_questions"] = recompute_prompts(B["row_header"], B["col_header"], A["row_header"], reg["a2"]["row_header"] if reg["a2"] else None, op)
            rows.append(reg)
        with open(OUT / f"registry_{setname}.jsonl", "w") as f:
            for reg in rows: f.write(json.dumps(reg, ensure_ascii=False) + "\n")
        print(setname, len(rows), dict(stats))
    old = {json.loads(l)["qid"]: json.loads(l) for l in open(BASE / "handoff2/recompute_items.jsonl")}
    reg = {json.loads(l)["item_id"]: json.loads(l) for l in open(OUT / "registry_sp.jsonl")}
    same = diff = inel = 0; ex = []
    for q, o in old.items():
        r = reg[q]
        if r["a2"] is None: inel += 1; ex.append((q, "ineligible", r["recompute_ineligible_reason"], o["a2"]["value"]))
        elif r["a2"]["value"] == o["a2"]["value"]: same += 1
        else: diff += 1; ex.append((q, "different A2", o["a2"]["value"], r["a2"]["value"]))
    print(f"existing recompute_items 33: same A2 {same}, different A2 {diff}, ineligible under the new rule {inel}"); [print("  ", e) for e in ex[:10]]
if __name__ == "__main__": main()
