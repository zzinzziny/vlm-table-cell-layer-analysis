#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from PIL import Image
from geometry_tokens import tokens_for_bbox, rule_for_item
LAB = {"qwen35": "../data/external/qa_enrichment/patch_cell_labels/patch_cell_labels_test_qwen35.jsonl",
       "qwen25": "../data/external/qa_enrichment/patch_cell_labels/patch_cell_labels_test.jsonl", "llava_ov2": "../data/external/qa_enrichment/patch_cell_labels/patch_cell_labels_test.jsonl"}
ap = argparse.ArgumentParser(); ap.add_argument("--geo", required=True); ap.add_argument("--set", default="sp"); a = ap.parse_args()
rows = [json.loads(l) for l in open(f"registry_{a.set}.jsonl")]
labels = {}
if a.geo in LAB and Path(LAB[a.geo]).exists():
    need = {(r["doc_id"], r["table_index"]) for r in rows}
    for l in open(LAB[a.geo]):
        d = json.loads(l)
        if (d["doc_id"], d["table_index"]) in need:
            for p in d["pages"]:
                for c in p.get("cells", []): labels[(d["doc_id"], d["table_index"], p["page_id"], c["row_idx"], c["col_idx"])] = c
wh_cache = {}
def wh(img):
    if img not in wh_cache: wh_cache[img] = Image.open(img).size
    return wh_cache[img]
stat = {"A_label": 0, "A_formula": 0, "A2_label": 0, "A2_formula": 0, "A_eq_pool": 0, "rule_stretch": 0, "rule_plain": 0, "rule_unmatched": 0}
out = []
for r in rows:
    g = r["geometry"][a.geo]; imgs = r["base_images"] or [r["base_image"]]; dimgs = r["donor_images"] or [r["donor_image"]]
    bgrid = g["grid_hw"] if a.set == "sp" else g["pages_grid_hw"][g["b_page_pos"]]
    stretch, matched = rule_for_item(r["B"]["bbox"], g["B_tokens"], wh(imgs[g["b_page_pos"]]), bgrid, a.geo)
    stat["rule_stretch" if (stretch and matched) else "rule_plain" if matched else "rule_unmatched"] += 1
    def cell_tokens(cell, page_id, page_pos, kind):
        key = (r["doc_id"], r["table_index"], page_id, cell["row_idx"], cell["col_idx"]); lc = labels.get(key)
        grid = g["grid_hw"] if (a.set == "sp") else g["pages_grid_hw"][page_pos]
        if lc and lc.get("vision_token_indices") and (a.set == "sp" or True):
            stat[kind + "_label"] += 1; return list(lc["vision_token_indices"])
        stat[kind + "_formula"] += 1; return tokens_for_bbox(cell["bbox"], wh(imgs[page_pos]), grid, a.geo, stretch)
    A = dict(r["A"]); B = dict(r["B"]); A["vision_token_indices"] = cell_tokens(r["A"], r["a_page_id"], g["a_page_pos"], "A"); B["vision_token_indices"] = list(g["B_tokens"])
    if a.geo == "qwen35" and A["vision_token_indices"] == g["A_tokens"]: stat["A_eq_pool"] += 1
    a2 = None
    if r.get("a2"):
        a2 = dict(r["a2"]); pos = r["page_ids"].index(a2["page_id"]) if a2.get("page_id") in r["page_ids"] else g["b_page_pos"]
        a2["page_pos"] = pos; a2["vision_token_indices"] = (a2.get("tokens") or {}).get(a.geo) or cell_tokens(r["a2"], a2.get("page_id", r["b_page_id"]), pos, "A2")
        if (a2.get("tokens") or {}).get(a.geo): stat["A2_label"] += 1
    item = {"qid": r["item_id"], "item_id": r["item_id"], "set": r["set"], "pool": r["pool"], "doc_id": r["doc_id"], "table_index": r["table_index"], "n_pages": r["n_pages"],
            "pages": [{"page_id": pid, "image": imgs[i], "grid_hw": g["pages_grid_hw"][i] if a.set == "mp" else g["grid_hw"]} for i, pid in enumerate(r["page_ids"])],
            "op": r["op"], "operands": [A, B], "b_value": r["B"]["value"], "b_new": r["b_new"], "compute_gold": r["R"], "compute_gold_new": r["R_new"],
            "lookup_a": {"question": r["questions"]["lookup_a"], "gold": r["A"]["value"]}, "lookup_b": {"question": r["questions"]["lookup_b"], "gold": r["B"]["value"]},
            "lookup_b_question": r["questions"]["lookup_b"], "compute_question": r["questions"]["compute"],
            "base_image": imgs[g["b_page_pos"]] if a.set == "sp" else None, "donor_image": dimgs[g["b_page_pos"]] if a.set == "sp" else None,
            "vision_token_indices": list(g["B_tokens"]), "grid_hw": g["grid_hw"], "b_page_pos": g["b_page_pos"], "a_page_pos": g["a_page_pos"], "geometry_label": g["label"],
            "a2": a2, "recompute_vals": r.get("recompute_vals"), "recompute_ineligible_reason": r.get("recompute_ineligible_reason")}
    if a.set == "mp": item["base_images"], item["donor_images"] = imgs, dimgs
    else: item["base_images"], item["donor_images"] = None, None
    for k in ("base_images", "donor_images"):
        if item[k] is None: del item[k]
    out.append(item)
fn = f"items_{a.set}_{a.geo}.jsonl"
open(fn, "w").writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out); print(fn, len(out), stat)
