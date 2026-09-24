#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
from PIL import Image
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "mpreq"))
L28 = "../data/external/qa_enrichment/patch_cell_labels/patch_cell_labels_test.jsonl"
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--geo", required=True); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True); ap.add_argument("--by-geometry", action="store_true", help='convert qwen25 with the processor geometry instead of the label file (pool including mini-label cells)'); a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp)]
    if a.geo == "qwen25" and not a.by_geometry:
        need = {(r["doc_id"], r["table_index"]) for r in rows}; cells = {}; grids = {}
        for l in open(L28):
            d = json.loads(l)
            if (d["doc_id"], d["table_index"]) not in need: continue
            for p in d["pages"]:
                grids[(d["doc_id"], d["table_index"], p["page_id"])] = p["grid_hw"]
                for c in p.get("cells", []): cells[(d["doc_id"], d["table_index"], p["page_id"], c["row_idx"], c["col_idx"])] = c
        def rc(r, pos, c):
            pid = r["pages"][pos]["page_id"]; src = cells[(r["doc_id"], r["table_index"], pid, c["row_idx"], c["col_idx"])]
            d = dict(c); d["bbox"], d["patches"], d["vision_token_indices"] = src["bbox"], src["patches"], src["vision_token_indices"]; return d
        for r in rows:
            for p in r["pages"]: p["grid_hw"] = grids[(r["doc_id"], r["table_index"], p["page_id"])]
            r["operands"] = [rc(r, r["a_page_pos"], r["operands"][0]), rc(r, r["b_page_pos"], r["operands"][1])]
            r["bbox"], r["vision_token_indices"], r["grid_hw"] = r["operands"][1]["bbox"], r["operands"][1]["vision_token_indices"], r["pages"][r["b_page_pos"]]["grid_hw"]
            r["geometry"] = "qwen25_28px"
    else:
        from remap_items_geometry import Geometry, bbox_to_tokens, fix_path, TOKEN_PX, GEO_TAG
        G = Geometry(a.geo); px = TOKEN_PX[a.geo]
        for r in rows:
            sizes = {}
            for p in r["pages"]:
                g = G.get(p["image"]); sizes[p["page_id"]] = Image.open(fix_path(p["image"])).size; p["grid_hw_qwen35"] = p["grid_hw"]; p["grid_hw"] = [g[0], g[1]]; p["_g"] = g
            def rc(pos, c):
                p = r["pages"][pos]; W, H = sizes[p["page_id"]]; d = dict(c); d["patches"], d["vision_token_indices"] = bbox_to_tokens(c["bbox"], W, H, p["_g"], px); return d
            r["operands"] = [rc(r["a_page_pos"], r["operands"][0]), rc(r["b_page_pos"], r["operands"][1])]
            r["vision_token_indices"], r["grid_hw"] = r["operands"][1]["vision_token_indices"], r["pages"][r["b_page_pos"]]["grid_hw"]
            for p in r["pages"]: p.pop("_g", None)
            r["geometry"] = GEO_TAG[a.geo]
        G.save()
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {a.out}: {len(rows)} rows, B tokens/cell median {sorted(len(r['vision_token_indices']) for r in rows)[len(rows)//2]}, zero-token B {sum(1 for r in rows if not r['vision_token_indices'])}")
if __name__ == "__main__": main()
