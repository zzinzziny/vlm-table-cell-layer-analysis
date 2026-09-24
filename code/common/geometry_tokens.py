#!/usr/bin/env python3
import math
TPX = {"qwen35": 32, "qwen25": 28, "llava_ov2": 28, "ministral3": 28, "gemma4": 48}
def tokens_for_bbox(bbox, page_wh, grid_hw, geo, stretch=True):
    t = TPX[geo]; W, H = page_wh; gh, gw = grid_hw; sx, sy = (gw * t / W, gh * t / H) if stretch else (1.0, 1.0)
    x1, y1, x2, y2 = bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy; out = []
    for r in range(max(0, math.floor(y1 / t)), min(gh - 1, math.floor(y2 / t)) + 1):
        for c in range(max(0, math.floor(x1 / t)), min(gw - 1, math.floor(x2 / t)) + 1):
            ox = max(0, min(x2, (c + 1) * t) - max(x1, c * t)); oy = max(0, min(y2, (r + 1) * t) - max(y1, r * t))
            if ox * oy > 0: out.append(r * gw + c)
    return out
def rule_for_item(b_bbox, b_tokens, page_wh, grid_hw, geo):
    for st in (True, False):
        if tokens_for_bbox(b_bbox, page_wh, grid_hw, geo, st) == list(b_tokens): return st, True
    return True, False
if __name__ == "__main__":
    import json, sys
    from PIL import Image
    from collections import Counter
    tot = Counter(); bad = []
    for f in ("registry_sp.jsonl", "registry_mp.jsonl"):
        for l in open(f):
            r = json.loads(l)
            for geo, g in r["geometry"].items():
                bpos = g["b_page_pos"]; grid = g["pages_grid_hw"][bpos]
                img = (r["base_images"] or [r["base_image"]])[bpos] if r.get("base_images") else r["base_image"]
                wh = Image.open(img).size
                pred = tokens_for_bbox(r["B"]["bbox"], wh, grid, geo)
                ok = pred == g["B_tokens"]; tot[(f, geo, ok)] += 1
                if not ok and len(bad) < 8: bad.append((r["item_id"], geo, g["B_tokens"], pred, wh, grid))
    for k, v in sorted(tot.items()): print(k, v)
    for b in bad: print("mismatch", b)
