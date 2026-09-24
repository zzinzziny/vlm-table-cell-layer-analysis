#!/usr/bin/env python3
import argparse, json, math, os
from pathlib import Path
from PIL import Image
BASE = Path(__file__).resolve().parent; ROOT = BASE.parent
IMG = "../data/external/pubtables-v2/Full Documents/test/images"
TOKEN_PX = {"gemma4": 48, "ministral3": 28, "llava_ov2": 28, "qwen35": 32, "qwen25": 28}
GEO_TAG = {"gemma4": "gemma4_soft1120_48px", "ministral3": "ministral3_28px", "llava_ov2": "llava_ov2_28px", "qwen25": "qwen25_28px"}
MODEL_ID = {"ministral3": "mistralai/Ministral-3-8B-Instruct-2512-BF16", "llava_ov2": "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct", "qwen25": "Qwen/Qwen2.5-VL-7B-Instruct"}


def fix_path(p): return os.path.join(IMG, os.path.basename(p))


class Geometry:
    """page image path -> (grid_h, grid_w, resized_w, resized_h) for the target geometry."""
    def __init__(self, geo, max_soft=1120):
        self.geo, self.max_soft = geo, max_soft
        self.cache_path = BASE / f"page_grid_cache_{geo}.json"
        self.cache = json.load(open(self.cache_path)) if self.cache_path.exists() else {}
        self.proc = None
        if geo in MODEL_ID:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from transformers import AutoProcessor
            self.proc = AutoProcessor.from_pretrained(MODEL_ID[geo], **({"fix_mistral_regex": True} if geo == "ministral3" else ({} if geo == "qwen25" else {"trust_remote_code": True})))

    def get(self, path):
        key = os.path.basename(path)
        if key in self.cache: return tuple(self.cache[key])
        im = Image.open(fix_path(path)).convert("RGB"); W, H = im.size
        if self.geo == "gemma4":
            f = math.sqrt(self.max_soft * 9 * 256 / (W * H)); Wp, Hp = int(W * f // 48) * 48, int(H * f // 48) * 48
            g = (Hp // 48, Wp // 48, Wp, Hp)
        elif self.geo == "ministral3":
            inp = self.proc(text=["<s>[INST][IMG][/INST]"], images=[im], return_tensors="pt"); h, w = [int(x) for x in inp["image_sizes"][0]]
            g = (math.ceil(h / 28), math.ceil(w / 28), math.ceil(W / 14) * 14, math.ceil(H / 14) * 14)
        else:
            inp = self.proc(text=["<|vision_start|><|image_pad|><|vision_end|>"], images=[im], return_tensors="pt"); t, hp, wp = [int(x) for x in inp["image_grid_thw"][0]]
            g = (hp // 2, wp // 2, wp * 14, hp * 14)
        self.cache[key] = list(g); return g

    def save(self): json.dump(self.cache, open(self.cache_path, "w"))


def bbox_to_tokens(bbox, orig_w, orig_h, grid, px):
    gh, gw, rw, rh = grid; sx, sy = rw / orig_w, rh / orig_h
    x1, y1, x2, y2 = bbox; rx1, ry1, rx2, ry2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
    cs, ce = max(0, int(rx1 // px)), min(gw - 1, int(rx2 // px)); rs, re_ = max(0, int(ry1 // px)), min(gh - 1, int(ry2 // px))
    patches = [(r, c) for r in range(rs, re_ + 1) for c in range(cs, ce + 1)]
    return patches, [r * gw + c for r, c in patches]


def token_to_orig_rect(k, grid_hw, orig_w, orig_h, px):
    gh, gw = grid_hw; r, c = divmod(k, gw); rw, rh = gw * px, gh * px
    return [c * px * orig_w / rw, r * px * orig_h / rh, (c + 1) * px * orig_w / rw, (r + 1) * px * orig_h / rh]


def remap_cell(c, page_size, grid, px):
    d = dict(c); W, H = page_size
    d["patches"], d["vision_token_indices"] = bbox_to_tokens(c["bbox"], W, H, grid, px)
    return d


def remap_item(it, G, px):
    out = json.loads(json.dumps(it)); sizes, grids = {}, {}
    for p in out["pages"]:
        g = G.get(p["image"]); im = Image.open(fix_path(p["image"])); sizes[p["page_id"]] = im.size; grids[p["page_id"]] = g
        p["grid_hw_qwen35"] = p["grid_hw"]; p["grid_hw"] = [g[0], g[1]]
    def rc(c): return remap_cell(c, sizes[c["page_id"]], grids[c["page_id"]], px)
    for f in ("operands", "controls", "key_cells"):
        if out.get(f): out[f] = [rc(c) for c in out[f]]
    for f in ("target", "control", "key_cell"):
        if out.get(f): out[f] = rc(out[f])
    for f in ("sweep_cells", "sweep_other_controls"):
        if out.get(f): out[f] = {pid: [rc(c) for c in cs] for pid, cs in out[f].items()}
    tti = {}
    for pid, idxs in out["table_token_indices"].items():
        q35 = next(p["grid_hw_qwen35"] for p in out["pages"] if p["page_id"] == pid); W, H = sizes[pid]; s = set()
        for k in idxs:
            rect = token_to_orig_rect(k, q35, W, H, 32); s.update(bbox_to_tokens(rect, W, H, grids[pid], px)[1])
        tti[pid] = sorted(s)
    out["table_token_indices"] = tti; out["geometry"] = GEO_TAG[G.geo]
    return out


def validate_pools(G, px):
    import collections
    pairs = {"gemma4": [("cf_pool_qwen35.jsonl", "cf_pool_gemma4.jsonl"), ("cf5_pool_qwen35.jsonl", "cf5_pool_gemma4.jsonl"), ("cfmp_pool_qwen35.jsonl", "cfmp_pool_gemma4.jsonl")],
             "ministral3": [("cf_pool_qwen35.jsonl", "cf_pool_ministral3.jsonl"), ("cf5_pool_qwen35.jsonl", "cf_pool_ministral3.jsonl"), ("cfmp_pool_qwen35.jsonl", "cfmp_pool_ministral3.jsonl")],
             "llava_ov2": [("cf_pool_qwen35.jsonl", "cf_pool_llava_ov2.jsonl"), ("cf5_pool_qwen35.jsonl", "cf_pool_llava_ov2.jsonl"), ("cfmp_pool_qwen35.jsonl", "cfmp_pool_llava_ov2.jsonl")],
             "qwen25": [("cf_pool_qwen35.jsonl", "cf_pool_qwen25.jsonl"), ("cf5_pool_qwen35.jsonl", "cf5_pool_qwen25.jsonl"), ("cfmp_pool_qwen35.jsonl", "cfmp_pool_qwen25.jsonl")]}[G.geo]
    for a, b in pairs:
        A = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / a)}; B = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / b)}
        n = ok_g = ok_t = sub = sup = 0; ex = []
        for q in set(A) & set(B):
            r = A[q]; bp = r["pages"][r.get("b_page_pos", 0)]; g = G.get(bp["image"]); W, H = Image.open(fix_path(bp["image"])).size
            _, toks = bbox_to_tokens(r["operands"][1]["bbox"], W, H, g, px); n += 1
            ok_g += (list(B[q]["grid_hw"]) == [g[0], g[1]]); same = (toks == B[q]["vision_token_indices"]); ok_t += same
            if not same:
                st, sb = set(toks), set(B[q]["vision_token_indices"]); sub += st < sb; sup += st > sb
                if len(ex) < 3: ex.append((q, toks, B[q]["vision_token_indices"]))
        print(f"[validate {G.geo}] {a} vs {b}: n={n} grid same {ok_g} | B tokens same {ok_t} (ours⊂pool {sub}, ours⊃pool {sup}) ex {ex}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--geo", required=True, choices=list(GEO_TAG)); ap.add_argument("--items", default=str(BASE / "items_v2nat_qwen35.jsonl"))
    ap.add_argument("--out", default=""); ap.add_argument("--validate-pools", action="store_true"); a = ap.parse_args()
    G = Geometry(a.geo); px = TOKEN_PX[a.geo]
    if a.validate_pools: validate_pools(G, px); G.save()
    items = [json.loads(l) for l in open(a.items)]; out = Path(a.out) if a.out else BASE / f"items_v2nat_{a.geo}.jsonl"
    with open(out, "w") as f:
        for it in items: f.write(json.dumps(remap_item(it, G, px), ensure_ascii=False) + "\n")
    G.save()
    import collections
    r = [json.loads(l) for l in open(out)]
    ops = [c for i in r for c in (i.get('operands') or [])] or [i['target'] for i in r if i.get('target')]
    print(f"-> {out.name}: {len(r)} items, pages grid {collections.Counter(tuple(p['grid_hw']) for i in r for p in i['pages']).most_common(4)}, "
          f"operand/target tokens/cell median {sorted(len(c['vision_token_indices']) for c in ops)[len(ops)//2] if ops else '—'}, "
          f"cells with 0 tokens {sum(1 for i in r for fld in ('operands','controls','key_cells') for c in (i.get(fld) or []) if not c['vision_token_indices'])}")


if __name__ == "__main__":
    main()
