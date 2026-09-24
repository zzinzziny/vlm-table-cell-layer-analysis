#!/usr/bin/env python3
import argparse, json, math, random, re, sys, unicodedata, torch
from pathlib import Path
from PIL import Image
D = Path(__file__).resolve().parent; B_ = D.parent; sys.path.insert(0, str(B_))
from cf_mp_common import load, build, page_token_positions, image_token_id
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--backend", required=True); ap.add_argument("--geo", required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--no-thinking-flag", action="store_true"); ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
OUT = D / f"necessity_{args.tag}.jsonl"
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s)).strip().lower().replace("−", "-").replace(",", "")).rstrip(".")
cells_all = json.load(open(D / "necessity_cells.json")); sel = json.load(open(D / "necessity_items.json"))[args.tag]
items = {json.loads(l)["qid"]: json.loads(l) for l in open(B_ / f"common/items_sp_{args.geo}.jsonl")}
done = {(json.loads(l)["qid"], json.loads(l)["task"]) for l in open(OUT)} if OUT.exists() else set()
def cell_tokens(bbox, W, H, gh, gw):
    if args.backend == "mistral3": sx = 28 * W / (math.ceil(W / 14) * 14); sy = 28 * H / (math.ceil(H / 14) * 14)
    else: sx, sy = W / gw, H / gh
    x1, y1, x2, y2 = bbox
    c0, c1 = max(0, int(x1 // sx)), min(gw - 1, int((x2 - 1e-6) // sx)); r0, r1 = max(0, int(y1 // sy)), min(gh - 1, int((y2 - 1e-6) // sy))
    return [r * gw + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
proc, model = load(args.model, args.backend); tok = proc.tokenizer; itid = image_token_id(model, args.backend)
lm = getattr(model.model, "language_model", model.model); layers = lm.layers
st = {"rows": None}
def hook(m, a, k):
    hs = a[0] if a else k["hidden_states"]
    if hs.shape[1] > 1 and st["rows"] is not None:
        rows, vis = st["rows"]; hs[:, rows, :] = hs[:, vis, :].mean(1, keepdim=True).to(hs.dtype)
    return None
layers[0].register_forward_pre_hook(hook, with_kwargs=True)
def gen(inp):
    with torch.no_grad(): g = model.generate(**inp, max_new_tokens=16, do_sample=False)
    return tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
jobs = [(q, t) for t in ("lookup_b", "compute") for q in sel[t] if (q, t) not in done]
if args.limit: jobs = jobs[:args.limit]
print("jobs", len(jobs), flush=True)
for n, (q, task) in enumerate(jobs):
    it = items[q]; img = Image.open(it["base_image"]).convert("RGB"); W, H = img.size; gh, gw = it["grid_hw"]
    A, Bc = it["operands"]; cinfo = cells_all[q]
    btok = sorted(it["vision_token_indices"]); atok = set(cell_tokens(A["bbox"], W, H, gh, gw)); ev = set(btok) | atok
    cands = []
    for c in cinfo["cells"]:
        if c.get("is_header") or c["row_idx"] in (Bc["row_idx"], A["row_idx"]) or c["col_idx"] == Bc["col_idx"]: continue
        v = str(c["value"]).strip()
        if v in (str(it["b_value"]), str(it["compute_gold"]), ""): continue
        t = cell_tokens(c["bbox"], W, H, gh, gw)
        if set(t) & ev or not t: continue
        cands.append((abs(len(t) - len(btok)), c["row_idx"], c["col_idx"], t, v))
    if not cands:
        open(OUT, "a").write(json.dumps({"qid": q, "task": task, "skip": "no control"}) + "\n"); continue
    best = min(x[0] for x in cands); ties = sorted([x for x in cands if x[0] == best], key=lambda x: (x[1], x[2]))
    ctl = random.Random(hash((q, "ctl")) % 2**32 if False else 0).choice(ties)
    question, gold = (it["lookup_b_question"], it["b_value"]) if task == "lookup_b" else (it["compute_question"], it["compute_gold"])
    inp, grids = build(proc, model, [img], question, args.backend, not args.no_thinking_flag)
    ids = inp["input_ids"][0]; pos = page_token_positions(ids, itid, it, grids)[0]; vis = torch.tensor(pos, device=model.device)
    st["rows"] = None; base = gen(inp)
    st["rows"] = (torch.tensor([pos[k] for k in btok], device=model.device), vis); abl_t = gen(inp)
    st["rows"] = (torch.tensor([pos[k] for k in ctl[3]], device=model.device), vis); abl_c = gen(inp); st["rows"] = None
    r = {"qid": q, "task": task, "gold": gold, "base": base, "base_ok": norm(base) == norm(gold), "target": abl_t, "target_ok": norm(abl_t) == norm(gold),
         "control": abl_c, "control_ok": norm(abl_c) == norm(gold), "n_tok_b": len(btok), "n_tok_ctl": len(ctl[3]), "ctl_cell": [ctl[1], ctl[2], ctl[4]]}
    open(OUT, "a").write(json.dumps(r, ensure_ascii=False) + "\n")
    if n % 25 == 0: print(n, q, task, "base", r["base_ok"], "target", r["target_ok"], "control", r["control_ok"], flush=True)
print("NECESSITY_DONE", args.tag)
