#!/usr/bin/env python3
import argparse, json, math, sys, time, torch
from pathlib import Path
from PIL import Image
BASE = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(BASE))
from cf_mp_common import load, build, page_token_positions, image_token_id
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--backend", required=True); ap.add_argument("--geo", required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--no-thinking-flag", action="store_true")
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--qids", default=""); ap.add_argument("--out", default=""); ap.add_argument("--check", action="store_true")
args = ap.parse_args()
D = Path(__file__).parent
OUT = Path(args.out) if args.out else D / f"occl_main_{args.tag}.jsonl"
cells_all = json.load(open(D / "occl_main_cells.json"))
qids = args.qids.split(",") if args.qids else json.load(open(D / "occl_main_items.json"))[args.tag]
items = {json.loads(l)["qid"]: json.loads(l) for l in open(BASE / f"common/items_sp_{args.geo}.jsonl")}
done = {(json.loads(l)["qid"], json.loads(l)["task"]) for l in open(OUT)} if OUT.exists() else set()

def cell_tokens(bbox, W, H, gh, gw):
    if args.backend == "mistral3": sx = 28 * W / (math.ceil(W / 14) * 14); sy = 28 * H / (math.ceil(H / 14) * 14)
    else: sx, sy = W / gw, H / gh
    x1, y1, x2, y2 = bbox
    c0, c1 = max(0, int(x1 // sx)), min(gw - 1, int((x2 - 1e-6) // sx)); r0, r1 = max(0, int(y1 // sy)), min(gh - 1, int((y2 - 1e-6) // sy))
    return [r * gw + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

proc, model = load(args.model, args.backend); tok = proc.tokenizer; itid = image_token_id(model, args.backend)
lm = getattr(model.model, "language_model", model.model); layers = lm.layers
st = {"rows": None, "mean": None, "capture": None}
def hook(module, hargs, hkwargs):
    hs = hargs[0] if hargs else hkwargs["hidden_states"]
    if hs.shape[1] > 1:
        if st["capture"] is not None: st["mean"] = hs[0, st["capture"], :].mean(0).detach().clone(); st["capture"] = None
        if st["rows"] is not None:
            for b, rows in enumerate(st["rows"]):
                if rows is not None and len(rows): hs[b, rows, :] = st["mean"]
    return None
layers[0].register_forward_pre_hook(hook, with_kwargs=True)

def with_answer(inp, ans_ids):
    L = inp["input_ids"].shape[1]; n = len(ans_ids); out = {}
    for k, v in inp.items():
        if torch.is_tensor(v) and v.dim() == 2 and v.shape[0] == 1 and v.shape[1] == L:
            ext = torch.tensor([ans_ids], device=v.device, dtype=v.dtype) if k == "input_ids" else (torch.ones if k == "attention_mask" else torch.zeros)((1, n), device=v.device, dtype=v.dtype)
            out[k] = torch.cat([v, ext], 1)
        else: out[k] = v
    return out, L, n
def rep(inp, bs):
    return {k: (torch.cat([v] * bs, 0) if torch.is_tensor(v) else v) for k, v in inp.items()}
def logprobs(inp, L, n, ans_ids, rows_list):
    st["rows"] = rows_list
    with torch.no_grad(): lo = model(**rep(inp, len(rows_list))).logits[:, L - 1:L - 1 + n].float().log_softmax(-1)
    st["rows"] = None
    idx = torch.tensor(ans_ids, device=lo.device)
    return lo[:, torch.arange(n), idx].sum(1).tolist()

t0 = time.time()
for q in qids:
    it = items[q]; cinfo = cells_all[q]; img = Image.open(it["base_image"]).convert("RGB"); W, H = img.size
    ow, oh = cinfo["orig_size"]; scale = (W / ow, H / oh)
    gh, gw = it["grid_hw"]; A, B = it["operands"]
    btok_ok = cell_tokens(B["bbox"], W, H, gh, gw) == sorted(it["vision_token_indices"])
    for task, question, gold in (("lookup_b", it["lookup_b_question"], it["b_value"]), ("compute", it["compute_question"], it["compute_gold"])):
        if (q, task) in done: continue
        inp, grids = build(proc, model, [img], question, args.backend, not args.no_thinking_flag)
        ids = inp["input_ids"][0]; pos = page_token_positions(ids, itid, it, grids)[0]
        ans_ids = tok(gold, add_special_tokens=False)["input_ids"]; inp2, L, n = with_answer(inp, ans_ids)
        st["capture"] = (ids == itid).nonzero().squeeze(1)
        lp0 = logprobs(inp2, L, n, ans_ids, [None])[0]
        toks = []
        for c in cinfo["cells"]:
            bb = [c["bbox"][0] * scale[0], c["bbox"][1] * scale[1], c["bbox"][2] * scale[0], c["bbox"][3] * scale[1]]
            toks.append(cell_tokens(bb, W, H, gh, gw))
        rows = [torch.tensor([pos[k] for k in t], device=model.device) for t in toks]
        lps = []
        for s in range(0, len(rows), args.bs): lps += logprobs(inp2, L, n, ans_ids, rows[s:s + args.bs])
        if args.check:
            single = [logprobs(inp2, L, n, ans_ids, [r])[0] for r in rows[:16]]
            print("CHECK max|batch-single| =", max(abs(a - b) for a, b in zip(lps[:16], single)), "lp0", lp0, flush=True)
        res = []
        for ci, (c, t, lp) in enumerate(zip(cinfo["cells"], toks, lps)):
            key = (c["row_idx"], c["col_idx"])
            role = "B" if key == (B["row_idx"], B["col_idx"]) else ("A" if key == (A["row_idx"], A["col_idx"]) else "")
            res.append({"i": ci, "row": c["row_idx"], "col": c["col_idx"], "hdr": bool(c.get("is_header")), "value": c["value"], "role": role, "tok": t, "delta": round(lp0 - lp, 5)})
        r = {"qid": q, "task": task, "model": args.tag, "gold": gold, "lp_clean": round(lp0, 5), "b_tokens_match_pool": btok_ok, "grid": [gh, gw], "cells": res}
        open(OUT, "a").write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{q} {task} lp0={lp0:.3f} cells={len(res)} Btok={btok_ok} t={time.time()-t0:.0f}s", flush=True)
        if args.check: sys.exit(0)
print("OCCL_MAIN_DONE", args.tag)
