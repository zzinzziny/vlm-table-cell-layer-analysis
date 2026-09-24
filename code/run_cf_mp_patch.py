#!/usr/bin/env python3
import argparse, json, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from cf_mp_common import load, build, b_rows, image_token_id, page_token_positions
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--backend", default="qwen"); ap.add_argument("--no-thinking-flag", action="store_true")
ap.add_argument("--plan", required=True); ap.add_argument("--qids", default=""); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max-new-tokens", type=int, default=16)
ap.add_argument("--ring", type=int, default=0, help='patch the B-cell tokens plus same-page grid neighbours within radius R (0 = cell tokens only)')
args = ap.parse_args()
plan = [json.loads(l) for l in open(args.plan)]
if args.qids:
    keep = {l.strip() for l in open(args.qids) if l.strip()}; plan = [p for p in plan if p["qid"] in keep]
if args.limit: plan = plan[:args.limit]
proc, model = load(args.model, args.backend); tok = proc.tokenizer
lm = getattr(model.model, "language_model", model.model); layers = lm.layers; n_layers = len(layers); itid = image_token_id(model, args.backend)
cap = {"store": None, "rows": None}; patch = {"layer": None, "rows": None, "donor": None}
def mk(l):
    def hook(module, hargs, hkwargs):
        hs = hargs[0] if hargs else hkwargs["hidden_states"]
        if hs.shape[1] == 1: return None
        if cap["store"] is not None: cap["store"][l] = hs[:, cap["rows"], :].detach().clone()
        if patch["layer"] == l: hs[:, patch["rows"], :] = patch["donor"][l]
        return None
    return hook
for l, layer in enumerate(layers): layer.register_forward_pre_hook(mk(l), with_kwargs=True)
def gen(inp):
    with torch.no_grad(): out = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False, output_scores=True, return_dict_in_generate=True)
    n = inp["input_ids"].shape[1]; return tok.decode(out.sequences[0][n:], skip_special_tokens=True).strip(), F.log_softmax(out.scores[0][0].float(), dim=-1)
def ft(s): return tok(str(s), add_special_tokens=False)["input_ids"][0]
with open(args.out, "w") as fout:
    for i, it in enumerate(plan):
        base = [Image.open(p).convert("RGB") for p in it["base_images"]]; donor = [Image.open(p).convert("RGB") for p in it["donor_images"]]
        for role in ("lookup_b", "compute"):
            q = it["compute_question"] if role == "compute" else it["lookup_b_question"]
            ans_base, ans_donor = (it["compute_gold"], it["compute_gold_new"]) if role == "compute" else (it["b_value"], it["b_new"])
            toks = {"base": ft(ans_base), "donor": ft(ans_donor)}
            in_b, gb = build(proc, model, base, q, args.backend, not args.no_thinking_flag); in_d, gd = build(proc, model, donor, q, args.backend, not args.no_thinking_flag)
            assert torch.equal(in_b["input_ids"], in_d["input_ids"])
            rows = b_rows(in_b["input_ids"][0], itid, it, gb)
            if args.ring:
                bpos = page_token_positions(in_b["input_ids"][0], itid, it, gb)[it["b_page_pos"]]; gh, gw = it["grid_hw"]; ks = set()
                for k in it["vision_token_indices"]:
                    r0, c0 = divmod(k, gw)
                    ks |= {r * gw + c for r in range(max(0, r0 - args.ring), min(gh, r0 + args.ring + 1)) for c in range(max(0, c0 - args.ring), min(gw, c0 + args.ring + 1))}
                rows = [bpos[k] for k in sorted(ks)]
            cap["store"], cap["rows"] = {}, rows; pred_d, lp_d = gen(in_d); donor_store = cap["store"]; cap["store"] = None
            pred_b, lp_b = gen(in_b)
            def row(layer, pred, lp): return {"qid": it["qid"], "role": role, "layer": layer, "pred": pred, "ans_base": ans_base, "ans_donor": ans_donor, "lp_base_tok": round(float(lp[toks["base"]]), 4), "lp_donor_tok": round(float(lp[toks["donor"]]), 4)}
            fout.write(json.dumps(row(-1, pred_b, lp_b)) + "\n"); fout.write(json.dumps(row(-2, pred_d, lp_d)) + "\n")
            print(f"[{i+1}/{len(plan)}] {it['qid']} {role}: base={pred_b!r} donor={pred_d!r} (want {ans_base!r}/{ans_donor!r}) rows={len(rows)}", flush=True)
            for l in range(n_layers):
                patch.update(layer=l, rows=rows, donor=donor_store); pred_p, lp_p = gen(in_b); patch["layer"] = None
                fout.write(json.dumps(row(l, pred_p, lp_p)) + "\n")
            fout.flush()
print(f"-> {args.out}")
