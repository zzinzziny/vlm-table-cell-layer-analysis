#!/usr/bin/env python3
"""Counterfactual cell-value patching: donor(B') -> base(B), B-cell rows only.

Same table, two image versions (base: B, donor: B' edited in the cell).
For each question role (lookup_b / compute), capture the donor run's
layer-l INPUT hidden states at the B cell's vision token rows, then re-run
the base prompt patching those rows at exactly one layer l (prefill only)
and greedily generate the answer.

Question: does the SAME cell-representation swap make
  LOOKUP answer B -> B'   (copy), and
  COMPUTE answer R -> R'=f(A,B')   (transform)?

Gates (from full-donor generation, recorded per item):
  G2 lookup_b(donor) == B'    G3 compute(donor) == R'

Output rows: one per (qid, role, layer) + baselines (layer=-1 base,
layer=-2 donor). Usage:
  CUDA_VISIBLE_DEVICES=1 python run_cf_patch.py [--limit N]
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

BASE = Path(__file__).resolve().parent
MODEL_ID = "google/gemma-4-12B-it"
PLAN = BASE / "cf_images/plan.json"
OUT = BASE / "cf_patch_qwen35_9b.jsonl"
INSTR = ("Reply with the answer only. No explanation, no reasoning, "
         "no bullet points.")


def get_lm(model):
    return getattr(model.model, "language_model", model.model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--no-thinking-flag", action="store_true")
    ap.add_argument("--max-soft-tokens", type=int, default=1120)
    ap.add_argument("--plan", default=str(PLAN),
                    help=".json list or .jsonl of plan rows")
    ap.add_argument("--qids", default="",
                    help="file with one qid per line; restrict to these")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--ring", type=int, default=0,
                    help='patch the B-cell tokens plus grid neighbours within radius R (0 = cell tokens only)')
    args = ap.parse_args()

    if args.plan.endswith(".jsonl"):
        plan = [json.loads(l) for l in open(args.plan)]
    else:
        plan = json.load(open(args.plan))
    if args.qids:
        keep = {l.strip() for l in open(args.qids) if l.strip()}
        plan = [p for p in plan if p["qid"] in keep]
    if args.limit:
        plan = plan[:args.limit]

    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(args.model)
    tok = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    lm = get_lm(model)
    layers = lm.layers
    n_layers = len(layers)
    image_token_id = model.config.image_token_id
    print(f"decoder layers: {n_layers}", flush=True)

    cap = {"store": None, "rows": None}
    patch = {"layer": None, "rows": None, "donor": None}

    def make_pre_hook(l):
        def pre_hook(module, hargs, hkwargs):
            hs = hargs[0] if hargs else hkwargs["hidden_states"]
            if hs.shape[1] == 1:          # decode step - skip
                return None
            if cap["store"] is not None:
                cap["store"][l] = hs[:, cap["rows"], :].detach().clone()
            if patch["layer"] == l:
                hs[:, patch["rows"], :] = patch["donor"][l]
            return None
        return pre_hook

    for l, layer in enumerate(layers):
        layer.register_forward_pre_hook(make_pre_hook(l), with_kwargs=True)

    def build(img, question):
        content = [{"type": "image", "image": img},
                   {"type": "text", "text": f"{question}\n{INSTR}"}]
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, return_dict=True,
            add_generation_prompt=True, return_tensors="pt", max_soft_tokens=args.max_soft_tokens)
        inputs = {k: v for k, v in inputs.items() if k != "num_soft_tokens_per_image"}
        return {k: v.to(model.device) for k, v in inputs.items()}

    def gen(inputs):
        with torch.no_grad():
            out = model.generate(**inputs,
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=False, output_scores=True,
                                 return_dict_in_generate=True)
        n_in = inputs["input_ids"].shape[1]
        text = tok.decode(out.sequences[0][n_in:],
                          skip_special_tokens=True).strip()
        first_lp = F.log_softmax(out.scores[0][0].float(), dim=-1)
        return text, first_lp

    def ft(s):
        return tok(str(s), add_special_tokens=False)["input_ids"][0]

    with open(args.out, "w") as fout:
        for i, it in enumerate(plan):
            base_img = Image.open(it["base_image"]).convert("RGB")
            donor_img = Image.open(it["donor_image"]).convert("RGB")
            for role in ("lookup_b", "compute"):
                q = it[f"{role}_question"] if role == "compute" \
                    else it["lookup_b_question"]
                if role == "compute":
                    q = it["compute_question"]
                    ans_base, ans_donor = (it["compute_gold"],
                                           it["compute_gold_new"])
                else:
                    ans_base, ans_donor = it["b_value"], it["b_new"]
                toks = {"base": ft(ans_base), "donor": ft(ans_donor)}

                in_b = build(base_img, q)
                in_d = build(donor_img, q)
                assert torch.equal(in_b["input_ids"], in_d["input_ids"])
                ids = in_b["input_ids"][0]
                vis = (ids == image_token_id).nonzero().squeeze(1)
                block0 = int(vis[0])
                assert int(vis[-1]) - block0 + 1 == len(vis), "split block"
                rows = [block0 + k for k in it["vision_token_indices"]]
                if args.ring:
                    gh, gw = it["grid_hw"]
                    assert len(vis) == gh * gw, ("grid", len(vis), gh, gw)
                    ks = set()
                    for k in it["vision_token_indices"]:
                        r0, c0 = divmod(k, gw)
                        ks |= {r * gw + c
                               for r in range(max(0, r0 - args.ring), min(gh, r0 + args.ring + 1))
                               for c in range(max(0, c0 - args.ring), min(gw, c0 + args.ring + 1))}
                    rows = [block0 + k for k in sorted(ks)]

                # donor pass: capture all layer inputs at B-cell rows
                cap["store"], cap["rows"] = {}, rows
                pred_d, lp_d = gen(in_d)
                donor_store = cap["store"]
                cap["store"] = None
                assert len(donor_store) == n_layers

                pred_b, lp_b = gen(in_b)

                def row(layer, pred, lp):
                    return {"qid": it["qid"], "role": role, "layer": layer,
                            "pred": pred, "ans_base": ans_base,
                            "ans_donor": ans_donor,
                            "lp_base_tok": round(float(lp[toks["base"]]), 4),
                            "lp_donor_tok": round(float(lp[toks["donor"]]), 4)}

                fout.write(json.dumps(row(-1, pred_b, lp_b)) + "\n")
                fout.write(json.dumps(row(-2, pred_d, lp_d)) + "\n")
                print(f"[{i+1}/{len(plan)}] {it['qid']} {role}: "
                      f"base={pred_b!r} donor={pred_d!r} "
                      f"(want {ans_base!r}/{ans_donor!r})", flush=True)

                for l in range(n_layers):
                    patch.update(layer=l, rows=rows, donor=donor_store)
                    pred_p, lp_p = gen(in_b)
                    patch["layer"] = None
                    fout.write(json.dumps(row(l, pred_p, lp_p)) + "\n")
                fout.flush()
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
