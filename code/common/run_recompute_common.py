#!/usr/bin/env python3
import argparse, json, re, sys, unicodedata
from pathlib import Path
import torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cf_mp_common import load, image_token_id, build, page_token_positions, INSTR
COMBOS = [("d2c", "copy", "donor2_compute", None), ("d2d", "delta", "donor2_compute", "donor2b_compute"), ("lkd", "delta", "donor_lookup", "base_lookup"),
          ("sac", "copy", "donor_compute", None), ("d3c", "copy", "donor3_compute", None), ("d3d", "delta", "donor3_compute", "donor3b_compute")]
def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).strip().lower(); return re.sub(r"\s+", " ", s.replace("−", "-").replace(",", "")).rstrip(".")
def other(op): return "sum" if op == "difference" else "difference"
def apply(op, x, y): return str(x + y) if op == "sum" else str(abs(x - y))
def prompts(it):
    a, b, a2 = it["operands"][0], it["operands"][1], it["a2"]; col = b["col_header"]
    prefix = f"In the table, find the row '{b['row_header']}' and look at its value in the '{col}' column."
    def comp(row, op):
        if op == "sum": return prefix + f" What is the sum of that value and the '{col}' value for the row '{row}'?"
        return prefix + f" What is the difference between the '{col}' value for the row '{row}' and that value?"
    return prefix, prefix + " What is that value?", comp(a["row_header"], it["op"]), comp(a2["row_header"], it["op"]), comp(a2["row_header"], other(it["op"]))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True); ap.add_argument("--geometry", default="qwen35"); ap.add_argument("--model", required=True); ap.add_argument("--backend", default="qwen")
    ap.add_argument("--out", required=True); ap.add_argument("--qids", default=""); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sites", default="b_cell,text_all"); ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--skip-from", default="", help='comma-separated jsonl files: qids found there are skipped (re-checked per item; for splitting across GPUs)'); ap.add_argument("--reverse", action="store_true", help='process items in reverse order (for splitting across GPUs)'); ap.add_argument("--start-frac", type=float, default=0.0, help='start from this fraction of the items to the end, then the beginning (for splitting across GPUs)')
    ap.add_argument("--no-thinking-flag", action="store_true"); ap.add_argument("--max-soft-tokens", type=int, default=1120); ap.add_argument("--no-lp", action="store_true", help='skip the continuous score (teacher-forced log-prob)')
    args = ap.parse_args(); sites = args.sites.split(",")
    items = [json.loads(l) for l in open(args.items)]
    if args.qids: keep = {l.strip() for l in open(args.qids) if l.strip()}; items = [i for i in items if i.get("item_id", i.get("qid")) in keep]
    items = [i for i in items if i.get("a2")]
    if args.limit: items = items[:args.limit]
    if args.resume and Path(args.out).exists():
        done = {json.loads(l)["qid"] for l in open(args.out)}; items = [i for i in items if i.get("item_id", i.get("qid")) not in done]
    proc, model = load(args.model, args.backend); tok = proc.tokenizer; itid = image_token_id(model, args.backend)
    lm = getattr(model.model, "language_model", model.model); layers = lm.layers; n_layers = len(layers)
    cap = {"store": None, "needed": None}; patch = {"layer": None, "rows": None, "state": None}
    def make_pre_hook(l):
        def pre_hook(module, hargs, hkwargs):
            hs = hargs[0] if hargs else hkwargs["hidden_states"]
            if hs.shape[1] == 1: return None
            if cap["store"] is not None: cap["store"][l] = hs[:, cap["needed"], :].detach().clone()
            if patch["layer"] == l: hs[:, patch["rows"], :] = patch["state"]
            return None
        return pre_hook
    for l, layer in enumerate(layers): layer.register_forward_pre_hook(make_pre_hook(l), with_kwargs=True)
    def b_tok_indices(it):
        if "geometry" in it and args.geometry in it["geometry"]: return it["geometry"][args.geometry]["vision_token_indices"]
        return it["vision_token_indices"]
    def spans_after_vision(ids):
        ids_l = ids.tolist(); last_vis = max(i for i, t in enumerate(ids_l) if t == itid); spans, acc = [], ""
        for t in ids_l[last_vis + 1:]:
            s = tok.decode([t]); spans.append((len(acc), len(acc) + len(s))); acc += s
        return last_vis, spans, acc
    def prefix_positions(ids, prefix):
        last_vis, spans, acc = spans_after_vision(ids); k = acc.find(prefix); assert k >= 0, "prefix not found"
        return [last_vis + 1 + j for j, (s0, s1) in enumerate(spans) if s1 > k and s0 < k + len(prefix)]
    def tail_positions(ids, question):
        last_vis, spans, acc = spans_after_vision(ids); k = acc.find(question); assert k >= 0, "question not found"; end = k + len(question)
        return [last_vis + 1 + j for j, (s0, s1) in enumerate(spans) if s0 >= end]
    def gen(inp):
        with torch.no_grad(): o = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
        return tok.decode(o[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    def seq_lp(inp, answer):
        ans_ids = tok(str(answer), add_special_tokens=False)["input_ids"]
        ids = torch.cat([inp["input_ids"], torch.tensor([ans_ids], device=model.device)], 1)
        kw = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
        if "attention_mask" in inp: kw["attention_mask"] = torch.ones_like(ids)
        if "mm_token_type_ids" in inp: kw["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros((1, len(ans_ids)), dtype=inp["mm_token_type_ids"].dtype, device=model.device)], 1)
        with torch.no_grad(): out = model(input_ids=ids, **kw)
        lp = F.log_softmax(out.logits[0, inp["input_ids"].shape[1] - 1:-1].float(), -1)
        return round(float(lp.gather(1, torch.tensor(ans_ids, device=model.device)[:, None]).sum()), 3)
    if args.reverse: items = items[::-1]
    if args.start_frac > 0: k = int(len(items) * args.start_frac); items = items[k:] + items[:k]
    skip_files = [f for f in args.skip_from.split(",") if f]
    def done_elsewhere():
        s = set()
        for f in skip_files:
            try:
                for l in open(f): s.add(json.loads(l)["qid"])
            except Exception: pass
        return s
    with open(args.out, "a" if args.resume else "w") as fout:
        for i, it in enumerate(items):
            qid = it.get("item_id", it.get("qid"))
            if skip_files and qid in done_elsewhere(): continue
            A, B, Bp, A2 = int(it["operands"][0]["value"]), int(it["b_value"]), int(it["b_new"]), int(it["a2"]["value"]); f, g = it["op"], other(it["op"])
            vals = {"R": apply(f, A, B), "r": apply(f, A, Bp), "d": apply(f, A2, Bp), "z": apply(f, A2, B), "e": apply(g, A2, Bp), "y": apply(g, A2, B), "B": str(B), "b": str(Bp)}
            inv = {}
            for k, v in vals.items(): inv.setdefault(norm(v), k)
            prefix, q_l, q_cA, q_cA2, q_cA2g = prompts(it)
            base_imgs = [Image.open(p).convert("RGB") for p in it.get("base_images", [it["base_image"]])]; donor_imgs = [Image.open(p).convert("RGB") for p in it.get("donor_images", [it["donor_image"]])]
            runs = {"base_compute": (base_imgs, q_cA), "donor_compute": (donor_imgs, q_cA), "donor2_compute": (donor_imgs, q_cA2), "donor2b_compute": (base_imgs, q_cA2),
                    "donor3_compute": (donor_imgs, q_cA2g), "donor3b_compute": (base_imgs, q_cA2g), "base_lookup": (base_imgs, q_l), "donor_lookup": (donor_imgs, q_l)}
            targets = {"base_compute": "R", "donor_compute": "r", "donor2_compute": "d", "donor2b_compute": "z", "donor3_compute": "e", "donor3b_compute": "y", "base_lookup": "B", "donor_lookup": "b"}
            it_pages = it if "b_page_pos" in it else {**it, "b_page_pos": 0, "pages": it["pages"][:1]}
            inputs, pos, store, gate, preds = {}, {}, {}, {}, {}
            try:
                for name, (imgs, q) in runs.items():
                    inp, grids = build(proc, model, imgs, q, args.backend, thinking_flag=not args.no_thinking_flag, max_soft_tokens=args.max_soft_tokens); ids = inp["input_ids"][0]
                    bpos = page_token_positions(ids, itid, it_pages, grids)[it_pages["b_page_pos"]]
                    p = {"b_cell": [bpos[k] for k in b_tok_indices(it)], "prefix": prefix_positions(ids.cpu(), prefix), "tail": tail_positions(ids.cpu(), q)}
                    p["text_all"] = p["prefix"] + p["tail"]; pos[name] = p
                    needed = sorted(set(sum(p.values(), []))); cap["store"], cap["needed"] = {}, torch.tensor(needed, device=model.device)
                    pred = gen(inp); st = cap["store"]; cap["store"] = None; row_of = {q_: j for j, q_ in enumerate(needed)}
                    store[name] = {s: {l: st[l][:, [row_of[q_] for q_ in p[s]], :] for l in range(n_layers)} for s in sites}
                    preds[name] = pred; gate[name] = (norm(pred) == norm(vals[targets[name]])); inputs[name] = inp
            except AssertionError as ex:
                fout.write(json.dumps({"qid": qid, "skip": f"position:{ex}"}) + "\n"); fout.flush(); print(f"[{i+1}/{len(items)}] {qid} skip {ex}", flush=True); continue
            core = all(gate[k] for k in ("base_compute", "donor_compute", "donor2_compute", "donor2b_compute", "base_lookup", "donor_lookup")); all8 = all(gate.values())
            row = {"qid": qid, "op": f, "A": str(A), "B": str(B), "Bp": str(Bp), "A2": str(A2), "vals": vals, "gates": gate, "gate_preds": preds, "gate_core6": core, "gate_all8": all8,
                   "n_b_tok": len(pos["base_compute"]["b_cell"]), "n_text_tok": len(pos["base_compute"]["text_all"]), "n_layers": n_layers}
            if not core:
                row["skip"] = "gate"; fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
                print(f"[{i+1}/{len(items)}] {qid} gates {''.join('P' if v else '.' for v in gate.values())} -- skip", flush=True); del inputs, store; torch.cuda.empty_cache(); continue
            res, raw, lps = {}, {}, {}
            rname = "base_compute"
            for tag, mode, plus, minus in COMBOS:
                if plus.startswith("donor3") and not all8: continue
                for s in sites:
                    if not (len(pos[rname][s]) == len(pos[plus][s]) and (minus is None or len(pos[minus][s]) == len(pos[rname][s]))):
                        res[f"{tag}@{s}"] = "misaligned"; continue
                    rows_t = torch.tensor(pos[rname][s], device=model.device); seq, rw, lpseq = [], [], []
                    for l in range(n_layers):
                        patch["layer"], patch["rows"] = l, rows_t
                        patch["state"] = store[plus][s][l] if mode == "copy" else store[rname][s][l] + args.alpha * (store[plus][s][l] - store[minus][s][l])
                        pred = gen(inputs[rname]); seq.append(inv.get(norm(pred), ".")); rw.append(pred)
                        if not args.no_lp: lpseq.append([seq_lp(inputs[rname], vals["r"]), seq_lp(inputs[rname], vals["R"])])
                    patch["layer"] = None
                    res[f"{tag}@{s}"] = "".join(seq); raw[f"{tag}@{s}"] = rw
                    if lpseq: lps[f"{tag}@{s}"] = lpseq
            row["patch"], row["raw"], row["lp_rR"] = res, raw, lps
            fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
            print(f"[{i+1}/{len(items)}] {qid} PASS{'(8)' if all8 else '(6)'} " + " | ".join(f"{k}={v}" for k, v in res.items() if k.endswith("@b_cell")), flush=True)
            del inputs, store; torch.cuda.empty_cache()
    print("->", args.out)
if __name__ == "__main__": main()
