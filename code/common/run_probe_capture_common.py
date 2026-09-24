#!/usr/bin/env python3
import argparse, json, sys, re, unicodedata
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cf_mp_common import load, image_token_id, build, page_token_positions, INSTR
def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).strip().lower(); return re.sub(r"\s+", " ", s.replace("−", "-").replace(",", "")).rstrip(".")
SITES = ["b_cell", "a_cell", "last", "tail"]; PROMPTS = ["lookup_b", "compute"]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True); ap.add_argument("--model", required=True); ap.add_argument("--backend", default="qwen")
    ap.add_argument("--outdir", required=True); ap.add_argument("--resume", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-thinking-flag", action="store_true"); ap.add_argument("--max-soft-tokens", type=int, default=1120); ap.add_argument("--max-new-tokens", type=int, default=8)
    args = ap.parse_args(); outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in open(args.items)]
    if args.limit: items = items[:args.limit]
    done = {p.stem for p in outdir.glob("*.npz")} if args.resume else set()
    items = [it for it in items if it["qid"] not in done]
    proc, model = load(args.model, args.backend); tok = proc.tokenizer; itid = image_token_id(model, args.backend)
    def gen(inp):
        with torch.no_grad(): o = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
        return tok.decode(o[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    def tail_positions(ids, question):
        ids_l = ids.tolist(); last_vis = max(i for i, t in enumerate(ids_l) if t == itid)
        spans, acc = [], ""
        for t in ids_l[last_vis + 1:]:
            s = tok.decode([t]); spans.append((len(acc), len(acc) + len(s))); acc += s
        k = acc.find(question); assert k >= 0, "question not found in decoded tail"; end = k + len(question)
        return [last_vis + 1 + j for j, (s0, s1) in enumerate(spans) if s0 >= end]
    meta_f = open(outdir / "meta.jsonl", "a" if args.resume else "w")
    for i, it in enumerate(items):
        images = [Image.open(p).convert("RGB") for p in it.get("base_images", [it["base_image"]])]
        it_pages = it if "b_page_pos" in it else {**it, "b_page_pos": 0, "pages": it["pages"][:1]}
        qs = {"lookup_b": (it["lookup_b_question"], it["b_value"]), "compute": (it["compute_question"], it["compute_gold"])}
        feats, preds, ok = [], {}, {}
        for pr in PROMPTS:
            q, gold = qs[pr]
            inp, grids = build(proc, model, images, q, args.backend, thinking_flag=not args.no_thinking_flag, max_soft_tokens=args.max_soft_tokens)
            ids = inp["input_ids"][0]
            pages_pos = page_token_positions(ids, itid, it_pages, grids)
            bpos = pages_pos[it_pages["b_page_pos"]]; apos = pages_pos[it.get("a_page_pos", it_pages["b_page_pos"])]
            b_rows = [bpos[k] for k in it["operands"][1]["vision_token_indices"]]; a_rows = [apos[k] for k in it["operands"][0]["vision_token_indices"]]
            tail = tail_positions(ids.cpu(), q); last = [ids.shape[0] - 1]
            with torch.no_grad(): out = model(**inp, output_hidden_states=True)
            hs = out.hidden_states  # L+1 x [1, T, d]
            arr = np.stack([np.stack([h[0, rows, :].float().mean(0).cpu().numpy() for rows in (b_rows, a_rows, last, tail)]) for h in hs]).astype(np.float16)
            feats.append(arr); del out, hs; torch.cuda.empty_cache()
            preds[pr] = gen(inp); ok[pr] = norm(preds[pr]) == norm(gold)
        np.savez_compressed(outdir / f"{it['qid']}.npz", feat=np.stack(feats))
        meta_f.write(json.dumps({"qid": it["qid"], "doc_id": it["doc_id"], "A": it["operands"][0]["value"], "B": it["b_value"], "R": it["compute_gold"], "op": it["op"],
                                 "n_b_tok": len(b_rows), "n_a_tok": len(a_rows), "n_tail": len(tail), "preds": preds, "ok": ok, "n_layers_plus1": int(feats[0].shape[0]), "sites": SITES, "prompts": PROMPTS}, ensure_ascii=False) + "\n"); meta_f.flush()
        print(f"[{i+1}/{len(items)}] {it['qid']} lookup={preds['lookup_b']!r}({'P' if ok['lookup_b'] else '.'}) compute={preds['compute']!r}({'P' if ok['compute'] else '.'}) shape={feats[0].shape}", flush=True)
    print("-> done", outdir)
if __name__ == "__main__": main()
