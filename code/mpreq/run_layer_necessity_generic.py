#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import torch, torch.nn.functional as F
from PIL import Image
BASE = Path(__file__).resolve().parent; sys.path.insert(0, str(BASE.parent))
from cf_mp_common import load, build, image_token_id  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True); ap.add_argument("--out", required=True); ap.add_argument("--model", required=True); ap.add_argument("--backend", default="qwen")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--qids", default=""); ap.add_argument("--sp-only", action="store_true"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-thinking-flag", action="store_true"); ap.add_argument("--max-soft-tokens", type=int, default=1120)
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.items)]
    if args.qids:
        keep = {l.strip() for l in open(args.qids) if l.strip()}; items = [it for it in items if it["qid"] in keep]
    if args.sp_only: items = [it for it in items if it["n_pages"] == 1]
    if args.limit: items = items[:args.limit]
    done = set()
    if args.resume and Path(args.out).exists():
        done = {json.loads(l)["qid"] for l in open(args.out)}; items = [it for it in items if it["qid"] not in done]
    print(f"items: {len(items)} (done {len(done)})", flush=True)

    proc, model = load(args.model, args.backend); tok = proc.tokenizer; itid = image_token_id(model, args.backend)
    lm = getattr(model.model, "language_model", model.model); layers = lm.layers; n_layers = len(layers)
    print(f"backend {args.backend}, decoder layers {n_layers}, image token id {itid}", flush=True)

    corrupt = {"rows": None, "page_pos": None}
    cap = {"store": None, "needed": None}; patch = {"layer": None, "rows": None, "state": None}; zero = {"layer": None, "rows": None}

    def make_pre_hook(l):
        def pre_hook(module, hargs, hkwargs):
            hs = hargs[0] if hargs else hkwargs["hidden_states"]
            if hs.shape[1] == 1: return None
            if l == 0 and corrupt["rows"] is not None:
                rows_t = torch.tensor(corrupt["rows"], device=hs.device)
                for pos in corrupt["page_pos"]:
                    pos_t = torch.tensor(pos, device=hs.device)
                    sel = rows_t[torch.isin(rows_t, pos_t)]
                    if len(sel): hs[:, sel, :] = hs[:, pos_t, :].mean(dim=1, keepdim=True).to(hs.dtype)
            dev = lambda t: t.to(hs.device) if torch.is_tensor(t) else t
            if cap["store"] is not None: cap["store"][l] = hs[:, dev(cap["needed"]), :].detach().clone()
            if patch["layer"] == l: hs[:, dev(patch["rows"]), :] = patch["state"].to(hs.device)
            if zero["layer"] == l: hs[:, dev(zero["rows"]), :] = 0
            return None
        return pre_hook
    for l, layer in enumerate(layers): layer.register_forward_pre_hook(make_pre_hook(l), with_kwargs=True)

    def score(full, prompt_len, gold_ids):
        with torch.no_grad(): logits = model(**full).logits[0]
        lp = F.log_softmax(logits[prompt_len - 1:prompt_len - 1 + len(gold_ids)].float(), dim=-1)
        tok_lp = lp.gather(1, torch.tensor(gold_ids, device=lp.device)[:, None]).squeeze(1)
        return tok_lp.mean().item(), tok_lp[0].item(), [round(x, 4) for x in tok_lp.tolist()]

    with open(args.out, "a" if args.resume else "w") as fout:
        for i, it in enumerate(items):
            images = [Image.open(p["image"]).convert("RGB") for p in it["pages"]]
            inputs, grids = build(proc, model, images, it["question"], args.backend, thinking_flag=not args.no_thinking_flag, max_soft_tokens=args.max_soft_tokens)
            ids = inputs["input_ids"][0]; vis = (ids == itid).nonzero().squeeze(1).tolist()
            page_pos, off, bad = {}, 0, False
            for pi, page in enumerate(it["pages"]):
                gh, gw = (grids[pi] if grids is not None else page["grid_hw"])
                if [gh, gw] != list(page["grid_hw"]): bad = True; break
                page_pos[page["page_id"]] = vis[off:off + gh * gw]; off += gh * gw
            if bad or off != len(vis):
                print(f"[{it['qid']}] grid mismatch (item {[p['grid_hw'] for p in it['pages']]} vs proc {grids}, vis {len(vis)} vs {off}) -- skip", flush=True); continue
            prompt_len = ids.shape[0]

            def rows_of(cells): return sorted({page_pos[c["page_id"]][k] for c in cells for k in c["vision_token_indices"]})
            group = it.get("group", "compute")
            if group == "sweep":
                ev_rows = rows_of([it["target"]]); ctl_rows = rows_of([it["control"]])
                key_rows = rows_of(it["key_cells"]) if it.get("key_cells") else (rows_of([it["key_cell"]]) if it.get("key_cell") else None)
            else:
                ev_rows = rows_of(it["operands"]); ctl_rows = rows_of(it["controls"]) if it.get("controls") else None
                key_rows = rows_of(it["key_cells"]) if it.get("key_cells") else None
            sites = {"evidence_visual": ev_rows, "last_text": [prompt_len - 1]}
            if ctl_rows: sites["control_visual"] = ctl_rows
            if key_rows: sites["key_visual"] = key_rows

            gold_ids = tok(it["gold"], add_special_tokens=False)["input_ids"]
            full = dict(inputs); full["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([gold_ids], dtype=torch.long, device=model.device)], dim=1)
            full["attention_mask"] = torch.ones_like(full["input_ids"])
            if "mm_token_type_ids" in inputs:
                mm = inputs["mm_token_type_ids"]; full["mm_token_type_ids"] = torch.cat([mm, torch.zeros((1, len(gold_ids)), dtype=mm.dtype, device=mm.device)], dim=1)
            needed = sorted({p for ps in sites.values() for p in ps}); row_of = {p: j for j, p in enumerate(needed)}
            needed_t = torch.tensor(needed, device=model.device)
            stensors = {s: (torch.tensor(ps, device=model.device), [row_of[p] for p in ps]) for s, ps in sites.items()}
            pp = list(page_pos.values())

            cap["store"], cap["needed"] = {}, needed_t; corrupt["rows"], corrupt["page_pos"] = ev_rows, pp
            corrupt_lp, corrupt_f, corrupt_v = score(full, prompt_len, gold_ids); corrupt_store = cap["store"]; cap["store"] = None; corrupt["rows"] = None
            key_store, key_corrupt_f, key_corrupt_v = None, None, None
            if key_rows:
                cap["store"], cap["needed"] = {}, needed_t; corrupt["rows"], corrupt["page_pos"] = key_rows, pp
                _, key_corrupt_f, key_corrupt_v = score(full, prompt_len, gold_ids); key_store = cap["store"]; cap["store"] = None; corrupt["rows"] = None
            clean_lp, clean_f, clean_v = score(full, prompt_len, gold_ids)
            gaps_v = [c - k for c, k in zip(clean_v, corrupt_v)]; disc = max(range(len(gaps_v)), key=lambda t: gaps_v[t]); disc_gap = gaps_v[disc]
            key_disc, key_disc_gap = None, None
            if key_corrupt_v is not None:
                kg = [c - k for c, k in zip(clean_v, key_corrupt_v)]; key_disc = max(range(len(kg)), key=lambda t: kg[t]); key_disc_gap = kg[key_disc]
            seq_info = {"gold_ntok": len(gold_ids), "clean_tok": clean_v, "corrupt_tok": corrupt_v, "key_corrupt_tok": key_corrupt_v, "disc_pos": disc, "disc_gap": round(disc_gap, 4),
                        "key_disc_pos": key_disc, "key_disc_gap": (round(key_disc_gap, 4) if key_disc_gap is not None else None), "clean_mean": round(clean_lp, 4), "corrupt_mean": round(corrupt_lp, 4)}
            head = {"qid": it["qid"], "group": group, "gold": it["gold"], "case_name": it.get("case_name", ""), "n_pages": it["n_pages"], "n_layers": n_layers, "k_ev": len(ev_rows),
                    "clean_first": round(clean_f, 4), "corrupt_first": round(corrupt_f, 4), "key_corrupt_first": (round(key_corrupt_f, 4) if key_corrupt_f is not None else None), "backend": args.backend}
            if not (disc_gap > 0.5) and not (key_disc_gap is not None and key_disc_gap > 0.5):
                fout.write(json.dumps({**head, **seq_info, "skipped": "gap<=0.5"}, ensure_ascii=False) + "\n"); fout.flush()
                print(f"[{i+1}/{len(items)}] {it['qid']} gap={clean_f - corrupt_f:.2f} -- skipped", flush=True)
                del full, corrupt_store, key_store; torch.cuda.empty_cache(); continue
            inj = {s: [] for s in sites}; inj_disc = {s: [] for s in sites}
            for l in range(n_layers):
                for s, (rows, sel) in stensors.items():
                    store = key_store if s == "key_visual" else corrupt_store
                    patch["layer"], patch["rows"], patch["state"] = l, rows, store[l][:, sel, :]
                    _, f_, v_ = score(full, prompt_len, gold_ids)
                    inj[s].append(round(f_, 4)); inj_disc[s].append(v_[key_disc if (s == "key_visual" and key_disc is not None) else disc])
            patch["layer"] = None
            zsites = {s: stensors[s][0] for s in ("evidence_visual", "control_visual", "key_visual") if s in stensors}
            zer = {s: [] for s in zsites}; zer_disc = {s: [] for s in zsites}
            for l in range(n_layers):
                for s, rows in zsites.items():
                    zero["layer"], zero["rows"] = l, rows
                    _, f_, v_ = score(full, prompt_len, gold_ids)
                    zer[s].append(round(f_, 4)); zer_disc[s].append(v_[key_disc if (s == "key_visual" and key_disc is not None) else disc])
            zero["layer"] = None
            fout.write(json.dumps({**head, "inject_first": inj, "zero_first": zer, **seq_info, "inject_disc": inj_disc, "zero_disc": zer_disc}, ensure_ascii=False) + "\n"); fout.flush()
            print(f"[{i+1}/{len(items)}] {it['qid']} ({group}, p={it['n_pages']}) gap={clean_f - corrupt_f:.2f}" + (f" keygap={clean_f - key_corrupt_f:.2f}" if key_corrupt_f is not None else ""), flush=True)
            del full, corrupt_store, key_store; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
