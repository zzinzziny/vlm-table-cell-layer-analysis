#!/usr/bin/env python3
"""Layer-wise necessity of the evidence-cell representation, multi-image
capable (Qwen3.5-9B). Two scans per item, teacher-forced gold first token:

  inject : corrupt-injection (mirror of run_patch_inject.py) — capture the
           layer-l input states of the CORRUPT run (evidence cells
           mean-ablated, per-image mean) and overwrite ONE site at ONE layer
           in the CLEAN run. D = (clean-patched)/(clean-corrupt).
           Sites: evidence_visual / control_visual / last_text
           (+ key_visual with its own corrupt run, sweep items only).
  zero   : knockout (mirror of run_zero_layers.py) — clean run, h_site^(l)->0
           at one layer. Sites: evidence_visual / control_visual.

Items file formats:
  mpreq  (items_mpreq_qwen35.jsonl): group sweep (target/control/key_cell)
         or operand (operands/controls)
  compute(items_compute_qwen35.jsonl): operands/controls (SP when n_pages==1)

Usage:
  CUDA_VISIBLE_DEVICES=1 python run_layer_necessity.py --items X --out Y [--sp-only]
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

MODEL_ID = "Qwen/Qwen3.5-9B"  # default; override with --model
INSTR = ("Reply with the answer only. No explanation, no reasoning, "
         "no bullet points.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sp-only", action="store_true",
                    help="keep only n_pages==1 items")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mode", default="full", choices=["full", "ablate"],
                    help='ablate = replace the evidence rows at the input of layer l in place with their mean (in-place removal); inject/zero are not run. Same gate.')
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--no-thinking-flag", action="store_true",
                    help="model template has no enable_thinking (Qwen2.5-VL)")
    args = ap.parse_args()
    MODEL = args.model
    TKW = {} if args.no_thinking_flag else {"enable_thinking": False}

    items = [json.loads(l) for l in open(args.items)]
    if args.sp_only:
        items = [it for it in items if it["n_pages"] == 1]
    if args.limit:
        items = items[:args.limit]
    done = set()
    if args.resume and Path(args.out).exists():
        done = {json.loads(l)["qid"] for l in open(args.out)}
        items = [it for it in items if it["qid"] not in done]
    print(f"items: {len(items)} (done {len(done)})", flush=True)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(MODEL)
    tok = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0").eval()
    lm = getattr(model.model, "language_model", model.model)
    layers = lm.layers
    n_layers = len(layers)
    image_token_id = model.config.image_token_id

    ablate = {"spec": None}

    def vis_hook(module, hargs, output):
        spec = ablate["spec"]
        if spec is None:
            return output
        embeds = output.pooler_output.clone()
        assert embeds.shape[0] == spec["total"]
        rows = torch.tensor(spec["rows"])
        for start, end in spec["spans"]:
            span_rows = rows[(rows >= start) & (rows < end)]
            if len(span_rows):
                mean_vec = embeds[start:end].mean(dim=0, keepdim=True)
                embeds[span_rows] = mean_vec.to(embeds.dtype)
        output.pooler_output = embeds
        return output

    model.model.visual.register_forward_hook(vis_hook)

    cap = {"store": None, "needed": None}
    patch = {"layer": None, "rows": None, "state": None}
    zero = {"layer": None, "rows": None}
    abl = {"layer": None, "groups": None}

    def make_pre_hook(l):
        def pre_hook(module, hargs, hkwargs):
            hs = hargs[0] if hargs else hkwargs["hidden_states"]
            if cap["store"] is not None:
                cap["store"][l] = hs[:, cap["needed"], :].detach().clone()
            if patch["layer"] == l:
                hs[:, patch["rows"], :] = patch["state"]
            if zero["layer"] == l:
                hs[:, zero["rows"], :] = 0
            if abl["layer"] == l:
                for span_rows, site_rows in abl["groups"]:
                    hs[:, site_rows, :] = hs[:, span_rows, :].mean(dim=1, keepdim=True).to(hs.dtype)
            return None
        return pre_hook

    for l, layer in enumerate(layers):
        layer.register_forward_pre_hook(make_pre_hook(l), with_kwargs=True)

    def score(inputs, prompt_len, gold_ids):
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        lp = F.log_softmax(logits[prompt_len - 1:prompt_len - 1 + len(gold_ids)]
                           .float(), dim=-1)
        gold_t = torch.tensor(gold_ids, device=lp.device)
        tok_lp = lp.gather(1, gold_t[:, None]).squeeze(1)
        return tok_lp.mean().item(), tok_lp[0].item(), [round(x, 4) for x in tok_lp.tolist()]

    with open(args.out, "a" if args.resume else "w") as fout:
        for i, it in enumerate(items):
            paths = [p["image"] for p in it["pages"]]
            images = [Image.open(p).convert("RGB") for p in paths]
            content = ([{"type": "image", "image": p} for p in paths]
                       + [{"type": "text", "text": f"{it['question']}\n{INSTR}"}])
            text = processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True, **TKW)
            inputs = processor(text=[text], images=images, return_tensors="pt")
            grids = inputs["image_grid_thw"].tolist()
            offsets, spans, off, bad = {}, [], 0, False
            for page, (t, hp, wp) in zip(it["pages"], grids):
                gh, gw = hp // 2, wp // 2
                if [gh, gw] != page["grid_hw"]:
                    bad = True
                    break
                offsets[page["page_id"]] = off
                spans.append((off, off + gh * gw))
                off += gh * gw
            if bad:
                print(f"[{it['qid']}] grid mismatch -- skip", flush=True)
                continue
            total = off
            ids = inputs["input_ids"][0]
            vis_pos = (ids == image_token_id).nonzero().squeeze(1)
            assert vis_pos.shape[0] == total
            prompt_len = ids.shape[0]

            def rows_of(cells):
                return sorted({offsets[c["page_id"]] + k
                               for c in cells for k in c["vision_token_indices"]})

            group = it.get("group", "compute")
            if group == "sweep":
                ev_rows = rows_of([it["target"]])
                ctl_rows = rows_of([it["control"]])
                key_rows = rows_of(it["key_cells"]) if it.get("key_cells") else (rows_of([it["key_cell"]]) if it.get("key_cell") else None)
            else:
                ev_rows = rows_of(it["operands"])
                ctl_rows = rows_of(it["controls"]) if it.get("controls") else None
                key_rows = rows_of(it["key_cells"]) if it.get("key_cells") else None
            sites = {"evidence_visual": [int(vis_pos[r]) for r in ev_rows],
                     "last_text": [prompt_len - 1]}
            if ctl_rows:
                sites["control_visual"] = [int(vis_pos[r]) for r in ctl_rows]
            if key_rows:
                sites["key_visual"] = [int(vis_pos[r]) for r in key_rows]

            gold_ids = tok(it["gold"], add_special_tokens=False)["input_ids"]
            full = {k: v for k, v in inputs.items()}
            full["input_ids"] = torch.cat(
                [inputs["input_ids"], torch.tensor([gold_ids], dtype=torch.long)], dim=1)
            full["attention_mask"] = torch.ones_like(full["input_ids"])
            if "mm_token_type_ids" in inputs:
                mm = inputs["mm_token_type_ids"]
                full["mm_token_type_ids"] = torch.cat(
                    [mm, torch.zeros((1, len(gold_ids)), dtype=mm.dtype)], dim=1)
            full = {k: v.to(model.device) for k, v in full.items()}

            needed = sorted({p for ps in sites.values() for p in ps})
            row_of = {p: j for j, p in enumerate(needed)}
            needed_t = torch.tensor(needed, device=model.device)
            stensors = {s: (torch.tensor(ps, device=model.device),
                            [row_of[p] for p in ps]) for s, ps in sites.items()}

            # corrupt run (evidence ablated) with capture
            cap["store"], cap["needed"] = {}, needed_t
            ablate["spec"] = {"rows": ev_rows, "spans": spans, "total": total}
            corrupt_lp, corrupt_f, corrupt_v = score(full, prompt_len, gold_ids)
            corrupt_store = cap["store"]
            cap["store"] = None
            ablate["spec"] = None
            # corrupt run for key (sweep only)
            key_store, key_corrupt_f, key_corrupt_v = None, None, None
            if key_rows:
                cap["store"], cap["needed"] = {}, needed_t
                ablate["spec"] = {"rows": key_rows, "spans": spans, "total": total}
                _, key_corrupt_f, key_corrupt_v = score(full, prompt_len, gold_ids)
                key_store = cap["store"]
                cap["store"] = None
                ablate["spec"] = None
            clean_lp, clean_f, clean_v = score(full, prompt_len, gold_ids)
            # discriminative position: gold token where evidence ablation hurts most
            # (first token is uninformative when gold starts with '0.'/'1' etc.)
            gaps_v = [c - k for c, k in zip(clean_v, corrupt_v)]
            disc = max(range(len(gaps_v)), key=lambda t: gaps_v[t])
            disc_gap = gaps_v[disc]
            key_disc, key_disc_gap = None, None
            if key_corrupt_v is not None:
                kg = [c - k for c, k in zip(clean_v, key_corrupt_v)]
                key_disc = max(range(len(kg)), key=lambda t: kg[t]); key_disc_gap = kg[key_disc]
            seq_info = {"gold_ntok": len(gold_ids), "clean_tok": clean_v, "corrupt_tok": corrupt_v,
                        "key_corrupt_tok": key_corrupt_v, "disc_pos": disc, "disc_gap": round(disc_gap, 4),
                        "key_disc_pos": key_disc, "key_disc_gap": (round(key_disc_gap, 4) if key_disc_gap is not None else None),
                        "clean_mean": round(clean_lp, 4), "corrupt_mean": round(corrupt_lp, 4)}
            gap_ok = disc_gap > 0.5          # was: first-token gap only
            key_ok = key_disc_gap is not None and key_disc_gap > 0.5
            if not gap_ok and not key_ok:
                fout.write(json.dumps({
                    "qid": it["qid"], "group": group, "gold": it["gold"],
                    "case_name": it.get("case_name", ""), "n_pages": it["n_pages"],
                    "n_layers": n_layers, "k_ev": len(ev_rows),
                    "clean_first": round(clean_f, 4), "corrupt_first": round(corrupt_f, 4),
                    "key_corrupt_first": (round(key_corrupt_f, 4) if key_corrupt_f is not None else None),
                    **seq_info, "skipped": "gap<=0.5"}, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[{i+1}/{len(items)}] {it['qid']} gap={clean_f - corrupt_f:.2f} -- skipped", flush=True)
                del full, corrupt_store, key_store
                torch.cuda.empty_cache()
                continue

            if args.mode == "ablate":
                asites = {"evidence_visual": ev_rows}
                if ctl_rows: asites["control_visual"] = ctl_rows
                abl_f = {s: [] for s in asites}; abl_d = {s: [] for s in asites}
                for s_, srows in asites.items():
                    groups = []
                    for start, end in spans:
                        sr = [r for r in srows if start <= r < end]
                        if sr: groups.append((vis_pos[start:end].to(model.device), torch.tensor([int(vis_pos[r]) for r in sr], device=model.device)))
                    for l in range(n_layers):
                        abl["layer"], abl["groups"] = l, groups
                        _, f_, v_ = score(full, prompt_len, gold_ids)
                        abl_f[s_].append(round(f_, 4)); abl_d[s_].append(v_[disc])
                    abl["layer"] = None
                fout.write(json.dumps({
                    "qid": it["qid"], "group": group, "gold": it["gold"],
                    "case_name": it.get("case_name", ""), "n_pages": it["n_pages"],
                    "n_layers": n_layers, "k_ev": len(ev_rows),
                    "clean_first": round(clean_f, 4), "corrupt_first": round(corrupt_f, 4),
                    "key_corrupt_first": (round(key_corrupt_f, 4) if key_corrupt_f is not None else None),
                    "ablate_first": abl_f, **seq_info, "ablate_disc": abl_d, "mode": "ablate",
                }, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[{i+1}/{len(items)}] {it['qid']} ({group}, p={it['n_pages']}) gap={clean_f - corrupt_f:.2f} ablate k_ev={len(ev_rows)}", flush=True)
                del full, corrupt_store, key_store
                torch.cuda.empty_cache()
                continue
            inj = {s: [] for s in sites}
            inj_disc = {s: [] for s in sites}
            for l in range(n_layers):
                for s, (rows, sel) in stensors.items():
                    store = key_store if s == "key_visual" else corrupt_store
                    patch["layer"], patch["rows"] = l, rows
                    patch["state"] = store[l][:, sel, :]
                    _, f_, v_ = score(full, prompt_len, gold_ids)
                    inj[s].append(round(f_, 4))
                    inj_disc[s].append(v_[key_disc if (s == "key_visual" and key_disc is not None) else disc])
            patch["layer"] = None

            zsites = {"evidence_visual": stensors["evidence_visual"][0]}
            if "control_visual" in stensors:
                zsites["control_visual"] = stensors["control_visual"][0]
            if "key_visual" in stensors:
                zsites["key_visual"] = stensors["key_visual"][0]
            zer = {s: [] for s in zsites}
            zer_disc = {s: [] for s in zsites}
            for l in range(n_layers):
                for s, rows in zsites.items():
                    zero["layer"], zero["rows"] = l, rows
                    _, f_, v_ = score(full, prompt_len, gold_ids)
                    zer[s].append(round(f_, 4))
                    zer_disc[s].append(v_[key_disc if (s == "key_visual" and key_disc is not None) else disc])
            zero["layer"] = None

            fout.write(json.dumps({
                "qid": it["qid"], "group": group, "gold": it["gold"],
                "case_name": it.get("case_name", ""), "n_pages": it["n_pages"],
                "n_layers": n_layers, "k_ev": len(ev_rows),
                "clean_first": round(clean_f, 4), "corrupt_first": round(corrupt_f, 4),
                "key_corrupt_first": (round(key_corrupt_f, 4) if key_corrupt_f is not None else None),
                "inject_first": inj, "zero_first": zer,
                **seq_info, "inject_disc": inj_disc, "zero_disc": zer_disc,
            }, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i+1}/{len(items)}] {it['qid']} ({group}, p={it['n_pages']}) "
                  f"gap={clean_f - corrupt_f:.2f}"
                  + (f" keygap={clean_f - key_corrupt_f:.2f}" if key_corrupt_f is not None else ""),
                  flush=True)
            del full, corrupt_store, key_store
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
