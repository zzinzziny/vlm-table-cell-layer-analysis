#!/usr/bin/env python3
"""Gate run for cf2 candidates: 5 greedy generations per candidate.

  base : lookup_a, lookup_b, compute        (clean-correct triplet)
  donor: lookup_b (=B'?), compute (=R'?)    (counterfactual-correct pair)

Writes cf2_gates.jsonl (one row per candidate with all preds + pass flags)
and two qid lists: cf2_pass_lens.txt (base 3 correct) and
cf2_pass_patch.txt (all 5 correct).

Usage: CUDA_VISIBLE_DEVICES=1 python run_cf_gates.py [--limit N]
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import torch
from PIL import Image

BASE = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3.5-9B"
CANDS = BASE / "cf2_candidates.jsonl"
OUT = BASE / "cf2_gates.jsonl"
INSTR = ("Reply with the answer only. No explanation, no reasoning, "
         "no bullet points.")


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    return re.sub(r"\s+", " ", s.replace("−", "-").replace(",", "")).rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cands", default=str(CANDS))
    ap.add_argument("--pass-prefix", default="cf2",
                    help="writes <prefix>_pass_lens.txt / _pass_patch.txt")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--resume", action="store_true",
                    help="skip qids already in --out, append new rows")
    args = ap.parse_args()

    cands = [json.loads(l) for l in open(args.cands)]
    if args.limit:
        cands = cands[:args.limit]

    done = {}
    if args.resume and Path(args.out).exists():
        done = {r["qid"]: r for r in
                (json.loads(l) for l in open(args.out))}
        print(f"resume: {len(done)} qids already gated, skipping them")

    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0").eval()

    def ask(img, q):
        content = [{"type": "image", "image": "img"},
                   {"type": "text", "text": f"{q}\n{INSTR}"}]
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=[text], images=[img], return_tensors="pt")
        dev = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**dev, max_new_tokens=16, do_sample=False)
        return processor.tokenizer.decode(
            gen[0][dev["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()

    results = dict(done)
    with open(args.out, "a" if done else "w") as fout:
        for i, c in enumerate(cands):
            if c["qid"] in done:
                continue
            base_img = Image.open(c["base_image"]).convert("RGB")
            donor_img = Image.open(c["donor_image"]).convert("RGB")
            preds = {
                "base_lookup_a": ask(base_img, c["lookup_a"]["question"]),
                "base_lookup_b": ask(base_img, c["lookup_b"]["question"]),
                "base_compute": ask(base_img, c["compute_question"]),
                "donor_lookup_b": ask(donor_img, c["lookup_b"]["question"]),
                "donor_compute": ask(donor_img, c["compute_question"]),
            }
            ok = {
                "base_lookup_a": norm(preds["base_lookup_a"])
                == norm(c["lookup_a"]["gold"]),
                "base_lookup_b": norm(preds["base_lookup_b"])
                == norm(c["b_value"]),
                "base_compute": norm(preds["base_compute"])
                == norm(c["compute_gold"]),
                "donor_lookup_b": norm(preds["donor_lookup_b"])
                == norm(c["b_new"]),
                "donor_compute": norm(preds["donor_compute"])
                == norm(c["compute_gold_new"]),
            }
            pass_lens = all(ok[k] for k in
                            ("base_lookup_a", "base_lookup_b", "base_compute"))
            pass_patch = all(ok.values())
            row = {"qid": c["qid"], "preds": preds, "ok": ok,
                   "pass_lens": pass_lens, "pass_patch": pass_patch}
            results[c["qid"]] = row
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i+1}/{len(cands)}] {c['qid']} "
                  f"lens={'P' if pass_lens else '.'} "
                  f"patch={'P' if pass_patch else '.'} "
                  + " ".join(f"{k}:{'O' if v else 'X'}"
                             for k, v in ok.items()), flush=True)

    ordered = [results[c["qid"]] for c in cands if c["qid"] in results]
    with open(BASE / f"{args.pass_prefix}_pass_lens.txt", "w") as flens:
        flens.writelines(r["qid"] + "\n" for r in ordered if r["pass_lens"])
    with open(BASE / f"{args.pass_prefix}_pass_patch.txt", "w") as fpatch:
        fpatch.writelines(r["qid"] + "\n" for r in ordered if r["pass_patch"])
    n_lens = sum(r["pass_lens"] for r in ordered)
    n_patch = sum(r["pass_patch"] for r in ordered)
    print(f"pass_lens {n_lens} / pass_patch {n_patch} / {len(cands)}")


if __name__ == "__main__":
    main()
