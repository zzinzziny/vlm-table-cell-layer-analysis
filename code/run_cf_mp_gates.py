#!/usr/bin/env python3
import argparse, json, re, unicodedata, torch
from pathlib import Path
from PIL import Image
from cf_mp_common import load, build, INSTR
BASE = Path(__file__).resolve().parent
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s)).strip().lower().replace("−", "-").replace(",", "")).rstrip(".")
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--backend", default="qwen"); ap.add_argument("--no-thinking-flag", action="store_true")
ap.add_argument("--cands", required=True); ap.add_argument("--out", required=True); ap.add_argument("--pass-prefix", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--resume", action="store_true")
args = ap.parse_args()
cands = [json.loads(l) for l in open(args.cands)]
if args.limit: cands = cands[:args.limit]
done = {}
if args.resume and Path(args.out).exists():
    for l in open(args.out): r = json.loads(l); done[r["qid"]] = r
proc, model = load(args.model, args.backend); tok = proc.tokenizer
def ask(images, q):
    inp, _ = build(proc, model, images, q, args.backend, not args.no_thinking_flag)
    with torch.no_grad(): g = model.generate(**inp, max_new_tokens=16, do_sample=False)
    return tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
results = dict(done)
with open(args.out, "a" if done else "w") as fout:
    for i, c in enumerate(cands):
        if c["qid"] in done: continue
        base = [Image.open(p).convert("RGB") for p in c["base_images"]]; donor = [Image.open(p).convert("RGB") for p in c["donor_images"]]
        preds = {"base_lookup_a": ask(base, c["lookup_a"]["question"]), "base_lookup_b": ask(base, c["lookup_b"]["question"]), "base_compute": ask(base, c["compute_question"]),
                 "donor_lookup_b": ask(donor, c["lookup_b"]["question"]), "donor_compute": ask(donor, c["compute_question"])}
        ok = {"base_lookup_a": norm(preds["base_lookup_a"]) == norm(c["lookup_a"]["gold"]), "base_lookup_b": norm(preds["base_lookup_b"]) == norm(c["b_value"]),
              "base_compute": norm(preds["base_compute"]) == norm(c["compute_gold"]), "donor_lookup_b": norm(preds["donor_lookup_b"]) == norm(c["b_new"]),
              "donor_compute": norm(preds["donor_compute"]) == norm(c["compute_gold_new"])}
        row = {"qid": c["qid"], "preds": preds, "ok": ok, "pass_lens": all(ok[k] for k in ("base_lookup_a", "base_lookup_b", "base_compute")), "pass_patch": all(ok.values())}
        results[c["qid"]] = row; fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
        print(f"[{i+1}/{len(cands)}] {c['qid']} patch={'P' if row['pass_patch'] else '.'} " + " ".join(f"{k}:{'O' if v else 'X'}" for k, v in ok.items()), flush=True)
with open(BASE / f"{args.pass_prefix}_pass_lens.txt", "w") as f: f.write("".join(q + "\n" for q, r in results.items() if r["pass_lens"]))
with open(BASE / f"{args.pass_prefix}_pass_patch.txt", "w") as f: f.write("".join(q + "\n" for q, r in results.items() if r["pass_patch"]))
print(f"pass_lens {sum(r['pass_lens'] for r in results.values())} / pass_patch {sum(r['pass_patch'] for r in results.values())} / {len(results)}")
