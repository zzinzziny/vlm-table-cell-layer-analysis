#!/usr/bin/env python3
import json, collections, re, unicodedata
from pathlib import Path

B = Path(__file__).resolve().parent
MODELS = [("qwen35_9b", "Qwen3.5-9B"), ("qwen3vl_8b", "Qwen3-VL-8B"), ("qwen25_7b", "Qwen2.5-VL-7B"),
          ("gemma4_12b", "Gemma-4-12B"), ("ministral3_8b", "Ministral-3-8B"), ("llava_ov2_8b", "LLaVA-OV2-8B"),
          ("qwen3vl_32b", "Qwen3-VL-32B"), ("gemma4_31b", "Gemma-4-31B")]
PATCH = {t: [f"cfmp_patch_{t}.jsonl", f"cfmp2_patch_{t}.jsonl"] for t, _ in MODELS}


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    return re.sub(r"\s+", " ", s.replace("−", "-").replace(",", "")).rstrip(".")


lines = ['# Value specificity, multi-page pool', "",
         'Pool = cfmp 190 + cfmp2 631 = **821**. For gate-passed items, share of full-generation answers that follow the donor answer (Lookup: B-prime, Compute: f(A, B-prime)) when the layer-0 B-cell state is replaced by the donor state. Same computation as score_valuespec_sp.py with the multi-page patch files.', "",
         '| model | n | LOOKUP follow | COMPUTE follow | sanity: base answer kept when patching the last layer (LOOKUP) |',
         "|---|---|---|---|---|"]
summary = {}
for t, name in MODELS:
    by = collections.defaultdict(dict)
    for f in PATCH[t]:
        if (B / f).exists():
            for l in open(B / f):
                r = json.loads(l); by[(r["qid"], r["role"])][r["layer"]] = r
    if not by:
        lines.append(f"| {name} | - | file missing | | |"); continue
    fol = {}
    for role in ("lookup_b", "compute"):
        v = [norm(d[0]["pred"]) == norm(d[0]["ans_donor"]) for (q, ro), d in by.items() if ro == role and 0 in d]
        fol[role] = (len(v), round(100 * sum(v) / len(v)) if v else None)
    last = [norm(d[max(d)]["pred"]) == norm(d[max(d)]["ans_base"]) for (q, ro), d in by.items() if ro == "lookup_b" and d]
    summary[t] = dict(n=fol["lookup_b"][0], lookup=fol["lookup_b"][1], compute=fol["compute"][1])
    lines.append(f"| {name} | {fol['lookup_b'][0]} | {fol['lookup_b'][1]}% | {fol['compute'][1]}% | "
                 f"{100*sum(last)/len(last):.0f}% (layer {max(max(d) for d in by.values())}) |")

lo = [v["lookup"] for v in summary.values() if v["lookup"] is not None]
co = [v["compute"] for v in summary.values() if v["compute"] is not None]
lines += ["", f"**Summary**: across the eight models, LOOKUP {min(lo)}--{max(lo)}%, COMPUTE {min(co)}--{max(co)}% of answers follow the donor answer."]

(B / "score_valuespec_mp.md").write_text("\n".join(lines) + "\n")
json.dump(summary, open(B / "score_valuespec_mp.json", "w"), ensure_ascii=False, indent=1)
print("\n".join(lines))
