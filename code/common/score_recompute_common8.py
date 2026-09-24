#!/usr/bin/env python3
import json, glob, collections
MODELS = [("qwen35_9b", "Qwen3.5-9B"), ("qwen3vl_8b", "Qwen3-VL-8B"), ("qwen25_7b", "Qwen2.5-VL-7B"), ("gemma4_12b", "Gemma-4-12B"),
          ("ministral3_8b", "Ministral-3-8B"), ("llava_ov2_8b", "LLaVA-OV2-8B"), ("qwen3vl_32b", "Qwen3-VL-32B"), ("gemma4_31b", "Gemma-4-31B")]
SRC = [("d2d", 'same operation'), ("d3d", 'other operation'), ("lkd", 'Lookup difference')]
SITES = [("b_cell", 'cell'), ("text_all", 'text')]
out = {}
lines = ['# Recomputation control on the items shared by the three difference sources', "",
         'Gate = all 8 runs correct (gate_all8). Share (%) of items on which each of the three sources (same operation d2d, other operation d3d, Lookup difference lkd) yields the recomputed answer r in at least one layer, on **the same item set**.',
         'score_recompute_common.md uses the 6-run gate (core6), so its n is larger.', "",
         '| model | common n | cell: same op / other op / Lookup | text: same op / other op / Lookup |', "|---|---|---|---|"]
for tag, name in MODELS:
    rows = [json.loads(l) for l in open(f"recompute_sp_{tag}.jsonl")]
    ok = [r for r in rows if not r.get("skip") and r.get("gate_all8")]
    keys = [f"{c}@{s}" for c, _ in SRC for s, _ in SITES]
    ok = [r for r in ok if all(k in r["patch"] and r["patch"][k] != "misaligned" for k in keys)]
    rec = {"n": len(ok)}
    for c, _ in SRC:
        for s, _ in SITES:
            seqs = [r["patch"][f"{c}@{s}"] for r in ok]
            rec[f"{c}_{s}"] = round(100 * sum(1 for x in seqs if "r" in x) / len(seqs)) if seqs else None
    out[tag] = rec
    f = lambda s: " / ".join(str(rec[f"{c}_{s}"]) for c, _ in SRC)
    lines.append(f"| {name} | {rec['n']} | {f('b_cell')} | {f('text_all')} |")
open("recompute_common8.md", "w").write("\n".join(lines) + "\n")
json.dump(out, open("recompute_common8.json", "w"), indent=1)
print("\n".join(lines))
