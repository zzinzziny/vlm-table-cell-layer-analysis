#!/usr/bin/env python3
import json, math
from pathlib import Path
D = Path(__file__).parent
MODELS = [("qwen35_9b", "Qwen3.5-9B"), ("qwen3vl_8b", "Qwen3-VL-8B"), ("qwen25_7b", "Qwen2.5-VL-7B"), ("ministral3_8b", "Ministral-3-8B"),
          ("llava_ov2_8b", "LLaVA-OV2-8B"), ("gemma4_12b", "Gemma-4-12B"), ("qwen3vl_32b", "Qwen3-VL-32B"), ("gemma4_31b", "Gemma-4-31B")]
sel = json.load(open(D / "necessity_items.json"))
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)
out = {}; lines = ['# Necessity: ablating target cell B (score_necessity_B.py)', "", '| model | task | base-correct n | collapse with B ablated % | collapse with control ablated % | B only : control only | McNemar p | complete |', "|---|---|---|---|---|---|---|---|"]
for t, name in MODELS:
    f = D / f"necessity_{t}.jsonl"
    if not f.exists(): continue
    rows = [json.loads(l) for l in open(f)]
    for task in ("lookup_b", "compute"):
        rs = [r for r in rows if r["task"] == task and "skip" not in r and r["base_ok"]]
        if not rs: continue
        done = sum(1 for r in rows if r["task"] == task) >= len(sel[t][task])
        tc = sum(not r["target_ok"] for r in rs); cc = sum(not r["control_ok"] for r in rs)
        b = sum((not r["target_ok"]) and r["control_ok"] for r in rs); c = sum(r["target_ok"] and (not r["control_ok"]) for r in rs)
        res = dict(n=len(rs), target=100 * tc / len(rs), control=100 * cc / len(rs), disc=[b, c], p=mcnemar(b, c), done=done,
                   base_wrong=sum(1 for r in rows if r["task"] == task and "skip" not in r and not r["base_ok"]), skipped=sum(1 for r in rows if r["task"] == task and "skip" in r))
        out[f"{t}/{task}"] = res
        lines.append(f"| {name} | {task} | {len(rs)} | {res['target']:.0f} | {res['control']:.0f} | {b}:{c} | {res['p']:.1e} | {'done' if done else 'running'} |")
(D / "score_necessity_B.md").write_text("\n".join(lines) + "\n"); json.dump(out, open(D / "score_necessity_B.json", "w"), indent=1); print("\n".join(lines))
for k, v in out.items(): print(k, 'base answer wrong on rerun', v["base_wrong"], 'no control cell, skipped', v["skipped"])
