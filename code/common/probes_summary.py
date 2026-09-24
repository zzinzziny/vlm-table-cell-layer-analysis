#!/usr/bin/env python3
import re, glob, sys
from pathlib import Path
TAGS = [("qwen35_9b","Qwen3.5-9B"),("qwen3vl_8b","Qwen3-VL-8B"),("qwen25_7b","Qwen2.5-VL-7B"),("gemma4_12b","Gemma 4 12B"),("ministral3_8b","Ministral 3 8B"),("llava_ov2_8b","LLaVA-OV2 8B"),("qwen3vl_32b","Qwen3-VL-32B"),("gemma4_31b","Gemma 4 31B")]
def parse(path):
    txt = Path(path).read_text().splitlines()
    head = txt[0]; m = re.search(r"n=(\d+), docs=(\d+), layers=(\d+)/(\d+)", head)
    info = {"n": int(m.group(1)), "docs": int(m.group(2)), "nl": int(m.group(3)), "L1": int(m.group(4))}
    cur = None; layers = None; rows = {}
    for l in txt:
        if l.startswith("## target"):
            mm = re.match(r"## target (\S+)(?: \(n=(\d+), majority ([\d.]+)\))?", l); cur = mm.group(1)
            rows[cur] = {"n": int(mm.group(2)) if mm.group(2) else None, "maj": float(mm.group(3)) if mm.group(3) else None}
        elif l.startswith("| prompt | site |"):
            layers = [int(x[1:]) for x in l.strip("| ").split(" | ")[2:]]
        elif l.startswith("| ") and cur and not l.startswith("|---"):
            p = [x.strip() for x in l.strip("| ").split(" | ")]
            if len(p) < 3: continue
            pr, st = p[0], p[1]; vals = [float(x) for x in p[2:]]
            rows[cur][(pr, st)] = (vals, layers)
    return info, rows
def best(rows, t, pr, st):
    r = rows.get(t, {}).get((pr, st))
    if not r: return None
    vals, layers = r; i = max(range(len(vals)), key=lambda k: vals[k]); return vals[i], layers[i]
out = ['# Probe summary (probes_summary.py; value = best-layer accuracy of out-of-document 5-fold probes (layer), shuffle = best value of the shuffled-label control; hidden_states[l] = input of layer l)', ""]
for setn in ("sp", "mp"):
    out.append(f"## {setn.upper()}"); out.append('| model | n (docs) | layer grid | B first digit majority | b_cell B first digit (lookup) | b_cell shuffled | b_cell B full value (lookup) | tail B first digit (compute) | last B first digit (compute) | tail B first digit (lookup) | last B first digit (lookup) | tail A first digit (compute) | last R first digit (compute) |')
    out.append("|---|" + "---|" * 16)
    for tag, name in TAGS:
        f = f"probes_{setn}_{tag}.md"
        if not Path(f).exists(): out.append(f"| {name} | | | | | | | | | | | | | | | | |"); continue
        info, rows = parse(f)
        def fmt(t, pr, st):
            b = best(rows, t, pr, st); return f"{b[0]:.2f} (L{b[1]})" if b else ""
        def fmtsh(t, pr, st):
            b = best(rows, t, pr, st + " (shuffle)"); return f"{b[0]:.2f}" if b else ""
        maj = rows.get("B_first", {}).get("maj"); rmaj = rows.get("R_first", {}).get("maj")
        out.append(f"| {name} | {info['n']} ({info['docs']}) | {info['nl']}layers/{info['L1']} | {maj:.2f} | {fmt('B_first','lookup_b','b_cell')} | {fmtsh('B_first','lookup_b','b_cell')} | {fmt('B_full','lookup_b','b_cell')} | {fmt('B_first','compute','tail')} | {fmt('B_first','compute','last')} | {fmt('B_first','lookup_b','tail')} | {fmt('B_first','lookup_b','last')} | {fmt('A_first','compute','tail')} | {fmt('R_first','compute','last')} | {fmt('R_first','compute','tail')} | {fmt('R_full','compute','last')} | {rmaj:.2f} | {fmt('R_first','compute','b_cell')} |")
    out.append("")
out.append('## Accuracy of the first digit of B across layers (compute prompt; grid layer closest to relative depth 0.2/0.4/0.6/0.8/1.0)')
out.append('| model, set | site | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |'); out.append("|---|---|---|---|---|---|---|")
for setn in ("sp", "mp"):
    for tag, name in TAGS:
        f = f"probes_{setn}_{tag}.md"
        if not Path(f).exists(): continue
        info, rows = parse(f); L1 = info["L1"] - 1
        for st in ("b_cell", "tail", "last"):
            r = rows.get("B_first", {}).get(("compute", st))
            if not r: continue
            vals, layers = r; cells = []
            for rel in (0.2, 0.4, 0.6, 0.8, 1.0):
                tgt = rel * L1; i = min(range(len(layers)), key=lambda k: abs(layers[k] - tgt)); cells.append(f"{vals[i]:.2f} (L{layers[i]})")
            out.append(f"| {name} {setn.upper()} | {st} | " + " | ".join(cells) + " |")
Path("probes_summary.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
