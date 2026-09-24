#!/usr/bin/env python3
import re, sys
from pathlib import Path
B = Path(__file__).resolve().parent
MPSET = sys.argv[1] if len(sys.argv) > 1 else "mp"
MODELS = [("qwen35_9b", "Qwen3.5-9B"), ("qwen3vl_8b", "Qwen3-VL-8B"), ("qwen25_7b", "Qwen2.5-VL-7B"), ("gemma4_12b", "Gemma-4-12B"),
          ("ministral3_8b", "Ministral-3-8B"), ("llava_ov2_8b", "LLaVA-OV2-8B"), ("qwen3vl_32b", "Qwen3-VL-32B"), ("gemma4_31b", "Gemma-4-31B")]
END = {}
# Compute endpoint = last layer of the cell-site recomputation window (d2d@b_cell), from score_recompute_common.py
cur = None
for l in open(B / "score_recompute_common.md"):
    m = re.match(r"## recompute_(sp|mp)_(\S+)\.jsonl", l)
    if m: cur = (m.group(1), m.group(2)); continue
    m = re.match(r"\| d2d@b_cell \| \d+ \| [^|]+ \| 0~(\d+) \|", l)
    if m and cur: END[cur] = int(m.group(1))
def curve(tag, setn):
    p = B / f"probes_{setn}_{tag}.md"
    if not p.exists(): return None
    sec = hdr = None; n = None
    for l in open(p):
        if l.startswith("# probes"):
            mm = re.search(r"n=(\d+), docs=(\d+)", l); n = (int(mm.group(1)), int(mm.group(2)))
        if l.startswith("## "): sec = l
        elif sec and "target B_first" in sec:
            if l.startswith("| prompt |"): hdr = [int(c.strip()[1:]) for c in l.strip().strip("|").split("|")[2:]]
            elif l.startswith("| lookup_b | b_cell |"): return n, hdr, [float(v) for v in l.strip().strip("|").split("|")[2:]]
    return None
out = [f"# Probe accuracy after the Compute endpoint (multi-page set = {MPSET})", "",
       '| model | single-page endpoint | single-page after endpoint | multi-page endpoint | multi-page after endpoint |', "|---|---|---|---|---|"]
rng = {"sp": [], MPSET: []}
for tag, name in MODELS:
    row = [name]
    for setn, gate in (("sp", "sp"), (MPSET, "mp")):
        c = curve(tag, setn); e = END.get((gate, tag))
        if not c or e is None: row += ["--", "--"]; continue
        _, hdr, acc = c
        post = [v for l, v in zip(hdr, acc) if l > e]
        if not post: row += [str(e), "--"]; continue
        row += [str(e), f"{min(post):.2f}--{max(post):.2f}"]
        rng[setn] += post
    out.append("| " + " | ".join(row) + " |")
out += ["", f"single-page range {min(rng['sp']):.2f}--{max(rng['sp']):.2f} - multi-page ({MPSET}) range {min(rng[MPSET]):.2f}--{max(rng[MPSET]):.2f}"]
(B / ("probe_after_endpoint.md" if MPSET == "mp" else f"probe_after_endpoint_{MPSET}.md")).write_text("\n".join(out) + "\n")
print("\n".join(out))
