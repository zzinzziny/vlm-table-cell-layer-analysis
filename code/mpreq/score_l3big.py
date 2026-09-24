#!/usr/bin/env python3
"""Score the large-L3 layer-by-layer run (items_l3big_qwen35.jsonl).

  layer necessity : layer_necessity_l3big_<tag>.jsonl  (inject D(l), knockout drop(l)) per subgroup
  ablation        : preds_ablation_l3big_<tag>.jsonl   (baseline / target / control / key / page collapse)

Metric (2nd arg, default auto):
  first : first gold token log-prob (original). Usable: clean_first - corrupt_first > 0.5 nat.
  disc  : discriminative gold token = position where evidence ablation hurts most (disc_pos, written by
          run_layer_necessity.py since 2026-09-12). Usable: disc_gap > 0.5. key site uses key_disc_pos.
  auto  : disc if the file has disc fields, else first.
Loss layer = last layer with D > 0.5 (inject) / drop > 0.5*max (knockout).
Tag '<x>_v2' reads layer_necessity_l3big_<x>_v2.jsonl but preds_ablation_l3big_<x>.jsonl (9B disc rerun).
Ablation collapse is reported on exact-match baseline-correct items (ANLS>=0.5 figure in parentheses).
Usage: python score_l3big.py [tag=qwen35_9b] [metric=auto]
   -> score_l3big_<tag>[_first].txt + layer_necessity_l3big_<tag>[_first].png
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from score_ablation import anls, norm  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "qwen35_9b"
METRIC = sys.argv[2] if len(sys.argv) > 2 else "auto"
ITEMS = BASE / "items_l3big_qwen35.jsonl"
LN = BASE / f"layer_necessity_l3big_{TAG}.jsonl"
AB = BASE / f"preds_ablation_l3big_{TAG[:-3] if TAG.endswith('_v2') else TAG}.jsonl"
GAP = 0.5
ORDER = ["lookup", "join", "chain", "split_pair_ops", "cross_table_compare", "split_top2"]
REL = (0.4, 0.6, 0.72, 0.85, 1.0)   # depth fractions for the D(l) columns (32L -> 12/19/22/26/31)


def loss_layer(curve, thr):
    over = [l for l, v in enumerate(curve) if v > thr]
    return over[-1] if over else None


def ok(p, g):
    return anls(p, g) >= 0.5


def exact(p, g):
    return norm(p) == norm(g)


def make_accessors(metric):
    """Return (usable, D, drop, key_usable, keyD) closures for the chosen metric."""
    if metric == "first":
        def usable(r): return r["clean_first"] - r["corrupt_first"] > GAP and "inject_first" in r
        def D(r, s, l): return (r["clean_first"] - r["inject_first"][s][l]) / (r["clean_first"] - r["corrupt_first"])
        def drop(r, s, l): return r["clean_first"] - r["zero_first"][s][l]
        def key_usable(r): return ("key_visual" in r.get("inject_first", {}) and r.get("key_corrupt_first") is not None
                                   and r["clean_first"] - r["key_corrupt_first"] > GAP)
        def keyD(r, l): return (r["clean_first"] - r["inject_first"]["key_visual"][l]) / (r["clean_first"] - r["key_corrupt_first"])
    else:
        def usable(r): return r.get("disc_gap") is not None and r["disc_gap"] > GAP and "inject_disc" in r
        def D(r, s, l):
            p = r["disc_pos"]; c = r["clean_tok"][p]
            return (c - r["inject_disc"][s][l]) / (c - r["corrupt_tok"][p])
        def drop(r, s, l): return r["clean_tok"][r["disc_pos"]] - r["zero_disc"][s][l]
        def key_usable(r): return ("key_visual" in r.get("inject_disc", {}) and r.get("key_disc_gap") is not None
                                   and r["key_disc_gap"] > GAP)
        def keyD(r, l):
            p = r["key_disc_pos"]; c = r["clean_tok"][p]
            return (c - r["inject_disc"]["key_visual"][l]) / (c - r["key_corrupt_tok"][p])
    return usable, D, drop, key_usable, keyD


def main():
    items = {json.loads(l)["qid"]: json.loads(l) for l in open(ITEMS)}
    lines = []

    def emit(s=""):
        lines.append(s); print(s)

    metric = METRIC
    rows_ln = [json.loads(l) for l in open(LN)] if LN.exists() else []
    if metric == "auto":
        metric = "disc" if rows_ln and any("disc_pos" in r for r in rows_ln) else "first"
    suffix = "" if metric == "disc" or not rows_ln or not any("disc_pos" in r for r in rows_ln) else "_first"
    emit(f"# L3 big pool, model {TAG}, metric {metric}, items {len(items)} "
         f"(sweep {sum(1 for i in items.values() if i['group']=='sweep')} / operand {sum(1 for i in items.values() if i['group']=='operand')})")

    # ---------------- layer necessity
    if rows_ln:
        rows = rows_ln
        usable, D, drop, key_usable, keyD = make_accessors(metric)
        L = max((r["n_layers"] for r in rows), default=32)
        cols = sorted({min(L - 1, round(f * (L - 1))) for f in REL})
        n_us = sum(1 for r in rows if usable(r))
        n_first = sum(1 for r in rows if r["clean_first"] - r["corrupt_first"] > GAP)
        emit(f"\n## layer necessity: {len(rows)} rows scored, n_layers {L}, usable ({metric} gap>{GAP}) = {n_us}"
             f" (first-token rule would give {n_first})")
        if metric == "disc":
            pos = [r["disc_pos"] for r in rows if usable(r)]
            emit(f"   disc_pos among usable: pos0 {sum(1 for p in pos if p==0)}, pos1 {sum(1 for p in pos if p==1)}, pos>=2 {sum(1 for p in pos if p>=2)}")
        by = defaultdict(list)
        for r in rows:
            by[items[r["qid"]]["subgroup"] if r["qid"] in items else "?"].append(r)
        emit(f"{'subgroup':<22}{'n':>5}{'usable':>7}  inject loss layer (curve | per-item med [IQR])  "
             + " ".join(f"D@L{l}" for l in cols) + "  ctl max | knockout half-max layer  ctl max")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for sg in ORDER + [k for k in by if k not in ORDER]:
            R = by.get(sg)
            if not R:
                continue
            U = [r for r in R if usable(r)]
            if not U:
                emit(f"{sg:<22}{len(R):>5}{0:>7}"); continue
            curve = [statistics.mean(max(-0.25, min(1.25, D(r, "evidence_visual", l))) for r in U) for l in range(L)]
            has_ctl = [r for r in U if "control_visual" in (r["inject_disc"] if metric == "disc" else r["inject_first"])]
            ctl = [statistics.mean(max(-0.25, min(1.25, D(r, "control_visual", l))) for r in has_ctl) for l in range(L)] if has_ctl else [0] * L
            per = [loss_layer([D(r, "evidence_visual", l) for l in range(L)], 0.5) for r in U]
            per = [p for p in per if p is not None]
            q = statistics.quantiles(per, n=4) if len(per) >= 4 else [None] * 3
            zc = [statistics.mean(drop(r, "evidence_visual", l) for r in U) for l in range(L)]
            zhas = [r for r in U if "control_visual" in (r["zero_disc"] if metric == "disc" else r["zero_first"])]
            zctl = [statistics.mean(drop(r, "control_visual", l) for r in zhas) for l in range(L)] if zhas else [0] * L
            emit(f"{sg:<22}{len(R):>5}{len(U):>7}  {str(loss_layer(curve, 0.5)):>5} | per-item med {statistics.median(per) if per else None} "
                 f"[{q[0]}-{q[2]}] (n={len(per)})  " + " ".join(f"{curve[l]:5.2f}" for l in cols)
                 + f"  {max(ctl):.2f} | {loss_layer(zc, 0.5*max(zc)) if max(zc) > 0 else None}  {max(abs(x) for x in zctl):.2f}")
            axes[0].plot(range(L), curve, label=f"{sg} (n={len(U)})")
            axes[1].plot(range(L), zc, label=f"{sg} (n={len(U)})")
            K = [r for r in R if key_usable(r)]
            if K:
                kc = [statistics.mean(keyD(r, l) for r in K) for l in range(L)]
                emit(f"   key cell ({sg}) n={len(K)}: loss layer {loss_layer(kc, 0.5)}; " + " ".join(f"L{l}={kc[l]:.2f}" for l in cols))
        for ax, t in zip(axes, (f"inject D(l): evidence site [{metric}]", f"knockout drop (nat): evidence site [{metric}]")):
            ax.set_title(t); ax.set_xlabel("layer"); ax.axhline(0.5 if "inject" in t else 0, color="gray", lw=0.5); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(BASE / f"layer_necessity_l3big_{TAG}{suffix}.png", dpi=130)
        emit(f"-> layer_necessity_l3big_{TAG}{suffix}.png")
    else:
        emit("\n(no layer necessity file yet)")

    # ---------------- ablation
    if AB.exists():
        rows = [json.loads(l) for l in open(AB)]
        emit(f"\n## ablation: {len(rows)} rows ({AB.name}); collapse = wrong after intervention among baseline-correct, exact match (ANLS>=0.5 in parens)")
        by = defaultdict(list)
        for r in rows:
            by[items[r["qid"]]["subgroup"] if r["qid"] in items else "?"].append(r)
        for sg in ORDER + [k for k in by if k not in ORDER]:
            R = by.get(sg)
            if not R:
                continue
            base_e = [r for r in R if exact(r["preds"]["none"], r["gold"])]
            base_a = [r for r in R if ok(r["preds"]["none"], r["gold"])]
            emit(f"### {sg}: n={len(R)}, baseline correct exact {len(base_e)} ({100*len(base_e)/len(R):.0f}%), ANLS {len(base_a)} ({100*len(base_a)/len(R):.0f}%)")

            def collapse(base, judge):
                conds = defaultdict(list)
                for r in base:
                    for c, p in r["preds"].items():
                        if c == "none" or p is None:
                            continue
                        key = "page_other(each)" if c.startswith("page_other_") else ("ops_page(each)" if c.startswith("ops_page_") else ("evpage(each)" if c.startswith("evpage_") else c))
                        conds[key].append(not judge(p, r["gold"]))
                return conds
            ce, ca = collapse(base_e, exact), collapse(base_a, ok)
            emit("   collapse: " + ", ".join(f"{c} {sum(v)}/{len(v)} ({sum(ca[c])}/{len(ca[c])})" for c, v in sorted(ce.items())))
            if "target" in ce and "control" in ce:
                tc = sum(1 for r in base_e if r["preds"].get("target") is not None and r["preds"].get("control") is not None
                         and not exact(r["preds"]["target"], r["gold"]) and exact(r["preds"]["control"], r["gold"]))
                emit(f"   target-only collapse (target wrong & control right, exact): {tc}")
    else:
        emit("\n(no ablation file yet)")
    (BASE / f"score_l3big_{TAG}{suffix}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
