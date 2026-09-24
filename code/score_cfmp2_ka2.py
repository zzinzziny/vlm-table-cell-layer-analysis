#!/usr/bin/env python3
import statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from score_cf_pairs_unified_rev import (endpoints, classify, sign_test, wilson, cluster_boot, doc_level,
                                        signflip_perm, fmt_p, fmt_perm, fmt_ci, load_jsonl, doc_of,
                                        GEMMA_BLOCK_OF, MODELS)
from score_cfmp2 import clip, loss_layer, inj_curves, SETS

BASE = Path(__file__).resolve().parent
SET_NAME = 'MP total (190+631)'
POOL = dict(SETS)[SET_NAME]
NAME = dict(MODELS)
BLOCK_ROWS = [(t + "_block", NAME[t].split(" (")[0] + ' block') for t in GEMMA_BLOCK_OF]


def inj_gaps(D, S):
    lk = {q: v for (q, r), v in D.items() if r == "lookup_b" and q in S}
    cp = {q: v for (q, r), v in D.items() if r == "compute" and q in S}
    out = {}
    for q in sorted(set(lk) & set(cp)):
        a_, b_ = loss_layer(lk[q]), loss_layer(cp[q])
        out[q] = (a_ if a_ is not None else -1) - (b_ if b_ is not None else -1)
    return out, lk, cp


def block(gaps):
    if len(gaps) < 2:
        return None
    pos = sum(v > 0 for v in gaps.values()); neg = sum(v < 0 for v in gaps.values())
    zero = len(gaps) - pos - neg
    wl = wilson(pos, pos + neg)
    dl = doc_level(gaps); _, pp = signflip_perm(gaps)
    return dict(n=len(gaps), pos=pos, zero=zero, neg=neg,
                prop=(f"{100*pos/(pos+neg):.0f}% [{100*wl[0]:.0f}, {100*wl[1]:.0f}]" if pos + neg else "—"),
                mean=statistics.mean(gaps.values()), ci=cluster_boot(gaps, statistics.mean),
                doc_mean=dl["doc_mean"], doc_ci=dl["doc_mean_ci"],
                doc_sign=f"{dl['doc_pos']}:{dl['doc_zero']}:{dl['doc_neg']}", doc_p=dl["doc_sign_p"],
                sign_p=sign_test(pos, neg), perm=pp, n_doc=dl["n_doc"])


def row(name, metric, b):
    if b is None:
        return f"| {name} | {metric} | — | — | — | — | — | — | — | — |"
    return (f"| {name} | {metric} | {b['n']} | {b['pos']}:{b['zero']}:{b['neg']} | {b['prop']} | "
            f"{b['mean']:.2f} {fmt_ci(b['ci'])} | {b['doc_mean']:.2f} {fmt_ci(b['doc_ci'])} | "
            f"{b['doc_sign']} ({fmt_p(b['doc_p'])}) | {fmt_p(b['sign_p'])} | {fmt_perm(b['perm'])} |")


HEAD = ('| model | metric | valid pairs | Compute first:same:reversed | Compute-first rate [Wilson] | mean dL [cluster CI] | document mean [CI] | document signs +:0:- (p) | pair sign test p | permutation p |')
SEP = "|---|---|---|---|---|---|---|---|---|---|"


def main():
    out = []; E = lambda s="": out.append(s)
    E('# Multi-page pool 821: both endpoint metrics')
    E()
    E('Pool = cfmp 190 + cfmp2 631 = **821**. Rules and data as in paired_cfmp2_stats.md; here the injection curve gets the same statistics as the endpoint (document mean, document signs, permutation test).')
    E()
    E('- (A) **Full-generation endpoint** dL* = Lookup endpoint - Compute endpoint. Pairs where either endpoint is missing are dropped.')
    E('- (B) **Injection curve** dL(0.5) = Lookup vanishing layer - Compute vanishing layer, threshold 0.5. Only items with gap > 0.5 nat; if a curve never drops below the threshold its vanishing layer is set to -1 (before the input), as in score_paired_injection.py.')
    E('- All CIs are document-clustered 95% bootstraps; the permutation test flips the signs of all pairs of a document together.')
    E('- Positive = Compute ends earlier than Lookup.')
    E()

    INJ, ENDS = {}, {}
    for tag, _ in MODELS:
        files = [f"cfmp_patch_{tag}.jsonl", f"cfmp2_patch_{tag}.jsonl"]
        ENDS[tag] = endpoints(files)
        if tag in GEMMA_BLOCK_OF:
            ENDS[tag + "_block"] = endpoints(files, GEMMA_BLOCK_OF[tag])
        INJ[tag] = inj_curves(files)

    E('## 1. Both metrics, each on its own valid items')
    E()
    E('Note: the two metrics have different valid items (endpoint: pairs with both endpoints; injection: items above the gap threshold), so n differs. Section 2 compares them on the same items.')
    E()
    E(HEAD); E(SEP)
    SELF = {}
    for tag, name in MODELS:
        ga = classify({q: v for q, v in ENDS[tag].items() if q in POOL})["gap"]
        gb, _, _ = inj_gaps(INJ[tag][0], POOL)
        SELF[tag] = (ga, gb)
        E(row(name, '(A) endpoint', block(ga)))
        E(row(name, '(B) injection', block(gb)))
    for tag, name in BLOCK_ROWS:
        ga = classify({q: v for q, v in ENDS[tag].items() if q in POOL})["gap"]
        E(row(name, '(A) endpoint', block(ga)))
    E()

    E('## 2. Both metrics on their **common items**')
    E()
    E('Only items valid under both metrics are kept, so the same pairs are compared.')
    E()
    E(HEAD); E(SEP)
    for tag, name in MODELS:
        ga, gb = SELF[tag]
        common = sorted(set(ga) & set(gb))
        E(row(name, '(A) endpoint', block({q: ga[q] for q in common})))
        E(row(name, '(B) injection', block({q: gb[q] for q in common})))
    E()

    E('## 3. Models where the two metrics disagree')
    E()
    E('Own-item and common-item versions side by side.')
    E()
    E('| model | basis | (A) endpoint Compute first | (A) permutation p | (B) injection Compute first | (B) permutation p | same direction |')
    E("|---|---|---|---|---|---|---|")
    for tag, name in MODELS:
        ga, gb = SELF[tag]
        common = sorted(set(ga) & set(gb))
        for base, A, B in [('own items', ga, gb),
                           ('common items', {q: ga[q] for q in common}, {q: gb[q] for q in common})]:
            ba, bb = block(A), block(B)
            if not ba or not bb:
                E(f"| {name} | {base} | — | — | — | — | — |"); continue
            da = ba["pos"] >= ba["neg"]; db = bb["pos"] >= bb["neg"]
            E(f"| {name} | {base} | {ba['prop']} | {fmt_perm(ba['perm'])} | {bb['prop']} | {fmt_perm(bb['perm'])} | "
              f"{'yes' if da == db else '**no**'} |")
    E()
    E('')
    E()

    (BASE / "paired_cfmp2_ka2.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))

    tex = ['% generated by score_cfmp2_ka2.py. Pool = cfmp 190 + cfmp2 631 = 821.',
           '% common items of the two metrics.',
           r"\begin{tabular}{l r rrr r r}", r"\toprule",
           r"Model & $n$ & \multicolumn{3}{c}{Compute-first : tie : reversed} & $\Delta L$ [95\% CI] & perm.\ $p$ \\",
           r"\midrule"]
    for metric, idx in [("(A) full-generation endpoint", 0), ("(B) injection curve", 1)]:
        tex.append(r"\multicolumn{7}{l}{\textit{%s}} \\" % metric)
        for tag, name in MODELS:
            ga, gb = SELF[tag]
            common = sorted(set(ga) & set(gb))
            g = {q: (ga if idx == 0 else gb)[q] for q in common}
            b = block(g)
            if not b:
                tex.append(r"%s & -- & -- & -- & -- & -- & -- \\" % name.split(" (")[0]); continue
            lo, hi = b["ci"]
            tex.append(r"%s & %d & %d & %d & %d & %.2f [%.2f, %.2f] & %s \\"
                       % (name.split(" (")[0], b["n"], b["pos"], b["zero"], b["neg"], b["mean"], lo, hi,
                          fmt_perm(b["perm"]).replace("<", r"$<$")))
        tex.append(r"\midrule" if idx == 0 else r"\bottomrule")
    tex.append(r"\end{tabular}")
    (BASE / "tab_ka2_pool821.tex").write_text("\n".join(tex) + "\n")
    print('\n[written] paired_cfmp2_ka2.md, tab_ka2_pool821.tex')


if __name__ == "__main__":
    main()
