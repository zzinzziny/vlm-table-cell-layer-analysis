#!/usr/bin/env python3
import json, statistics, random
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, mannwhitneyu
from score_cf_pairs_unified_rev import (endpoints, classify, sign_test, wilson, cluster_boot, doc_level, signflip_perm,
                                        fmt_p, fmt_perm, fmt_ci, load_jsonl, doc_of, GATE_KEYS, GEMMA_BLOCK, GEMMA_BLOCK_OF,
                                        MODELS)

BASE = Path(__file__).resolve().parent
NAME = dict(MODELS)
BLOCK_ROWS = [(t + "_block", NAME[t].split(" (")[0] + ' block') for t in GEMMA_BLOCK_OF]
GAP, THR = 0.5, 0.5

pool1 = {r["qid"]: r for r in load_jsonl("cfmp_pool_qwen35.jsonl")}
pool2 = {r["qid"]: r for r in load_jsonl("cfmp2_pool_qwen35.jsonl")}
poolrow = {**pool1, **pool2}
rit1 = {q for q, r in pool1.items() if str(r.get("result_in_table")) == "True"}
SETS = [("cfmp 190", set(pool1)), ("cfmp noRIT", set(pool1) - rit1), ("cfmp2 631", set(pool2)), ('MP total (190+631)', set(pool1) | set(pool2))]


def clip(x): return max(-0.25, min(1.25, x))
def loss_layer(curve, thr=THR):
    o = [l for l, v in enumerate(curve) if v > thr]; return o[-1] if o else None
def inj_curves(files):
    by = defaultdict(dict)
    for f in files:
        for r in load_jsonl(f):
            by[(r["qid"], r["role"])][r["layer"]] = r["lp_donor_tok"]
    L = max(l for d in by.values() for l in d) + 1
    D = {}
    for (q, role), d in by.items():
        if -1 not in d or -2 not in d: continue
        gap = d[-2] - d[-1]
        if gap <= GAP: continue
        D[(q, role)] = [clip((d[l] - d[-1]) / gap) if l in d else np.nan for l in range(L)]
    return D, L


def main():
    out = []; E = lambda s="": out.append(s)
    E('# Multi-page paired extension pool cfmp2')
    E()
    E('cfmp2 631 = integers with 2-4 digits, at most 12 per table and 16 per document, candidates whose result is printed in the table excluded, disjoint from cfmp 190 in (document, table, column, row A, row B). cfmp noRIT = the 102 cfmp items left after removing result-in-table candidates (same condition as cfmp2).')
    E()
    G, P, INJ = {}, {}, {}
    for tag, _ in MODELS:
        g = {r["qid"]: r for r in load_jsonl(f"cfmp_gates_{tag}.jsonl")}
        g.update({r["qid"]: r for r in load_jsonl(f"cfmp2_gates_{tag}.jsonl")})
        files = [f"cfmp_patch_{tag}.jsonl", f"cfmp2_patch_{tag}.jsonl"]
        G[tag] = g; P[tag] = endpoints(files)
        if tag in GEMMA_BLOCK_OF:
            P[tag + "_block"] = endpoints(files, GEMMA_BLOCK_OF[tag])
        INJ[tag] = inj_curves(files)
    sub = lambda d, S: {q: v for q, v in d.items() if q in S}

    # ------------------------------------------------ 1. funnel
    E('## 1. Funnel')
    E()
    E('| model | set | candidates | gate passed | patched | both valid | non-tied | Compute first | same | reversed | one side only | both invalid | documents with valid pairs |')
    E("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, name in MODELS:
        for sn, S in SETS:
            g = sub(G[tag], S); e = sub(P[tag], S); c = classify(e)
            E(f"| {name} | {sn} | {len(g)} | {sum(1 for r in g.values() if r['pass_patch'])} | {len(e)} | {len(c['both'])} | {len(c['earlier'])+len(c['later'])} | "
              f"{len(c['earlier'])} | {len(c['same'])} | {len(c['later'])} | {len(c['one'])} | {len(c['none'])} | {len({doc_of(q) for q in c['both']})} |")
    E()
    E('### 1b. Accuracy of each of the 5 gate questions')
    E()
    E('| model | set | n | base lookup-A | base lookup-B | base compute | donor lookup-B | donor compute | passed |')
    E("|---|---|---|---|---|---|---|---|---|")
    for tag, name in MODELS:
        for sn, S in SETS[:3]:
            g = list(sub(G[tag], S).values()); n = len(g)
            E(f"| {name} | {sn} | {n} | " + " | ".join(f"{100*sum(1 for r in g if r['ok'][k])/n:.0f}%" for k in GATE_KEYS)
              + f" | {100*sum(1 for r in g if r['pass_patch'])/n:.0f}% |")
    E()

    # ------------------------------------------------ 2. full-generation endpoint stats
    E('## 2. (A) Full-generation endpoint dL*: pair sign test and document-clustered CI')
    E()
    E('| model | set | valid pairs | Compute first:same:reversed | Compute-first rate [Wilson] | pair sign test p | median dL* [cluster CI] | mean dL* [cluster CI] | documents (pairs/doc median, max) | document mean [CI] | document signs +:0:- (p) | permutation p |')
    E("|---|---|---|---|---|---|---|---|---|---|---|---|")
    GAPS = {}
    for tag, name in MODELS + BLOCK_ROWS:
        for sn, S in SETS:
            c = classify(sub(P[tag], S)); gaps = c["gap"]; GAPS[(tag, sn)] = gaps
            if len(gaps) < 2:
                E(f"| {name} | {sn} | {len(gaps)} | — | — | — | — | — | — | — | — | — |"); continue
            a, b = len(c["earlier"]), len(c["later"]); wl = wilson(a, a + b)
            dl = doc_level(gaps); _, pp = signflip_perm(gaps)
            prop = f"{100*a/(a+b):.0f}% [{100*wl[0]:.0f}, {100*wl[1]:.0f}]" if a + b else "—"
            E(f"| {name} | {sn} | {len(gaps)} | {a}:{len(c['same'])}:{b} | {prop} | {fmt_p(sign_test(a, b))} | "
              f"{statistics.median(gaps.values()):g} {fmt_ci(cluster_boot(gaps, statistics.median), 1)} | {statistics.mean(gaps.values()):.2f} {fmt_ci(cluster_boot(gaps, statistics.mean))} | "
              f"{dl['n_doc']} ({dl['pairs_per_doc_med']:g}·{dl['pairs_per_doc_max']}) | {dl['doc_mean']:.2f} {fmt_ci(dl['doc_mean_ci'])} | {dl['doc_pos']}:{dl['doc_zero']}:{dl['doc_neg']} ({fmt_p(dl['doc_sign_p'])}) | {fmt_perm(pp)} |")
    E()
    E('### 2b. Endpoint distribution (valid pairs, MP total)')
    E()
    E('| model | modal Lookup endpoints (layer:pairs, top 3) | modal Compute endpoints |')
    E("|---|---|---|")
    for tag, name in MODELS:
        e = sub(P[tag], SETS[3][1]); both = classify(e)["both"]
        top = lambda cnt: " ".join(f"{l}:{n}" for l, n in cnt.most_common(3))
        E(f"| {name} | {top(Counter(e[q][0] for q in both))} | {top(Counter(e[q][1] for q in both))} |")
    E()
    E('### 2c. cfmp2 vs cfmp noRIT (difference in dL* distribution, two-sided Mann-Whitney; items do not overlap, so no paired test)')
    E()
    E('| model | cfmp noRIT n / mean | cfmp2 n / mean | difference (cfmp2 - noRIT) | MWU p |')
    E("|---|---|---|---|---|")
    for tag, name in MODELS + BLOCK_ROWS:
        a = list(GAPS[(tag, "cfmp noRIT")].values()); b = list(GAPS[(tag, "cfmp2 631")].values())
        if len(a) < 2 or len(b) < 2:
            E(f"| {name} | {len(a)} | {len(b)} | — | — |"); continue
        E(f"| {name} | {len(a)} / {statistics.mean(a):.2f} | {len(b)} / {statistics.mean(b):.2f} | {statistics.mean(b)-statistics.mean(a):+.2f} | {fmt_p(mannwhitneyu(a, b, alternative='two-sided').pvalue)} |")
    E()

    # ------------------------------------------------ 3. injection curve
    E('## 3. (B) Injection curve D_p(l), same metric as layer necessity')
    E()
    E('| model | set | usable Lookup / Compute / both | curve vanishing layer Lookup (rel.) | curve vanishing layer Compute (rel.) | dL(0.5) Compute first:same:reversed | mean dL [doc CI] | sign test p |')
    E("|---|---|---|---|---|---|---|---|")
    for tag, name in MODELS:
        D, L = INJ[tag]
        for sn, S in SETS:
            lk = {q: v for (q, r), v in D.items() if r == "lookup_b" and q in S}; cp = {q: v for (q, r), v in D.items() if r == "compute" and q in S}
            both = sorted(set(lk) & set(cp))
            if not lk or not cp:
                E(f"| {name} | {sn} | {len(lk)} / {len(cp)} / 0 | — | — | — | — | — |"); continue
            cl = [np.nanmean([lk[q][l] for q in lk]) for l in range(L)]; cc = [np.nanmean([cp[q][l] for q in cp]) for l in range(L)]
            ll, lc = loss_layer(cl), loss_layer(cc)
            dl = {}
            for q in both:
                a_, b_ = loss_layer(lk[q]), loss_layer(cp[q]); dl[q] = (a_ if a_ is not None else -1) - (b_ if b_ is not None else -1)
            pos = sum(v > 0 for v in dl.values()); neg = sum(v < 0 for v in dl.values()); zero = len(dl) - pos - neg
            ci = cluster_boot(dl, statistics.mean) if len(dl) >= 2 else (float("nan"),) * 2
            fl = lambda x: f"{x} ({x/(L-1):.2f})" if x is not None else "—"
            E(f"| {name} | {sn} | {len(lk)} / {len(cp)} / {len(both)} | {fl(ll)} | {fl(lc)} | {pos}:{zero}:{neg} | "
              f"{(statistics.mean(dl.values()) if dl else float('nan')):.2f} {fmt_ci(ci)} | {fmt_p(sign_test(pos, neg))} |")
    E()

    # ------------------------------------------------ 4. strata in pooled MP
    E('## 4. MP total by stratum (full-generation dL*): pages, operation, page order of A/B')
    E()
    E('| model | stratum | valid pairs | Compute first:same:reversed | mean dL* | pair sign test p |')
    E("|---|---|---|---|---|---|")
    strata = [('2 pages', lambda r: r["n_pages"] == 2), ('3+ pages', lambda r: r["n_pages"] >= 3), ('difference', lambda r: r["op"] == "difference"), ('sum', lambda r: r["op"] == "sum"),
              ('B on an earlier page than A', lambda r: r["b_page_pos"] < r["a_page_pos"]), ('B on a later page than A', lambda r: r["b_page_pos"] > r["a_page_pos"])]
    for tag, name in MODELS:
        gaps = GAPS[(tag, 'MP total (190+631)')]
        for sname, f in strata:
            g = [v for q, v in gaps.items() if f(poolrow[q])]
            if not g:
                E(f"| {name} | {sname} | 0 | — | — | — |"); continue
            a = sum(v > 0 for v in g); b = sum(v < 0 for v in g)
            E(f"| {name} | {sname} | {len(g)} | {a}:{len(g)-a-b}:{b} | {statistics.mean(g):.2f} | {fmt_p(sign_test(a, b))} |")
    E()
    (BASE / "paired_cfmp2_stats.md").write_text("\n".join(out) + "\n")
    print("\n".join(out)); print("-> paired_cfmp2_stats.md")


if __name__ == "__main__":
    random.seed(0); np.random.seed(0)
    main()
