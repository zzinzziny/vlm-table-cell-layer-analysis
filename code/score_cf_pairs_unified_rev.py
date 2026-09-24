#!/usr/bin/env python3
import json, math, re, sys, unicodedata, statistics, random, itertools
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, wilcoxon

BASE = Path(__file__).resolve().parent
PY = None
random.seed(0); np.random.seed(0)
N_BOOT, N_PERM = 10000, 20000
GEMMA_BLOCK = [5, 11, 17, 23, 29, 35, 41, 47]
GEMMA_BLOCK_OF = {"gemma4_12b": GEMMA_BLOCK, "gemma4_31b": list(range(5, 60, 6))}

MODELS = [
    ("qwen35_9b", 'Qwen3.5-9B (32 layers)'), ("qwen3vl_8b", 'Qwen3-VL-8B (36 layers)'), ("qwen25_7b", 'Qwen2.5-VL-7B (28 layers)'),
    ("gemma4_12b", 'Gemma 4 12B (48 layers)'), ("ministral3_8b", 'Ministral 3 8B (34 layers)'), ("llava_ov2_8b", 'LLaVA-OV2 8B (36 layers)'),
    ("qwen3vl_32b", 'Qwen3-VL-32B (64 layers)'), ("gemma4_31b", 'Gemma 4 31B (60 layers)'),
]
SP_GATES = {
    "qwen35_9b": ["cf2_gates.jsonl", "cf3_gates.jsonl", "cf5_gates_qwen35_9b.jsonl", "cf4_gates_qwen35_9b.jsonl"],
    "qwen3vl_8b": ["cf_gates_qwen3vl_8b.jsonl", "cf5_gates_qwen3vl_8b.jsonl"],
    "qwen25_7b": ["cf_gates_qwen25_7b.jsonl", "cf5_gates_qwen25_7b.jsonl"],
    "gemma4_12b": ["cf_gates_gemma4_12b.jsonl", "cf5_gates_gemma4_12b.jsonl"],
    "ministral3_8b": ["cf_gates_ministral3_8b.jsonl"],
    "llava_ov2_8b": ["cf_gates_llava_ov2_8b.jsonl"],
    "qwen3vl_32b": ["cf_gates_qwen3vl_32b.jsonl", "cf5_gates_qwen3vl_32b.jsonl"],
    "gemma4_31b": ["cf_gates_gemma4_31b.jsonl", "cf5_gates_gemma4_31b.jsonl"],
}
SP_PATCH = {
    "qwen35_9b": ["cf_patch_cf2_qwen35_9b.jsonl", "cf_patch_cf3_qwen35_9b.jsonl", "cf5_patch_qwen35_9b.jsonl"] + (["cf4_patch_qwen35_9b.jsonl"] if (Path(__file__).resolve().parent / "cf4_patch_qwen35_9b.jsonl").exists() else []),
    "qwen3vl_8b": ["cf_patch_qwen3vl_8b.jsonl", "cf5_patch_qwen3vl_8b.jsonl"],
    "qwen25_7b": ["cf_patch_qwen25_7b.jsonl", "cf_patch_qwen25_7b_ext.jsonl", "cf5_patch_qwen25_7b.jsonl"],
    "gemma4_12b": ["cf_patch_gemma4_12b.jsonl", "cf_patch_gemma4_12b_ext.jsonl", "cf5_patch_gemma4_12b.jsonl"],
    "ministral3_8b": ["cf_patch_ministral3_8b.jsonl"],
    "llava_ov2_8b": ["cf_patch_llava_ov2_8b.jsonl"],
    "qwen3vl_32b": ["cf_patch_qwen3vl_32b.jsonl", "cf5_patch_qwen3vl_32b.jsonl"],
    "gemma4_31b": ["cf_patch_gemma4_31b.jsonl", "cf5_patch_gemma4_31b.jsonl"],
}
POOL_OF = {"cf2": 100, "cf3": 150, "cf4": 160, "cf5": 329, "cfmp": 190}
GATE_KEYS = ["base_lookup_a", "base_lookup_b", "base_compute", "donor_lookup_b", "donor_compute"]


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    return re.sub(r"\s+", " ", s.replace("−", "-").replace(",", "")).rstrip(".")


def load_jsonl(f):
    return [json.loads(l) for l in open(BASE / f)]


def pool_of(qid):
    return qid.split("_")[0]


def doc_of(qid):
    return qid.split("_")[1]


# ---------------------------------------------------------------- endpoints
def endpoints(files, block=None):
    """qid -> (lookup_end, compute_end); None if no donor-flip layer. Returns also set of patched qids."""
    by = defaultdict(dict)
    for f in files:
        for r in load_jsonl(f):
            by[(r["qid"], r["role"])][r["layer"]] = r
    qids = sorted({q for q, _ in by})
    ends = {}
    for q in qids:
        e = []
        for role in ("lookup_b", "compute"):
            d = by.get((q, role), {})
            dl = [l for l, r in d.items() if l >= 0 and norm(r["pred"]) == norm(r["ans_donor"])]
            v = max(dl) if dl else None
            if v is not None and block:
                v = next((g for g in block if g >= v), v)
            e.append(v)
        ends[q] = tuple(e)
    return ends


def classify(ends):
    out = {"both": [], "earlier": [], "same": [], "later": [], "one": [], "none": [], "gap": {}}
    for q, (le, ce) in ends.items():
        if le is not None and ce is not None:
            out["both"].append(q); g = le - ce; out["gap"][q] = g
            (out["earlier"] if g > 0 else out["later"] if g < 0 else out["same"]).append(q)
        elif le is None and ce is None:
            out["none"].append(q)
        else:
            out["one"].append(q)
    return out


# ---------------------------------------------------------------- statistics
def sign_test(a, b):
    n = a + b
    return binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue if n else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def cluster_boot(gaps, stat, n=N_BOOT, seed=0):
    """gaps: dict qid->gap. Resample documents with replacement; stat(list_of_gaps) -> float."""
    rng = random.Random(seed)
    bydoc = defaultdict(list)
    for q, g in gaps.items():
        bydoc[doc_of(q)].append(g)
    docs = list(bydoc); vals = []
    for _ in range(n):
        s = [g for d in rng.choices(docs, k=len(docs)) for g in bydoc[d]]
        vals.append(stat(s))
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def doc_level(gaps):
    bydoc = defaultdict(list)
    for q, g in gaps.items():
        bydoc[doc_of(q)].append(g)
    dm = {d: statistics.mean(v) for d, v in bydoc.items()}
    means = list(dm.values())
    pos = sum(1 for m in means if m > 0); neg = sum(1 for m in means if m < 0)
    # majority by sign sum (as in score_cf_pairs_cluster.py)
    maj = Counter()
    for d, v in bydoc.items():
        t = sum((1 if g > 0 else -1 if g < 0 else 0) for g in v)
        maj["pos" if t > 0 else "neg" if t < 0 else "tie"] += 1
    try:
        wp = wilcoxon(means, alternative="two-sided", zero_method="wilcox").pvalue if sum(1 for m in means if m != 0) >= 1 else float("nan")
    except Exception:
        wp = float("nan")
    # doc-weighted mean CI: bootstrap over documents of mean of doc means
    rng = random.Random(1); docs = list(dm); bs = []
    for _ in range(N_BOOT):
        bs.append(statistics.mean(dm[d] for d in rng.choices(docs, k=len(docs))))
    return {"n_doc": len(bydoc), "pairs_per_doc_med": statistics.median(len(v) for v in bydoc.values()),
            "pairs_per_doc_max": max(len(v) for v in bydoc.values()),
            "doc_mean": statistics.mean(means), "doc_mean_ci": (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))),
            "doc_pos": pos, "doc_neg": neg, "doc_zero": len(means) - pos - neg, "doc_sign_p": sign_test(pos, neg), "doc_wilcoxon_p": wp,
            "maj": maj, "maj_p": sign_test(maj["pos"], maj["neg"]), "bydoc": bydoc}


def signflip_perm(gaps, n=N_PERM, seed=0):
    """statistic = pair-weighted mean ΔL* (sum of all gaps / n_pairs); flip sign of every pair within a document jointly."""
    bydoc = defaultdict(list)
    for q, g in gaps.items():
        bydoc[doc_of(q)].append(g)
    docs = list(bydoc); sums = np.array([sum(bydoc[d]) for d in docs]); N = len(gaps)
    obs = sums.sum() / N
    rng = np.random.default_rng(seed)
    flips = rng.choice([1, -1], size=(n, len(docs)))
    stats = (flips @ sums) / N
    cnt = int((np.abs(stats) >= abs(obs) - 1e-12).sum())
    return obs, (cnt / n if cnt else 0.5 / n)


def fmt_p(p):
    if p != p:
        return "—"
    return f"{p:.2g}" if p >= 1e-4 else f"{p:.1e}"


def fmt_perm(p):
    return f"<{1/N_PERM:.0e}" if p <= 0.5 / N_PERM + 1e-15 else f"{p:.2g}"


def fmt_ci(ci, nd=2):
    return f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


# ---------------------------------------------------------------- main
def main():
    out = []
    E = lambda s="": out.append(s)
    E('# Paired LOOKUP-COMPUTE: common denominators, scoring and statistics')
    E()
    E('Endpoint: the last layer l >= 0 at which the greedy answer equals the donor answer (after normalisation); a role is valid if such a layer exists. A pair has dL* = LOOKUP endpoint - COMPUTE endpoint only if both roles (lookup_b, compute) are valid. Compute first = dL* > 0, same = 0, reversed = < 0.')
    E()

    # ---------------- load pools
    pool_sp = load_jsonl("cf_pool_qwen35.jsonl") + load_jsonl("cf5_pool_qwen35.jsonl")
    pool_mp = load_jsonl("cfmp_pool_qwen35.jsonl")
    poolrow = {r["qid"]: r for r in pool_sp + pool_mp}
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--exclude-result-in-table", action="store_true"); ap.add_argument("--sp-ring1", action="store_true", help='SP patches: cell + radius-1 injection (ring1_sp/); MP uses the cell-only patches'); args = ap.parse_args()
    if args.sp_ring1:
        for _t in SP_PATCH: SP_PATCH[_t] = ["ring1_sp/" + f for f in SP_PATCH[_t]]
        E('> **SP patches = cell + radius-1 token injection (ring1_sp/)**. MP uses the cell-only patches.'); E()
    EXCL = {q for q, r in poolrow.items() if str(r.get("result_in_table")) == "True"} if args.exclude_result_in_table else set()
    if EXCL:
        pool_sp = [r for r in pool_sp if r["qid"] not in EXCL]; pool_mp = [r for r in pool_mp if r["qid"] not in EXCL]
        E(f"> **Denominator variant: candidates whose result already appears in the table are excluded**. SP {sum(1 for q in EXCL if pool_of(q) != 'cfmp')} · MP {sum(1 for q in EXCL if pool_of(q) == 'cfmp')} excluded -> SP {len(pool_sp)} · MP {len(pool_mp)}. The candidates column is the actual denominator.")
        E()
    keep = lambda d: {q: v for q, v in d.items() if q not in EXCL}

    # ---------------- gates / patches per model
    G = {}   # model -> {"sp": {qid: gate_row}, "mp": {...}}
    P = {}   # model -> {"sp": ends, "mp": ends}
    C = {}
    for tag, _ in MODELS:
        gsp = {}
        for f in SP_GATES[tag]:
            for r in load_jsonl(f):
                gsp[r["qid"]] = r
        gmp = {r["qid"]: r for r in load_jsonl(f"cfmp_gates_{tag}.jsonl")}
        gsp, gmp = keep(gsp), keep(gmp)
        G[tag] = {"sp": gsp, "mp": gmp}
        esp = keep(endpoints(SP_PATCH[tag])); emp = keep(endpoints([f"cfmp_patch_{tag}.jsonl"]))
        P[tag] = {"sp": esp, "mp": emp}
        C[tag] = {"sp": classify(esp), "mp": classify(emp)}
        if tag in ("gemma4_12b", "gemma4_31b"):
            P[tag]["sp_block"] = keep(endpoints(SP_PATCH[tag], GEMMA_BLOCK_OF[tag])); P[tag]["mp_block"] = keep(endpoints([f"cfmp_patch_{tag}.jsonl"], GEMMA_BLOCK_OF[tag]))
            C[tag]["sp_block"] = classify(P[tag]["sp_block"]); C[tag]["mp_block"] = classify(P[tag]["mp_block"])

    # ================================================================ 1. funnel
    E('## 1. Funnel: candidates -> gate passed -> patched -> both valid -> non-tied')
    E()
    E('Gate = all 5 questions correct (base lookup-A/lookup-B/compute, donor lookup-B/compute). non-tied = Compute first + reversed (denominator of the sign test). one side only = only one of the two roles is valid.')
    E()
    E('### 1a. Single page (SP), per model')
    E()
    E('| model | candidates | gate passed | patched | both valid | non-tied | Compute first | same | reversed | one side only | both invalid |')
    E("|---|---|---|---|---|---|---|---|---|---|---|")
    funnel = {}
    for tag, name in MODELS:
        g = G[tag]["sp"]; c = C[tag]["sp"]
        n_c = len(g); n_pass = sum(1 for r in g.values() if r["pass_patch"]); n_patch = len(P[tag]["sp"])
        funnel[(tag, "sp")] = (n_c, n_pass, n_patch, len(c["both"]), len(c["earlier"]) + len(c["later"]), len(c["earlier"]), len(c["same"]), len(c["later"]), len(c["one"]), len(c["none"]))
        E(f"| {name} | {n_c} | {n_pass} | {n_patch} | {len(c['both'])} | {len(c['earlier'])+len(c['later'])} | {len(c['earlier'])} | {len(c['same'])} | {len(c['later'])} | {len(c['one'])} | {len(c['none'])} |")
    E()
    E('### 1b. Single page (SP), model x pool')
    E()
    E('| model | pool | candidates | gate passed | patched | both valid | non-tied | Compute first | same | reversed | one side only | both invalid |')
    E("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, name in MODELS:
        g = G[tag]["sp"]; e = P[tag]["sp"]
        for pool in ("cf2", "cf3", "cf4", "cf5"):
            gq = {q: r for q, r in g.items() if pool_of(q) == pool}
            if not gq:
                E(f"| {name} | {pool} | 0 (not gated) | - | - | - | - | - | - | - | - | - |"); continue
            c = classify({q: v for q, v in e.items() if pool_of(q) == pool})
            n_pass = sum(1 for r in gq.values() if r["pass_patch"]); n_patch = sum(1 for q in e if pool_of(q) == pool)
            E(f"| {name} | {pool} | {len(gq)} | {n_pass} | {n_patch} | {len(c['both'])} | {len(c['earlier'])+len(c['later'])} | {len(c['earlier'])} | {len(c['same'])} | {len(c['later'])} | {len(c['one'])} | {len(c['none'])} |")
    E()
    E('### 1c. Multi-page (MP, cfmp 190)')
    E()
    E('| model | candidates | gate passed | patched | both valid | non-tied | Compute first | same | reversed | one side only | both invalid |')
    E("|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, name in MODELS:
        g = G[tag]["mp"]; c = C[tag]["mp"]
        n_pass = sum(1 for r in g.values() if r["pass_patch"]); n_patch = len(P[tag]["mp"])
        funnel[(tag, "mp")] = (len(g), n_pass, n_patch, len(c["both"]), len(c["earlier"]) + len(c["later"]), len(c["earlier"]), len(c["same"]), len(c["later"]), len(c["one"]), len(c["none"]))
        E(f"| {name} | {len(g)} | {n_pass} | {n_patch} | {len(c['both'])} | {len(c['earlier'])+len(c['later'])} | {len(c['earlier'])} | {len(c['same'])} | {len(c['later'])} | {len(c['one'])} | {len(c['none'])} |")
    E()
    E('### 1d. Accuracy of each gate question and the bottleneck (share of failed candidates that got each question wrong)')
    E()
    E('| model | condition | n | base lookup-A | base lookup-B | base compute | donor lookup-B | donor compute | failed: both computes wrong | failed: only lookups wrong |')
    E("|---|---|---|---|---|---|---|---|---|---|")
    for tag, name in MODELS:
        for cond in ("sp", "mp"):
            g = list(G[tag][cond].values()); n = len(g)
            acc = {k: sum(1 for r in g if r["ok"][k]) for k in GATE_KEYS}
            fail = [r for r in g if not r["pass_patch"]]
            both_c = sum(1 for r in fail if not r["ok"]["base_compute"] and not r["ok"]["donor_compute"])
            lk_only = sum(1 for r in fail if r["ok"]["base_compute"] and r["ok"]["donor_compute"])
            E(f"| {name} | {cond.upper()} | {n} | " + " | ".join(f"{100*acc[k]/n:.0f}%" for k in GATE_KEYS) + f" | {both_c}/{len(fail)} | {lk_only}/{len(fail)} |")
    E()
    E('The bottleneck in every model and condition is arithmetic (base/donor compute); few candidates fail on lookup alone. Because the gate pass rate reflects the two-digit arithmetic accuracy of each model, the per-model samples differ (selection toward items the model computes correctly); section 4 analyses the common items.')
    E()

    # ================================================================ 2. duplication
    E('## 2. Pool overlap (documents, tables, pages, images, cells)')
    E()

    def dup_stats(rows):
        docs = {r["doc_id"] for r in rows}; tabs = {(r["doc_id"], r["table_index"]) for r in rows}
        pages = {p["page_id"] for r in rows for p in r["pages"]}
        imgs = {r.get("base_image") or tuple(r.get("base_images", [])) for r in rows}
        bcell = Counter((r["doc_id"], r["table_index"], r["operands"][1]["row_idx"], r["operands"][1]["col_idx"]) for r in rows)
        ab = Counter((r["doc_id"], r["table_index"], r["operands"][0]["row_idx"], r["operands"][1]["row_idx"], r["operands"][1]["col_idx"]) for r in rows)
        dc = Counter(r["doc_id"] for r in rows)
        return dict(n=len(rows), docs=len(docs), tabs=len(tabs), pages=len(pages), imgs=len(imgs), bcells=len(bcell),
                    bcell_reuse=sum(1 for c in bcell.values() if c > 1), bcell_max=max(bcell.values()) if bcell else 0,
                    ab_dup=sum(1 for c in ab.values() if c > 1), doc_med=statistics.median(dc.values()) if dc else 0, doc_max=max(dc.values()) if dc else 0,
                    doc_ge10=sum(1 for c in dc.values() if c >= 10), ops=Counter(r["op"] for r in rows), npages=Counter(r["n_pages"] for r in rows),
                    docset=docs, tabset=tabs)
    groups = {"cf2": [r for r in pool_sp if pool_of(r["qid"]) == "cf2"], "cf3": [r for r in pool_sp if pool_of(r["qid"]) == "cf3"],
              "cf4": [r for r in pool_sp if pool_of(r["qid"]) == "cf4"], "cf5": [r for r in pool_sp if pool_of(r["qid"]) == "cf5"],
              'SP total (739)': pool_sp, "MP cfmp(190)": pool_mp}
    E('| pool | n | documents | tables | pages | base images | distinct B cells | reused B cells (>1, max) | identical (A,B) pairs | candidates per document median/max | documents with >=10 | operation (difference/sum) |')
    E("|---|---|---|---|---|---|---|---|---|---|---|---|")
    D = {}
    for k, rows in groups.items():
        d = dup_stats(rows); D[k] = d
        E(f"| {k} | {d['n']} | {d['docs']} | {d['tabs']} | {d['pages']} | {d['imgs']} | {d['bcells']} | {d['bcell_reuse']} ({d['bcell_max']}) | {d['ab_dup']} | {d['doc_med']:g}/{d['doc_max']} | {d['doc_ge10']} | {d['ops']['difference']}/{d['ops']['sum']} |")
    sp, mp = D['SP total (739)'], D["MP cfmp(190)"]
    # cross-pool overlaps
    E()
    E(f"- Document overlap between cf2/cf3/cf4/cf5 within the SP pool: " + ", ".join(
        f"{a}∩{b} documents {len(D[a]['docset'] & D[b]['docset'])}, tables {len(D[a]['tabset'] & D[b]['tabset'])}" for a, b in itertools.combinations(["cf2", "cf3", "cf4", "cf5"], 2)))
    E(f"- SP and MP share: documents {len(sp['docset'] & mp['docset'])} / tables {len(sp['tabset'] & mp['tabset'])} (of the 53 MP tables, {len(sp['tabset'] & mp['tabset'])} also appear in SP). SP items are all single-page; MP page counts {dict(sorted(mp['npages'].items()))}.")
    E(f"- The 739 SP candidates are distinct (A,B) pairs; B cells reused with a different A: {sp['bcell_reuse']} cells (max {sp['bcell_max']} times). Each candidate has its own render (different donor value), from {sp['pages']} source pages. MP base image sets: {mp['imgs']} (shared per table).")
    E()
    E('### 2b. Documents and tables with valid pairs (per model)')
    E()
    E('| model | condition | valid pairs | documents | tables | pages | pairs per document median (max) | distinct B cells |')
    E("|---|---|---|---|---|---|---|---|")
    for tag, name in MODELS:
        for cond in ("sp", "mp"):
            both = C[tag][cond]["both"]; rows = [poolrow[q] for q in both if q in poolrow]
            if not rows:
                E(f"| {name} | {cond.upper()} | 0 | — | — | — | — | — |"); continue
            d = dup_stats(rows)
            E(f"| {name} | {cond.upper()} | {len(both)} | {d['docs']} | {d['tabs']} | {d['pages']} | {d['doc_med']:g} ({d['doc_max']}) | {d['bcells']} |")
    E()

    # ================================================================ 3. statistics
    E('## 3. Statistics and document-level effects and CIs')
    E()
    E('**Specification**')
    E('- Pair sign test: exact two-sided binomial test over non-tied pairs (dL* != 0), H0: P(Compute first) = 0.5; tied pairs excluded. Compute-first rate = share among non-tied pairs, Wilson 95% CI.')
    E('- 95% CI of median and mean dL*: document-clustered bootstrap (resample documents = PMC id, the second token of the qid, pool all their pairs, recompute), 10,000 resamples, percentile CI; pair-weighted.')
    E('- Document-level effect: mean dL* per document as one observation; document-weighted mean with its bootstrap CI, sign test of document means (>0 vs <0, 0 excluded) and two-sided Wilcoxon signed-rank test.')
    E('- Document majority: sum of pair signs within a document, + = Compute-first document, - = reversed document, 0 = tie; exact binomial test excluding ties.')
    E('- Sign-flip permutation: statistic = pair-weighted mean dL* (sum over all pairs / number of pairs); the signs of all pairs of a document are flipped together, 20,000 random sign sets, two-sided (|stat| >= |observed|).')
    E('- For Gemma the endpoints are also reported rounded up to its global-attention layers (5, 11, ..., 47) (block version).')
    E()
    hdr = ('| model | condition | valid pairs | Compute first:same:reversed | Compute-first rate among non-tied [Wilson 95%] | pair sign test p | median dL* [cluster boot CI] | mean dL* [cluster boot CI] | documents (pairs/doc median, max) | document mean dL* [boot CI] | document signs +:0:- (p) | Wilcoxon p | document majority +:tie:- (p) | permutation p |')
    E(hdr); E("|" + "---|" * 14)
    stats_rows = {}
    for tag, name in MODELS:
        conds = [("sp", "SP"), ("mp", "MP")] + ([("sp_block", 'SP block'), ("mp_block", 'MP block')] if tag in ("gemma4_12b", "gemma4_31b") else [])
        for ck, cl in conds:
            c = C[tag][ck]; gaps = c["gap"]
            if not gaps:
                E(f"| {name} | {cl} | 0 | — | — | — | — | — | — | — | — | — | — | — |"); continue
            a, b = len(c["earlier"]), len(c["later"]); n_nt = a + b
            wl = wilson(a, n_nt)
            med = statistics.median(gaps.values()); mean = statistics.mean(gaps.values())
            med_ci = cluster_boot(gaps, statistics.median); mean_ci = cluster_boot(gaps, statistics.mean)
            dl = doc_level(gaps); obs, pp = signflip_perm(gaps)
            stats_rows[(tag, ck)] = dict(n=len(gaps), a=a, b=b, same=len(c["same"]), prop=a / n_nt if n_nt else float("nan"), wl=wl, sp=sign_test(a, b), med=med, med_ci=med_ci, mean=mean, mean_ci=mean_ci, dl=dl, perm=pp)
            E(f"| {name} | {cl} | {len(gaps)} | {a}:{len(c['same'])}:{b} | {100*a/n_nt if n_nt else float('nan'):.0f}% [{100*wl[0]:.0f}, {100*wl[1]:.0f}] | {fmt_p(sign_test(a, b))} | {med:g} {fmt_ci(med_ci, 1)} | {mean:.2f} {fmt_ci(mean_ci)} "
              f"| {dl['n_doc']} ({dl['pairs_per_doc_med']:g}·{dl['pairs_per_doc_max']}) | {dl['doc_mean']:.2f} {fmt_ci(dl['doc_mean_ci'])} | {dl['doc_pos']}:{dl['doc_zero']}:{dl['doc_neg']} ({fmt_p(dl['doc_sign_p'])}) | {fmt_p(dl['doc_wilcoxon_p'])} "
              f"| {dl['maj']['pos']}:{dl['maj']['tie']}:{dl['maj']['neg']} ({fmt_p(dl['maj_p'])}) | {fmt_perm(pp)} |")
    E()
    # endpoint distributions
    E('### 3b. Endpoint distribution (valid pairs, SP)')
    E()
    E('| model | LOOKUP endpoints (layer:pairs) | COMPUTE endpoints (layer:pairs) |')
    E("|---|---|---|")
    for tag, name in MODELS:
        e = P[tag]["sp"]; both = C[tag]["sp"]["both"]
        lc = Counter(e[q][0] for q in both); cc = Counter(e[q][1] for q in both)
        E(f"| {name} | " + " ".join(f"{l}:{n}" for l, n in sorted(lc.items())) + " | " + " ".join(f"{l}:{n}" for l, n in sorted(cc.items())) + " |")
    E()

    # ================================================================ 4. common items
    E('## 4. Common-item analysis (supplementary; the items passing the gate differ by model)')
    E()
    def common_block(cond, label):
        passed = {tag: {q for q, r in G[tag][cond].items() if r["pass_patch"]} for tag, _ in MODELS}
        patched = {tag: set(P[tag][cond]) for tag, _ in MODELS}
        both_valid = {tag: set(C[tag][cond]["both"]) for tag, _ in MODELS}
        tags = [t for t, _ in MODELS]
        E(f"### 4{label}. {cond.upper()}")
        E()
        E('Pairwise intersection of gate-passed qids (row x column):')
        E()
        E("| | " + " | ".join(n.split(' (')[0] for _, n in MODELS) + " |"); E("|---|" + "---|" * len(MODELS))
        for t1, n1 in MODELS:
            E(f"| {n1.split(' (')[0]} | " + " | ".join(str(len(passed[t1] & passed[t2])) if t1 != t2 else f"**{len(passed[t1])}**" for t2, _ in MODELS) + " |")
        E()
        sets = {
            'passed in all 6 models': set.intersection(*[passed[t] for t in tags]),
            'passed and patched in all 6 models': set.intersection(*[passed[t] & patched[t] for t in tags]),
            'both valid in all 6 models': set.intersection(*[both_valid[t] for t in tags]),
            '4 models (Qwen3.5, Qwen3-VL 8B, Qwen2.5-VL, Gemma): passed and patched': set.intersection(*[passed[t] & patched[t] for t in tags[:4]]),
            '3 Qwen models: passed and patched': set.intersection(*[passed[t] & patched[t] for t in tags[:3]]),
        }
        from collections import Counter as _C
        cnt_pass = _C(q for t in tags for q in (passed[t] & patched[t]))
        sets['passed and patched in >=4 models (each item counted for the models it passed)'] = {q for q, c in cnt_pass.items() if c >= 4}
        # 9B has no cf4; note it
        for sname, S in sets.items():
            E(f"**{sname}: {len(S)}**" + (f" (documents {len({doc_of(q) for q in S})})" if S else ""))
            if not S:
                E(); continue
            E()
            E('| model | both valid in this set | Compute first:same:reversed | median dL* | pair sign test p |')
            E("|---|---|---|---|---|")
            for tag, name in MODELS:
                if sname.startswith('4 models') and tag not in tags[:4]:
                    continue
                if sname.startswith('3 Qwen models') and tag not in tags[:3]:
                    continue
                for ck, lab in ([(cond, "")] + ([(cond + "_block", ' (block)')] if tag in ("gemma4_12b", "gemma4_31b") else [])):
                    e = P[tag][ck]; sub = {q: e[q] for q in S if q in e}
                    c = classify(sub); g = list(c["gap"].values())
                    E(f"| {name}{lab} | {len(c['both'])}/{len(sub)} | {len(c['earlier'])}:{len(c['same'])}:{len(c['later'])} | {statistics.median(g) if g else '—'} | {fmt_p(sign_test(len(c['earlier']), len(c['later'])))} |")
            E()
        E(f"Note: the 9B did not gate cf4 in this version, so the all-models-passed sets can only come from cf2/cf3/cf5 + cfmp.")
        E()
        E('**Agreement of dL* between model pairs on common valid pairs** (same qid valid in both models; sign match = identical sign of dL* including 0; direction match = both >0 or both <=0; Spearman rho):')
        E()
        E('| model A | model B | common valid pairs | A Compute first:same:reversed | B Compute first:same:reversed | exact sign match | direction match (>0 vs <=0) | Spearman rho (p) |')
        E("|---|---|---|---|---|---|---|---|")
        from scipy.stats import spearmanr
        for (t1, n1), (t2, n2) in itertools.combinations(MODELS, 2):
            g1, g2 = C[t1][cond]["gap"], C[t2][cond]["gap"]
            com = sorted(set(g1) & set(g2))
            if len(com) < 3:
                E(f"| {n1.split(' (')[0]} | {n2.split(' (')[0]} | {len(com)} | — | — | — | — | — |"); continue
            a = [g1[q] for q in com]; b = [g2[q] for q in com]
            sg = lambda x: (x > 0) - (x < 0)
            exact_ = sum(1 for x, y in zip(a, b) if sg(x) == sg(y)); dire = sum(1 for x, y in zip(a, b) if (x > 0) == (y > 0))
            rho, rp = spearmanr(a, b) if len(set(a)) > 1 and len(set(b)) > 1 else (float("nan"), float("nan"))
            tri = lambda v: f"{sum(1 for x in v if x>0)}:{sum(1 for x in v if x==0)}:{sum(1 for x in v if x<0)}"
            E(f"| {n1.split(' (')[0]} | {n2.split(' (')[0]} | {len(com)} | {tri(a)} | {tri(b)} | {exact_}/{len(com)} | {dire}/{len(com)} | {rho:.2f} ({fmt_p(rp)}) |")
        E()
    common_block("sp", "a"); common_block("mp", "b")

    outname = "paired_unified_stats_rev_noRIT.md" if EXCL else "paired_unified_stats_rev.md"
    if args.sp_ring1: outname = outname.replace(".md", "_spring1.md")
    (BASE / outname).write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"-> {outname}")


if __name__ == "__main__":
    main()
