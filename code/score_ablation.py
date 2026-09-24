#!/usr/bin/env python3
"""Score the cell-ablation pilot: EM/ANLS per condition, flip rates, McNemar."""
import json
import math
import re
import unicodedata
from pathlib import Path

BASE = Path(__file__).resolve().parent
PREDS = BASE / "preds_ablation_qwen35_9b.jsonl"
CONDS = ["none", "target", "control", "table"]


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    return re.sub(r"\s+", " ", s)


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred, gold):
    p, g = norm(pred), norm(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    d = levenshtein(p, g) / max(len(p), len(g))
    return 1.0 - d if d < 0.5 else 0.0


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


def main():
    rows = [json.loads(l) for l in open(PREDS)]
    print(f"items: {len(rows)}\n")

    print(f"{'condition':<10}{'EM':>8}{'ANLS':>8}{'changed vs base':>17}")
    scores = {c: [] for c in CONDS}
    for c in CONDS:
        em = [float(norm(r['preds'][c]) == norm(r['gold'])) for r in rows]
        an = [anls(r['preds'][c], r['gold']) for r in rows]
        ch = [r['preds'][c] != r['preds']['none'] for r in rows]
        scores[c] = an
        print(f"{c:<10}{100*sum(em)/len(em):>8.1f}{100*sum(an)/len(an):>8.2f}"
              f"{100*sum(ch)/len(ch):>16.1f}%")

    base_ok = [r for r in rows if anls(r['preds']['none'], r['gold']) >= 0.5]
    print(f"\nbaseline-correct subset (ANLS>=0.5): {len(base_ok)}")
    flips = {}
    for c in ("target", "control", "table"):
        f = [anls(r['preds'][c], r['gold']) < 0.5 for r in base_ok]
        flips[c] = f
        print(f"  {c:<8} broke {sum(f)}/{len(f)} ({100*sum(f)/len(f):.1f}%)")

    b = sum(1 for t, c in zip(flips['target'], flips['control']) if t and not c)
    c_ = sum(1 for t, c in zip(flips['target'], flips['control']) if not t and c)
    print(f"\nMcNemar target vs control: target-only broke {b}, "
          f"control-only broke {c_}, p={mcnemar_exact(b, c_):.4g}")

    print("\nper-item (baseline-correct):")
    for r in base_ok:
        t = anls(r['preds']['target'], r['gold']) < 0.5
        c = anls(r['preds']['control'], r['gold']) < 0.5
        mark = "T!" if t and not c else ("C!" if c and not t else
                                         ("TC" if t and c else "  "))
        print(f"  [{mark}] {r['qid']:<38} gold={r['gold']!r:<24} "
              f"target={r['preds']['target']!r:<24} "
              f"control={r['preds']['control']!r}")


if __name__ == "__main__":
    main()
