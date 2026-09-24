#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
def digits_targets(vals):
    m = max(len(v) for v in vals); out = np.full((len(vals), m), -1, dtype=int)
    for i, v in enumerate(vals):
        for j, ch in enumerate(reversed(v)): out[i, j] = int(ch)
    return out
def fit_predict(Xtr, ytr, Xte, seed):
    if len(set(ytr.tolist())) < 2: return np.full(len(Xte), ytr[0])
    sc = StandardScaler().fit(Xtr); clf = LogisticRegression(C=0.05, max_iter=2000, random_state=seed)
    clf.fit(sc.transform(Xtr), ytr); return clf.predict(sc.transform(Xte))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--featdir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="all"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--nfold", type=int, default=5)
    args = ap.parse_args(); fd = Path(args.featdir); rng = np.random.RandomState(args.seed)
    metas = [json.loads(l) for l in open(fd / "meta.jsonl") if not l.startswith("#")]
    metas = [m for m in metas if (fd / f"{m['qid']}.npz").exists()]
    feats = np.stack([np.load(fd / f"{m['qid']}.npz")["feat"] for m in metas])   # [n, 2, L+1, 4, d] float16
    n, P, L1, S, d = feats.shape; sites = metas[0]["sites"]; prompts = metas[0]["prompts"]
    layers = list(range(L1)) if args.layers == "all" else list(range(0, L1, 2)) + ([L1 - 1] if (L1 - 1) % 2 else [])
    docs = np.array([m["doc_id"] for m in metas]); ud = np.unique(docs); perm = rng.permutation(len(ud))
    fold_of_doc = {ud[perm[i]]: i % args.nfold for i in range(len(ud))}; fold = np.array([fold_of_doc[dd] for dd in docs])
    B = [m["B"] for m in metas]; A = [m["A"] for m in metas]; R = [m["R"] for m in metas]
    targets = {"B_first": np.array([int(v[0]) for v in B]), "A_first": np.array([int(v[0]) for v in A]), "R_first": np.array([int(v[0]) for v in R])}
    Bd, Rd = digits_targets(B), digits_targets(R)
    for j in range(Bd.shape[1]): targets[f"B_d{j}"] = Bd[:, j]
    for j in range(Rd.shape[1]): targets[f"R_d{j}"] = Rd[:, j]
    out = {"meta": json.dumps({"qids": [m["qid"] for m in metas], "docs": docs.tolist(), "fold": fold.tolist(), "layers": layers, "sites": sites, "prompts": prompts, "targets": list(targets)})}
    lines = [f"# probes {args.featdir}: n={n}, docs={len(ud)}, layers={len(layers)}/{L1}, d={d}"]
    summary = {}
    for tname, y in targets.items():
        valid = y >= 0; maj = np.bincount(y[valid]).max() / valid.sum()
        lines.append(f"\n## target {tname} (n={valid.sum()}, majority {maj:.3f})")
        lines.append("| prompt | site | " + " | ".join(f"L{l}" for l in layers) + " |"); lines.append("|---|---|" + "---|" * len(layers))
        for pi, pr in enumerate(prompts):
            for si, st in enumerate(sites):
                pred = np.full(n, -1); pred_sh = np.full(n, -1)
                for l in layers:
                    X = feats[:, pi, l, si, :].astype(np.float32)
                    for f in range(args.nfold):
                        tr = (fold != f) & valid; te = (fold == f) & valid
                        if te.sum() == 0 or tr.sum() < 10: continue
                        pred[te] = fit_predict(X[tr], y[tr], X[te], args.seed)
                        ysh = y[tr].copy(); rng.shuffle(ysh); pred_sh[te] = fit_predict(X[tr], ysh, X[te], args.seed)
                    out[f"pred__{tname}__{pr}__{st}__L{l}"] = pred.copy(); out[f"predsh__{tname}__{pr}__{st}__L{l}"] = pred_sh.copy()
                accs = [(out[f"pred__{tname}__{pr}__{st}__L{l}"][valid] == y[valid]).mean() for l in layers]
                shs = [(out[f"predsh__{tname}__{pr}__{st}__L{l}"][valid] == y[valid]).mean() for l in layers]
                summary[(tname, pr, st)] = (accs, shs)
                lines.append(f"| {pr} | {st} | " + " | ".join(f"{a:.2f}" for a in accs) + " |")
                lines.append(f"| {pr} | {st} (shuffle) | " + " | ".join(f"{a:.2f}" for a in shs) + " |")
    for base, D in (("B", Bd), ("R", Rd)):
        lines.append(f"\n## target {base}_full (all digit positions correct; up to {D.shape[1]} digits)")
        lines.append("| prompt | site | " + " | ".join(f"L{l}" for l in layers) + " |"); lines.append("|---|---|" + "---|" * len(layers))
        for pr in prompts:
            for st in sites:
                accs = []
                for l in layers:
                    ok = np.ones(n, bool)
                    for j in range(D.shape[1]):
                        p = out[f"pred__{base}_d{j}__{pr}__{st}__L{l}"]; ok &= (p == D[:, j]) | (D[:, j] < 0)
                    out[f"full__{base}__{pr}__{st}__L{l}"] = ok.astype(int); accs.append(ok.mean())
                lines.append(f"| {pr} | {st} | " + " | ".join(f"{a:.2f}" for a in accs) + " |")
    np.savez_compressed(args.out + ".npz", **out); Path(args.out + ".md").write_text("\n".join(lines) + "\n"); print("\n".join(lines[:6])); print("->", args.out + ".md/.npz")
if __name__ == "__main__": main()
