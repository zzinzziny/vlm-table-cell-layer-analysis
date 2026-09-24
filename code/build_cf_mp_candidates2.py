#!/usr/bin/env python3
import argparse, json, os, random, re, sys
from collections import defaultdict, Counter
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cf_candidates_relaxed import render_donor, phrase_col, LABELS  # noqa
BASE = Path(__file__).resolve().parent; OUTD = BASE / "cf2_images"; IMG = "../data/external/pubtables-v2/Full Documents/test/images"
INT = re.compile(r"^\d{2,4}$")
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--labels", default=str(LABELS)); ap.add_argument("--cap-table", type=int, default=8); ap.add_argument("--cap-doc", type=int, default=12)
    ap.add_argument("--max-pages", type=int, default=4); ap.add_argument("--target", type=int, default=800); ap.add_argument("--render", action="store_true")
    ap.add_argument("--exclude-rit", action="store_true"); ap.add_argument("--exclude-existing", default=str(BASE / "cfmp_candidates.jsonl"))
    ap.add_argument("--out", default=str(BASE / "cfmp2_candidates.jsonl")); ap.add_argument("--prefix", default="cfmp2"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--a-unique", default="table", choices=["table", "col"], help='uniqueness scope of the A value: table (original) | col (within the same column, cf2 rule)'); args = ap.parse_args()
    rng = random.Random(args.seed)
    excl = set()
    if args.exclude_existing and Path(args.exclude_existing).exists():
        for l in open(args.exclude_existing):
            c = json.loads(l); A, B = c["operands"]
            excl.add((c["doc_id"], c["table_index"], str(A["col_header"]), str(A["row_header"]).strip(), str(B["row_header"]).strip()))
            excl.add((c["doc_id"], c["table_index"], str(A["col_header"]), str(B["row_header"]).strip(), str(A["row_header"]).strip()))
    cands = []; per_doc = defaultdict(int); stat = Counter()
    for line in open(args.labels):
        d = json.loads(line); doc, tbl = d["doc_id"], d["table_index"]; pages = d["pages"]
        if len(pages) < 2 or len(pages) > args.max_pages or per_doc[doc] >= args.cap_doc: continue
        stat["tables"] += 1
        all_vals = [str(c.get("value", "")).strip() for p in pages for c in p.get("cells", [])]; val_cnt = Counter(all_vals)
        bycol = defaultdict(list)
        for pi, p in enumerate(pages):
            for c in p.get("cells", []):
                if INT.match(str(c.get("value", "")).strip()) and not c.get("is_header") and c.get("match_ratio", 0) >= 0.95 and c.get("bbox") and c.get("vision_token_indices") \
                   and (c["bbox"][2] - c["bbox"][0]) >= 12 and (c["bbox"][3] - c["bbox"][1]) >= 8 and 2 <= len(str(c.get("row_header", "")).strip()) <= 60 and 2 <= len(str(c.get("col_header", "")).strip()) <= 60:
                    bycol[(c["col_idx"], str(c["col_header"]))].append((pi, c))
        combos = []
        for (ci, ch), group in bycol.items():
            if len({pi for pi, _ in group}) < 2: continue
            rows = Counter(str(c["row_header"]).strip() for _, c in group); colvals = Counter(str(c["value"]).strip() for _, c in group)
            for i in range(len(group)):
                for j in range(len(group)):
                    if i == j: continue
                    (pa, A), (pb, B) = group[i], group[j]
                    if pa == pb: continue
                    combos.append((ch, pa, A, pb, B, rows, colvals))
        rng.shuffle(combos); n_tab = 0
        for ch, pa, A, pb, B, rows, colvals in combos:
            if n_tab >= args.cap_table or per_doc[doc] >= args.cap_doc: break
            av, bv = str(A["value"]).strip(), str(B["value"]).strip(); ar, br = str(A["row_header"]).strip(), str(B["row_header"]).strip()
            a_dup = (val_cnt[av] > 1) if args.a_unique == "table" else (colvals[av] > 1)
            if av == bv or ar == br or a_dup or val_cnt[bv] > 1 or rows[ar] > 1 or rows[br] > 1: stat["dup"] += 1; continue
            if (doc, tbl, ch, ar, br) in excl: stat["existing"] += 1; continue
            a_i, b_i = int(av), int(bv); use_sum = (len(cands) % 3 == 2); r_i = a_i + b_i if use_sum else abs(a_i - b_i)
            if r_i == 0: continue
            rv = str(r_i)
            if len({av[0], bv[0], rv[0]}) < 3: stat["firstdigit"] += 1; continue
            rit = rv in val_cnt
            if args.exclude_rit and rit: stat["rit"] += 1; continue
            pick = None
            for delta in (2, -2, 3, -3, 1, -1, 4, -4):
                nd = int(bv[0]) + delta
                if not (1 <= nd <= 9): continue
                bp = str(nd) + bv[1:]; bp_i = int(bp); rp_i = a_i + bp_i if use_sum else abs(a_i - bp_i); rp = str(rp_i)
                if rp_i == 0 or bp in val_cnt or rp[0] == rv[0] or bp[0] == av[0]: continue
                pick = (bp, rp); break
            if pick is None: stat["nodonor"] += 1; continue
            bp, rp = pick; opn = "sum" if use_sum else "difference"
            cq = (f"In the table, what is the {opn} {'of' if use_sum else 'between'} the values in {phrase_col(ch)} for the rows '{ar}' and '{br}'?")
            pg_list = [{"page_id": p["page_id"], "image": os.path.join(IMG, os.path.basename(p["image"])), "grid_hw": p["grid_hw"]} for p in pages]
            qid = f"{args.prefix}_{doc}_t{tbl}_{len(cands):04d}"
            cands.append({"qid": qid, "doc_id": doc, "table_index": tbl, "level": "synthetic_mp", "n_pages": len(pages), "pages": pg_list,
                          "a_page_pos": pa, "b_page_pos": pb, "op": opn, "n_digits": [len(av), len(bv)],
                          "lookup_a": {"question": f"In the table, what value appears in {phrase_col(ch)} for the row '{ar}'?", "gold": av},
                          "lookup_b": {"question": f"In the table, what value appears in {phrase_col(ch)} for the row '{br}'?", "gold": bv},
                          "lookup_b_question": f"In the table, what value appears in {phrase_col(ch)} for the row '{br}'?",
                          "compute_question": cq, "compute_gold": rv, "compute_gold_new": rp, "b_value": bv, "b_new": bp,
                          "operands": [A, B], "bbox": B["bbox"], "vision_token_indices": B["vision_token_indices"], "grid_hw": pages[pb]["grid_hw"],
                          "first_digits": {"A": av[0], "B": bv[0], "R": rv[0], "B'": bp[0], "R'": rp[0]}, "result_in_table": rit})
            n_tab += 1; per_doc[doc] += 1
    stat["before_target"] = len(cands); cands = cands[:args.target]
    if args.render:
        OUTD.mkdir(exist_ok=True)
        for c in cands:
            base_imgs, donor_imgs = [], []
            for pi, p in enumerate(c["pages"]):
                if pi == c["b_page_pos"]:
                    dp = OUTD / f"{c['qid']}__donor.png"
                    if not dp.exists():
                        img = Image.open(p["image"]).convert("RGB"); render_donor(img, c["bbox"], c["b_new"]).save(dp)
                    donor_imgs.append(str(dp)); base_imgs.append(p["image"])
                else: base_imgs.append(p["image"]); donor_imgs.append(p["image"])
            c["base_images"], c["donor_images"] = base_imgs, donor_imgs
    with open(args.out, "w") as f:
        for c in cands: f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(cands)} candidates ({len({c['doc_id'] for c in cands})} docs, {len({(c['doc_id'], c['table_index']) for c in cands})} tables, "
          f"pages {dict(Counter(c['n_pages'] for c in cands))}, op {dict(Counter(c['op'] for c in cands))}, digits(A,B) {dict(Counter(tuple(c['n_digits']) for c in cands))}, "
          f"result_in_table {sum(c['result_in_table'] for c in cands)}, per-table max {max(Counter((c['doc_id'], c['table_index']) for c in cands).values())}) stat {dict(stat)} -> {args.out}")
if __name__ == "__main__": main()
