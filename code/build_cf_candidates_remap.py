#!/usr/bin/env python3
"""Generate ~100 LOOKUP-A/LOOKUP-B/COMPUTE triplet candidates with
counterfactual donor images, from patch_cell_labels (test tables).

Per candidate: two integer cells A, B in the SAME column of one table page
-> LOOKUP-A / LOOKUP-B ask each cell's value, COMPUTE asks their
difference (or sum). Donor image: B's leading digit edited (B'), giving a
new compute gold R'. All questions use the same single page image.

Hard filters (lessons from the 6-item pilot):
  - integer values, 2-4 digits (single-digit B failed to flip: redundant/
    duplicated), A != B
  - B and B' unique on the WHOLE page word set; A unique in its column
  - first digits of A, B, R pairwise distinct (lens needs it; Qwen
    tokenizes numbers digit-by-digit) and R' first digit != R's
  - row headers unique within the column, 2-60 chars; col header 2-60
  - cell bbox >= 12x8 px, match_ratio >= 0.95
Soft (ordering): result R not present on the page.

Caps: 2 per table, 3 per doc, --target total.
Output: cf2_candidates.jsonl + donor/base PNGs + crops in cf2_images/.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
LABELS = Path("../data/external/qa_enrichment/patch_cell_labels/"
              "patch_cell_labels_test_qwen35.jsonl")
OUTD = BASE / "cf2_images"
OUT = BASE / "cf2_candidates.jsonl"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
INT_RE = re.compile(r"^\d{2,4}$")


def phrase_col(col):
    if " | " in str(col):
        parts = [p.strip() for p in str(col).split(" | ")]
        return f"the '{parts[-1]}' column under '{' / '.join(parts[:-1])}'"
    return f"the '{col}' column"


def render_donor(img, bbox, b_new):
    arr = np.asarray(img)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cell = arr[y1:y2, x1:x2]
    gray = cell.mean(axis=2)
    tm = gray <= np.percentile(gray, 5)
    text_color = tuple(int(v) for v in cell[tm].mean(axis=0)) if tm.any() \
        else (0, 0, 0)
    bm = gray > np.percentile(gray, 70)
    bg_color = tuple(int(v) for v in cell[bm].mean(axis=0)) if bm.any() \
        else (255, 255, 255)
    donor = img.copy()
    d = ImageDraw.Draw(donor)
    d.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], fill=bg_color)
    bw, bh = (x2 - x1), (y2 - y1)
    size, f = 6, ImageFont.truetype(FONT, 6)
    while size < 60:
        f2 = ImageFont.truetype(FONT, size + 1)
        bb = d.textbbox((0, 0), b_new, font=f2)
        if bb[2] - bb[0] > bw * 1.05 or bb[3] - bb[1] > bh * 1.05:
            break
        size, f = size + 1, f2
    bb = d.textbbox((0, 0), b_new, font=f)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    d.text((cx - (bb[2] - bb[0]) / 2 - bb[0], cy - (bb[3] - bb[1]) / 2 - bb[1]),
           b_new, font=f, fill=text_color)
    return donor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--render", action="store_true",
                    help="also write base/donor/crop images")
    ap.add_argument("--digits", type=int, default=0,
                    help="if set, require BOTH values to have exactly this "
                         "many digits (e.g. 2 - easier arithmetic)")
    ap.add_argument("--prefix", default="cf2")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--exclude", default="",
                    help="candidates .jsonl whose (page,col,rows) pairs to skip")
    ap.add_argument("--prefer-long", action="store_true",
                    help="reproduce batch-1 ordering (longer B first)")
    args = ap.parse_args()
    OUTD.mkdir(exist_ok=True)

    excl = set()
    if args.exclude:
        for line in open(args.exclude):
            c = json.loads(line)
            a_r = str(c["operands"][0]["row_header"]).strip()
            b_r = str(c["operands"][1]["row_header"]).strip()
            excl.add((c["pages"][0]["page_id"],
                      str(c["operands"][1]["col_header"]), a_r, b_r))

    cands = []
    per_doc = defaultdict(int)
    for line in open(LABELS):
        d = json.loads(line)
        doc, tbl = d["doc_id"], d["table_index"]
        if per_doc[doc] >= 3:
            continue
        for page in d["pages"]:
            cells = page.get("cells", [])
            page_vals = [str(c.get("value", "")).strip() for c in cells]
            # candidate integer cells
            ok = [c for c in cells
                  if INT_RE.match(str(c.get("value", "")).strip())
                  and not c.get("is_header")
                  and c.get("match_ratio", 0) >= 0.95
                  and c.get("bbox") and c.get("vision_token_indices")
                  and (c["bbox"][2] - c["bbox"][0]) >= 12
                  and (c["bbox"][3] - c["bbox"][1]) >= 8
                  and 2 <= len(str(c.get("row_header", "")).strip()) <= 60
                  and 2 <= len(str(c.get("col_header", "")).strip()) <= 60]
            bycol = defaultdict(list)
            for c in ok:
                bycol[(c["col_idx"], str(c["col_header"]))].append(c)
            n_from_table = 0
            for (ci, ch), group in bycol.items():
                if n_from_table >= 2 or per_doc[doc] >= 3:
                    break
                vals = [str(c["value"]).strip() for c in group]
                rows = [str(c["row_header"]).strip() for c in group]
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        if n_from_table >= 2 or per_doc[doc] >= 3:
                            break
                        A, B = group[i], group[j]
                        av, bv = str(A["value"]).strip(), str(B["value"]).strip()
                        ar, br = (str(A["row_header"]).strip(),
                                  str(B["row_header"]).strip())
                        if av == bv or ar == br:
                            continue
                        if args.digits and (len(av) != args.digits
                                            or len(bv) != args.digits):
                            continue
                        if (page["page_id"], ch, ar, br) in excl \
                                or (page["page_id"], ch, br, ar) in excl:
                            continue
                        if vals.count(av) > 1 or page_vals.count(bv) > 1:
                            continue
                        if rows.count(ar) > 1 or rows.count(br) > 1:
                            continue
                        a_i, b_i = int(av), int(bv)
                        use_sum = (len(cands) % 3 == 2)
                        r_i = a_i + b_i if use_sum else abs(a_i - b_i)
                        if r_i == 0:
                            continue
                        rv = str(r_i)
                        if len({av[0], bv[0], rv[0]}) < 3:
                            continue
                        # counterfactual: change B's leading digit
                        pick = None
                        for delta in (2, -2, 3, -3, 1, -1, 4, -4):
                            nd = int(bv[0]) + delta
                            if not (1 <= nd <= 9):
                                continue
                            bp = str(nd) + bv[1:]
                            bp_i = int(bp)
                            rp_i = (a_i + bp_i if use_sum
                                    else abs(a_i - bp_i))
                            rp = str(rp_i)
                            if rp_i == 0 or bp in page_vals:
                                continue
                            if rp[0] == rv[0] or bp[0] == av[0]:
                                continue
                            pick = (bp, rp)
                            break
                        if pick is None:
                            continue
                        bp, rp = pick
                        # lens control: another integer on page, distinct 1st
                        ctrl = next(
                            (v for v in page_vals
                             if INT_RE.match(v)
                             and v not in (av, bv, rv)
                             and v[0] not in (av[0], bv[0], rv[0])), None)
                        opn = "sum" if use_sum else "difference"
                        cq = (f"In the table, what is the {opn} "
                              f"{'of' if use_sum else 'between'} the values in "
                              f"{phrase_col(ch)} for the rows "
                              f"'{ar}' and '{br}'?")
                        qid = (f"{args.prefix}_{doc}_t{tbl}_"
                               f"p{page['page_num']}_{len(cands):03d}")
                        cands.append({
                            "qid": qid, "doc_id": doc, "table_index": tbl,
                            "level": "synthetic", "n_pages": 1,
                            "pages": [{"page_id": page["page_id"],
                                       "image": page["image"],
                                       "grid_hw": page["grid_hw"]}],
                            "op": opn,
                            "lookup_a": {"question":
                                         f"In the table, what value appears in "
                                         f"{phrase_col(ch)} for the row "
                                         f"'{ar}'?", "gold": av},
                            "lookup_b": {"question":
                                         f"In the table, what value appears in "
                                         f"{phrase_col(ch)} for the row "
                                         f"'{br}'?", "gold": bv},
                            "lookup_b_question":
                                f"In the table, what value appears in "
                                f"{phrase_col(ch)} for the row '{br}'?",
                            "compute_question": cq, "compute_gold": rv,
                            "compute_gold_new": rp,
                            "b_value": bv, "b_new": bp,
                            "operands": [A, B],
                            "bbox": B["bbox"],
                            "vision_token_indices": B["vision_token_indices"],
                            "grid_hw": page["grid_hw"],
                            "first_tokens": None,  # filled by lens runner? no:
                            "first_digits": {"A": av[0], "B": bv[0],
                                             "R": rv[0], "B'": bp[0],
                                             "R'": rp[0]},
                            "control_value": ctrl,
                            "result_in_table": rv in page_vals,
                        })
                        n_from_table += 1
                        per_doc[doc] += 1
    # prefer result not on page, then SHORTER values (easier arithmetic --
    # the 4-digit-first ordering of the first batch tanked base_compute)
    cands.sort(key=lambda c: (c["result_in_table"],
                              -len(c["b_value"]) if args.prefer_long
                              else len(c["b_value"])))
    cands = cands[:args.target]

    if args.render:
        for c in cands:
            import os
            c["pages"][0]["image"] = os.path.join("../data/external/pubtables-v2/Full Documents/test/images", os.path.basename(c["pages"][0]["image"]))
            img = Image.open(c["pages"][0]["image"]).convert("RGB")
            donor = render_donor(img, c["bbox"], c["b_new"])
            bpth = OUTD / f"{c['qid']}__base.png"
            dpth = OUTD / f"{c['qid']}__donor.png"
            img.save(bpth)
            donor.save(dpth)
            c["base_image"], c["donor_image"] = str(bpth), str(dpth)
            x1, y1, x2, y2 = [int(v) for v in c["bbox"]]
            m = 60
            box = (max(0, x1 - m), max(0, y1 - m),
                   min(img.width, x2 + m), min(img.height, y2 + m))
            cb, cd = img.crop(box), donor.crop(box)
            side = Image.new("RGB", (cb.width * 2 + 8, cb.height), (255, 0, 0))
            side.paste(cb, (0, 0))
            side.paste(cd, (cb.width + 8, 0))
            side.save(OUTD / f"{c['qid']}__crop.png")

    with open(args.out, "w") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    n_rit = sum(c["result_in_table"] for c in cands)
    n_docs = len({c["doc_id"] for c in cands})
    print(f"{len(cands)} candidates ({n_docs} docs, result_in_table {n_rit}) "
          f"-> {Path(args.out).name}"
          + (" (+images)" if args.render else ""))


if __name__ == "__main__":
    main()
