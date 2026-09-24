#!/usr/bin/env python3
"""Build per-cell (value, row_header, col_header) labels with vision-patch indices.

Pipeline (extends enrich_annotations.py + logit_lens_v4 patch mapping):
  1. For every cell of every table: align cell tokens to OCR words in reading
     order (monotonic sequence alignment) -> cell bbox in page-image pixels.
  2. Label each cell semantically: value, row_header (stub-column text of the
     row), col_header (from enriched header_map).
  3. Map each cell bbox to Qwen-VL vision-token patch indices using the same
     grid math as logit_lens_v4_cell_patches.pixel_to_patch_indices.

Output: one JSON line per table with per-page cell entries:
  {value, row_header, col_header, row_idx, col_idx, is_header, bbox,
   match_ratio, patches[[r,c]..], vision_token_indices[flat..]}

Usage:
    HF_HUB_OFFLINE=1 python build_patch_cell_labels.py \
        --enriched /path/to/enriched/enriched_test.jsonl \
        --root "/path/to/pubtables-v2/Full Documents/test" \
        --out-dir ../data/external/qa_enrichment/patch_cell_labels \
        [--limit N]
"""

import argparse
import json
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich_annotations import clean_text, normalize_for_match  # noqa: E402


# ---------------------------------------------------------------------------
# Token grid (mirrors logit_lens_v4_cell_patches.py exactly)
# ---------------------------------------------------------------------------

class TokenGrid:
    def __init__(self, grid_h, grid_w, token_pixel, resized_h, resized_w):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.token_pixel = token_pixel
        self.resized_h = resized_h
        self.resized_w = resized_w


def build_token_grid(image_grid_thw, merge_size=2, token_pixel=28):
    t, h_patches, w_patches = [int(x) for x in image_grid_thw]
    assert t == 1, "static image expected"
    return TokenGrid(
        grid_h=h_patches // merge_size,
        grid_w=w_patches // merge_size,
        token_pixel=token_pixel,
        resized_h=h_patches * token_pixel,
        resized_w=w_patches * token_pixel,
    )


def pixel_to_patch_indices(cell_bbox_orig, orig_w, orig_h, grid):
    """Same mapping as logit_lens_v4_cell_patches.pixel_to_patch_indices."""
    x1, y1, x2, y2 = cell_bbox_orig
    scale_x = grid.resized_w / orig_w
    scale_y = grid.resized_h / orig_h

    rx1 = x1 * scale_x
    ry1 = y1 * scale_y
    rx2 = x2 * scale_x
    ry2 = y2 * scale_y

    merge_px = grid.token_pixel * 2

    col_start = max(0, int(rx1 // merge_px))
    col_end = min(grid.grid_w - 1, int((rx2 - 1) // merge_px))
    row_start = max(0, int(ry1 // merge_px))
    row_end = min(grid.grid_h - 1, int((ry2 - 1) // merge_px))

    patches = []
    for r in range(row_start, row_end + 1):
        for c in range(col_start, col_end + 1):
            patches.append((r, c))
    return patches


# ---------------------------------------------------------------------------
# Word loading / ordering
# ---------------------------------------------------------------------------

def build_file_index(root: Path, pattern: str):
    index = {}
    for p in root.rglob(pattern):
        index[p.name] = p
    return index


def load_words_in_bbox(words_path, bbox, margin_x=20, margin_y=15):
    words = json.loads(Path(words_path).read_text())
    x1, y1, x2, y2 = bbox
    selected = []
    for w in words:
        wb = w.get("bbox")
        if not wb or len(wb) != 4:
            continue
        cx = (wb[0] + wb[2]) / 2
        cy = (wb[1] + wb[3]) / 2
        if (x1 - margin_x) <= cx <= (x2 + margin_x) and (y1 - margin_y) <= cy <= (y2 + margin_y):
            selected.append(w)
    selected.sort(key=lambda w: (w.get("block_num", 0), w.get("line_num", 0),
                                 w.get("span_num", 0)))
    return selected


# ---------------------------------------------------------------------------
# Cell -> word alignment (monotonic, handles duplicate values)
# ---------------------------------------------------------------------------

def align_cells_to_words(page_cells, words):
    """page_cells: list of dicts with 'tokens' (normalized). words: reading order.
    Returns {cell_pos: [word_idx, ...]} via SequenceMatcher on token sequences."""
    cell_tokens = []
    token_owner = []
    for pos, cell in enumerate(page_cells):
        for tok in cell["tokens"]:
            cell_tokens.append(tok)
            token_owner.append(pos)
    word_tokens = [normalize_for_match(w.get("text", "")) for w in words]

    matcher = SequenceMatcher(None, cell_tokens, word_tokens, autojunk=False)
    assigned = defaultdict(list)
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            assigned[token_owner[a + k]].append(b + k)
    return assigned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_row_headers(cells, n_rows):
    """row -> text of the stub (column 0) cell covering that row."""
    row_header = {}
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        if 0 not in (cell.get("column_nums") or []):
            continue
        text = clean_text(cell.get("xml_text_content") or cell.get("xml_raw_text_content"))
        if not text:
            continue
        for r in cell.get("row_nums", []):
            row_header.setdefault(r, text)
    return row_header


def process_table(rec, table, words_index, image_index, processor_cache):
    doc_id = rec["doc_id"]
    cells = [c for c in table.get("cells", []) if isinstance(c, dict)]
    parts = table.get("parts", [])
    row_to_page = {int(r): p for r, p in rec["row_to_page"].items()}
    header_map = rec.get("header_map", {})
    row_headers = build_row_headers(cells, rec["n_rows"])

    part_bbox_by_page = {}
    for part in parts:
        pn = part.get("page_num")
        if pn is not None and pn not in part_bbox_by_page and part.get("bbox"):
            part_bbox_by_page[pn] = part["bbox"]

    # Assign each cell to the page of its first mapped row
    cells_by_page = defaultdict(list)
    n_cells_total = 0
    for cell in cells:
        row_nums = cell.get("row_nums") or []
        col_nums = cell.get("column_nums") or []
        if not row_nums or not col_nums:
            continue
        value = clean_text(cell.get("xml_text_content") or cell.get("xml_raw_text_content"))
        if not value:
            continue
        n_cells_total += 1
        page = None
        for r in row_nums:
            if r in row_to_page:
                page = row_to_page[r]
                break
        if page is None or page not in part_bbox_by_page:
            continue
        tokens = [normalize_for_match(t) for t in value.split()]
        tokens = [t for t in tokens if t]
        if not tokens:
            continue
        r0, c0 = min(row_nums), min(col_nums)
        cells_by_page[page].append({
            "row_idx": r0, "col_idx": c0,
            "row_nums": row_nums, "column_nums": col_nums,
            "value": value, "tokens": tokens,
            "is_header": bool(cell.get("is_column_header")),
        })

    pages_out = []
    n_matched = 0
    for page, page_cells in sorted(cells_by_page.items()):
        page_id = f"{doc_id}_page_{page}"
        words_path = words_index.get(f"{page_id}_words.json")
        image_path = image_index.get(f"{page_id}.jpg")
        if not words_path or not image_path:
            continue
        # reading order: row-major, then column
        page_cells.sort(key=lambda c: (c["row_idx"], c["col_idx"]))
        words = load_words_in_bbox(words_path, part_bbox_by_page[page])
        assigned = align_cells_to_words(page_cells, words)

        image = Image.open(image_path)
        orig_w, orig_h = image.size
        grid = processor_cache.get_grid(image_path, image)

        out_cells = []
        for pos, cell in enumerate(page_cells):
            word_idxs = assigned.get(pos, [])
            entry = {
                "value": cell["value"],
                "row_header": row_headers.get(cell["row_idx"], ""),
                "col_header": header_map.get(str(cell["col_idx"]), ""),
                "row_idx": cell["row_idx"],
                "col_idx": cell["col_idx"],
                "is_header": cell["is_header"],
                "bbox": None,
                "match_ratio": round(len(word_idxs) / len(cell["tokens"]), 3),
                "patches": [],
                "vision_token_indices": [],
            }
            if word_idxs:
                boxes = [words[i]["bbox"] for i in word_idxs]
                bbox = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                        max(b[2] for b in boxes), max(b[3] for b in boxes)]
                patches = pixel_to_patch_indices(bbox, orig_w, orig_h, grid)
                entry["bbox"] = [round(v, 2) for v in bbox]
                entry["patches"] = patches
                entry["vision_token_indices"] = [r * grid.grid_w + c for r, c in patches]
                n_matched += 1
            out_cells.append(entry)

        pages_out.append({
            "page_num": page,
            "page_id": page_id,
            "image": str(image_path),
            "orig_size": [orig_w, orig_h],
            "grid_hw": [grid.grid_h, grid.grid_w],
            "cells": out_cells,
        })

    return {
        "doc_id": doc_id,
        "table_index": rec["table_index"],
        "n_rows": rec["n_rows"],
        "n_cols": rec["n_cols"],
        "caption_info": rec.get("caption_info", {}),
        "pages": pages_out,
        "stats": {
            "n_cells_nonempty": n_cells_total,
            "n_cells_with_bbox": n_matched,
        },
    }


class ProcessorGridCache:
    """Compute Qwen-VL image_grid_thw per page image, cached by path."""

    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
        from transformers import AutoProcessor
        self.ip = AutoProcessor.from_pretrained(model_id).image_processor
        # merged token pixel = patch_size * merge_size (Qwen2.5-VL: 14*2=28,
        # Qwen3.5: 16*2=32). image_grid_thw counts raw patches.
        self.merge_size = int(getattr(self.ip, "merge_size", None)
                              or getattr(self.ip, "spatial_merge_size", 2))
        # pixel_to_patch_indices' merge_px = token_pixel*2 assumes merge_size=2
        assert self.merge_size == 2, f"merge_size={self.merge_size} unsupported"
        patch = int(getattr(self.ip, "patch_size", 14))
        self.token_pixel = patch * self.merge_size
        print(f"grid geometry: patch={patch} merge={self.merge_size} "
              f"token_pixel={self.token_pixel}", flush=True)
        self.cache = {}

    def get_grid(self, image_path, image):
        key = str(image_path)
        if key not in self.cache:
            out = self.ip(images=[image.convert("RGB")], return_tensors="pt")
            self.cache[key] = build_token_grid(out["image_grid_thw"][0],
                                               merge_size=self.merge_size,
                                               token_pixel=self.token_pixel)
        return self.cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out-name", default="patch_cell_labels_test.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--doc-filter", default="")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Indexing files...", flush=True)
    tables_index = build_file_index(root / "tables", "*_tables.json")
    words_index = build_file_index(root / "words", "*_words.json")
    image_index = build_file_index(root / "images", "*.jpg")
    print(f"  tables={len(tables_index)} words={len(words_index)} images={len(image_index)}",
          flush=True)

    processor_cache = ProcessorGridCache(args.model_id)

    records = []
    with open(args.enriched) as f:
        for line in f:
            records.append(json.loads(line))
    if args.doc_filter:
        records = [r for r in records if args.doc_filter in r["doc_id"]]
    if args.limit:
        records = records[:args.limit]

    out_path = out_dir / args.out_name
    tables_cache = {}
    totals = {"tables": 0, "cells": 0, "with_bbox": 0, "full_match": 0}
    with open(out_path, "w") as fout:
        for i, rec in enumerate(records):
            doc_id = rec["doc_id"]
            tpath = tables_index.get(f"{doc_id}_tables.json")
            if not tpath:
                continue
            if doc_id not in tables_cache:
                tables_cache = {doc_id: json.loads(tpath.read_text())}  # keep one doc
            doc_tables = tables_cache[doc_id]
            tidx = rec["table_index"] - 1
            if tidx >= len(doc_tables):
                continue
            result = process_table(rec, doc_tables[tidx], words_index, image_index,
                                   processor_cache)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            totals["tables"] += 1
            totals["cells"] += result["stats"]["n_cells_nonempty"]
            totals["with_bbox"] += result["stats"]["n_cells_with_bbox"]
            for pg in result["pages"]:
                totals["full_match"] += sum(1 for c in pg["cells"]
                                            if c["match_ratio"] >= 1.0)
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(records)}] tables={totals['tables']} "
                      f"cells={totals['cells']} bbox={totals['with_bbox']}", flush=True)

    cov = totals["with_bbox"] / max(1, totals["cells"])
    print(f"\nDone. tables={totals['tables']} cells={totals['cells']} "
          f"with_bbox={totals['with_bbox']} ({cov:.1%}) "
          f"full_token_match={totals['full_match']}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
