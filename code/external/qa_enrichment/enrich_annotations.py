#!/usr/bin/env python3
"""Enrich PubTables-v2 annotations with row→page mapping and caption parsing.

Reads table JSON + per-page OCR words + original_markup to produce:
  1. row_to_page: which page each row belongs to
  2. cell_bbox: approximate bounding box for each cell (from OCR word alignment)
  3. caption_info: structured caption, label, footnotes, abbreviations

Usage:
    python enrich_annotations.py \
        --root /path/to/pubtables-v2 \
        --split test \
        --out-dir /path/to/enriched
"""

import argparse
import html
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_text(text: Any) -> str:
    text = str(text or "")
    text = html.unescape(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


_LIGATURE_MAP = {
    'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff',
    'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬅ': 'st', 'ﬆ': 'st',
}
_DASH_RE = re.compile(r'[‐-―⁃−﹘﹣－]')
_SQUOTE_RE = re.compile(r'[‘’‚‛ʼ]')
_DQUOTE_RE = re.compile(r'[“”„‟]')
_ZEROWIDTH_RE = re.compile(r'[​-‍⁠﻿\xad]')


def normalize_for_match(text: str) -> str:
    text = clean_text(text).lower()
    for lig, repl in _LIGATURE_MAP.items():
        text = text.replace(lig, repl)
    text = _DASH_RE.sub('-', text)
    text = _SQUOTE_RE.sub("'", text)
    text = _DQUOTE_RE.sub('"', text)
    text = _ZEROWIDTH_RE.sub('', text)
    text = text.replace('\xa0', ' ')
    text = re.sub(r"[^\w\s.,;:%()+\-/]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def build_file_index(root: Path, pattern: str) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not root.exists():
        return index
    for path in sorted(root.rglob(pattern)):
        if path.is_file():
            index.setdefault(path.name, path)
    return index


def load_json_file(path: Optional[Path]) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_words(words_index: Dict[str, Path], page_id: str) -> List[Dict[str, Any]]:
    data = load_json_file(words_index.get(f"{page_id}_words.json"))
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# Row → Page mapper
# ---------------------------------------------------------------------------

def words_in_bbox(words: List[Dict[str, Any]], bbox: List[float],
                  margin_x: float = 15.0, margin_y: float = 10.0) -> List[Dict[str, Any]]:
    if len(bbox) != 4:
        return []
    x1, y1, x2, y2 = bbox
    result = []
    for w in words:
        wb = w.get("bbox")
        if not isinstance(wb, list) or len(wb) != 4:
            continue
        cx = (wb[0] + wb[2]) / 2.0
        cy = (wb[1] + wb[3]) / 2.0
        if (x1 - margin_x <= cx <= x2 + margin_x and
                y1 - margin_y <= cy <= y2 + margin_y):
            result.append(w)
    return result


def page_table_text(words: List[Dict[str, Any]], bbox: List[float]) -> str:
    selected = words_in_bbox(words, bbox)
    tokens = [clean_text(w.get("text", "")) for w in selected if w.get("text")]
    return " ".join(tokens)


def build_row_cell_signatures(
    cells: List[Dict[str, Any]], n_rows: int, n_cols: int
) -> Dict[int, List[str]]:
    """For each row, collect unique cell text signatures for matching."""
    row_sigs: Dict[int, List[str]] = defaultdict(list)
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        row_nums = cell.get("row_nums", [])
        text = clean_text(cell.get("xml_text_content") or cell.get("xml_raw_text_content"))
        if not text or len(text) < 2:
            continue
        for r in row_nums:
            row_sigs[r].append(normalize_for_match(text))
    return dict(row_sigs)


def _token_sim(a: str, b: str) -> float:
    """Token-level similarity via SequenceMatcher."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    if abs(la - lb) > max(2, max(la, lb) * 0.4):
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _group_cells_by_row(
    cells: List[Dict[str, Any]], n_rows: int
) -> Dict[int, List[Dict[str, Any]]]:
    """Group cell dicts by row number."""
    row_cells: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        for r in cell.get("row_nums", []):
            row_cells[r].append(cell)
    return dict(row_cells)


def score_row_on_page(
    row_sigs: List[str], page_text_norm: str,
    page_tokens: Optional[List[str]] = None,
) -> Tuple[float, float]:
    """Score how many of a row's cell texts appear in the page's table OCR text.

    Returns (exact_score, fuzzy_score) where fuzzy includes exact matches.
    """
    if not row_sigs:
        return 0.0, 0.0
    sorted_sigs = sorted(row_sigs, key=len, reverse=True)
    exact_found = 0
    fuzzy_found = 0
    for sig in sorted_sigs:
        if len(sig) < 2:
            continue
        if sig in page_text_norm:
            exact_found += 1
            fuzzy_found += 1
            continue
        # Fuzzy token-level matching
        if page_tokens is None:
            continue
        sig_tokens = sig.split()
        if not sig_tokens:
            continue
        matched = 0.0
        for st in sig_tokens:
            if len(st) < 2:
                matched += 0.5
                continue
            best = 0.0
            for pt in page_tokens:
                if abs(len(st) - len(pt)) > max(2, len(st) * 0.4):
                    continue
                if len(st) >= 3 and len(pt) >= 3 and st[0] != pt[0]:
                    continue
                sim = _token_sim(st, pt)
                if sim > best:
                    best = sim
                    if best >= 0.95:
                        break
            if best >= 0.75:
                matched += 1
            elif best >= 0.55:
                matched += 0.5
        ratio = matched / len(sig_tokens)
        if ratio >= 0.6:
            fuzzy_found += 1
    total = len(sorted_sigs)
    return exact_found / total, fuzzy_found / total


def score_row_words(
    row_cells: List[Dict[str, Any]],
    page_table_words: List[Dict[str, Any]],
) -> float:
    """Word-level matching: match individual cell words against OCR words.

    More robust than text-in-text matching — handles tokenization differences
    and works even when OCR text joining produces different substrings.
    """
    if not row_cells or not page_table_words:
        return 0.0

    ocr_words_norm = []
    for w in page_table_words:
        t = normalize_for_match(w.get("text", ""))
        if t:
            ocr_words_norm.append(t)
    if not ocr_words_norm:
        return 0.0

    total_tokens = 0
    matched_tokens = 0
    for cell in row_cells:
        text = clean_text(cell.get("xml_text_content") or cell.get("xml_raw_text_content"))
        if not text or len(text) < 2:
            continue
        for token in normalize_for_match(text).split():
            if len(token) < 2:
                continue
            total_tokens += 1
            if token in ocr_words_norm:
                matched_tokens += 1
                continue
            best = 0.0
            for ow in ocr_words_norm:
                if abs(len(token) - len(ow)) > max(2, len(token) * 0.4):
                    continue
                sim = _token_sim(token, ow)
                if sim > best:
                    best = sim
                    if best >= 0.95:
                        break
            if best >= 0.75:
                matched_tokens += 1
    return matched_tokens / total_tokens if total_tokens > 0 else 0.0


def _row_page_evidence(
    row_cells: List[Dict[str, Any]],
    page_table_words: List[Dict[str, Any]],
    table_bbox: List[float],
) -> Tuple[float, Optional[float]]:
    """Word-match ratio and avg relative Y-position for a row on a page.

    Returns (match_ratio, y_rel) where y_rel is 0.0=top, 1.0=bottom of table bbox.
    """
    if not row_cells or not page_table_words or len(table_bbox) != 4:
        return 0.0, None
    ocr_data = [(normalize_for_match(w.get("text", "")), w.get("bbox", []))
                for w in page_table_words]
    ocr_data = [(t, b) for t, b in ocr_data if t and len(b) == 4]
    if not ocr_data:
        return 0.0, None

    bbox_top, bbox_bot = table_bbox[1], table_bbox[3]
    bbox_h = max(bbox_bot - bbox_top, 1.0)
    total = 0
    matched = 0
    ys: List[float] = []

    for cell in row_cells:
        text = clean_text(cell.get("xml_text_content") or cell.get("xml_raw_text_content"))
        if not text or len(text) < 2:
            continue
        for token in normalize_for_match(text).split():
            if len(token) < 2:
                continue
            total += 1
            for ot, ob in ocr_data:
                if ot == token or (len(token) >= 4 and _token_sim(token, ot) >= 0.8):
                    matched += 1
                    ys.append(((ob[1] + ob[3]) / 2.0 - bbox_top) / bbox_h)
                    break
    ratio = matched / total if total > 0 else 0.0
    y_rel = sum(ys) / len(ys) if ys else None
    return ratio, y_rel


def map_rows_to_pages(
    cells: List[Dict[str, Any]],
    parts: List[Dict[str, Any]],
    words_index: Dict[str, Path],
    doc_id: str,
    n_rows: int,
    n_cols: int,
) -> Dict[int, int]:
    """Map each row number to its page_num using multi-signal OCR cross-reference.

    Strategy:
    1. Build per-page data (OCR text + tokens + word objects)
    2. Score rows with exact text matching — assign clear winners
    3. Fuzzy token matching for ambiguous rows
    4. Word-level matching for still-ambiguous rows
    5. Unique-signature fallback
    6. Evidence-based interpolation for remaining gaps
    """
    # Phase 0: build per-page data
    page_data: Dict[int, Dict[str, Any]] = {}
    page_nums_ordered: List[int] = []
    for part in parts:
        pn = part.get("page_num")
        if pn is None:
            continue
        page_id = f"{doc_id}_page_{pn}"
        raw_words = load_words(words_index, page_id)
        bbox = part.get("bbox") or part.get("pdf_bbox") or []
        text_norm = normalize_for_match(page_table_text(raw_words, bbox))
        table_words = words_in_bbox(raw_words, bbox, margin_x=15, margin_y=10)
        page_data[pn] = {
            "text_norm": text_norm,
            "tokens": text_norm.split(),
            "table_words": table_words,
            "bbox": bbox,
        }
        if pn not in page_nums_ordered:
            page_nums_ordered.append(pn)

    if not page_data or len(page_nums_ordered) < 2:
        if page_nums_ordered:
            return {r: page_nums_ordered[0] for r in range(n_rows)}
        return {}

    row_sigs = build_row_cell_signatures(cells, n_rows, n_cols)
    row_cells_grouped = _group_cells_by_row(cells, n_rows)

    # Phase 1: score all rows — exact + fuzzy
    row_exact: Dict[int, Dict[int, float]] = {}
    row_fuzzy: Dict[int, Dict[int, float]] = {}
    for r in range(n_rows):
        sigs = row_sigs.get(r, [])
        ex: Dict[int, float] = {}
        fz: Dict[int, float] = {}
        for pn in page_nums_ordered:
            pd = page_data[pn]
            e, f = score_row_on_page(sigs, pd["text_norm"], pd["tokens"])
            ex[pn] = e
            fz[pn] = f
        row_exact[r] = ex
        row_fuzzy[r] = fz

    # Phase 2: assign rows with clear exact winner
    row_page: Dict[int, int] = {}
    needs_fuzzy: List[int] = []
    EXACT_THRESHOLD = 0.3
    EXACT_GAP = 0.15

    for r in range(n_rows):
        scores = row_exact[r]
        if not scores:
            needs_fuzzy.append(r)
            continue
        sorted_pages = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_page, best_score = sorted_pages[0]
        if best_score < EXACT_THRESHOLD:
            needs_fuzzy.append(r)
            continue
        if len(sorted_pages) >= 2:
            if best_score - sorted_pages[1][1] < EXACT_GAP:
                needs_fuzzy.append(r)
                continue
        row_page[r] = best_page

    # Phase 3: fuzzy text matching for unresolved rows
    needs_words: List[int] = []
    FUZZY_THRESHOLD = 0.25
    FUZZY_GAP = 0.1

    for r in needs_fuzzy:
        scores = row_fuzzy[r]
        if not scores:
            needs_words.append(r)
            continue
        sorted_pages = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_page, best_score = sorted_pages[0]
        if best_score < FUZZY_THRESHOLD:
            needs_words.append(r)
            continue
        if len(sorted_pages) >= 2:
            if best_score - sorted_pages[1][1] < FUZZY_GAP:
                needs_words.append(r)
                continue
        row_page[r] = best_page

    # Phase 4: word-level matching for still-ambiguous rows
    needs_unique: List[int] = []
    for r in needs_words:
        cells_for_row = row_cells_grouped.get(r, [])
        if not cells_for_row:
            needs_unique.append(r)
            continue
        word_scores: Dict[int, float] = {}
        for pn in page_nums_ordered:
            word_scores[pn] = score_row_words(cells_for_row, page_data[pn]["table_words"])
        sorted_pages = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        best_page, best_score = sorted_pages[0]
        if best_score <= 0:
            needs_unique.append(r)
            continue
        if len(sorted_pages) >= 2 and sorted_pages[1][1] >= best_score:
            needs_unique.append(r)
            continue
        row_page[r] = best_page

    # Phase 5: unique-signature fallback
    page_texts = {pn: page_data[pn]["text_norm"] for pn in page_nums_ordered}
    for r in needs_unique:
        unique = _try_unique_match(row_sigs.get(r, []), page_texts, page_nums_ordered)
        if unique is not None:
            row_page[r] = unique

    # Phase 6: evidence-based interpolation for remaining gaps
    combined_evidence: Dict[int, Dict[int, float]] = {}
    for r in range(n_rows):
        combined: Dict[int, float] = {}
        for pn in page_nums_ordered:
            exact = row_exact.get(r, {}).get(pn, 0)
            fuzzy = row_fuzzy.get(r, {}).get(pn, 0)
            combined[pn] = max(exact, fuzzy * 0.8)
        combined_evidence[r] = combined

    row_page = _interpolate_monotonic(row_page, n_rows, page_nums_ordered, combined_evidence)
    return row_page


def _try_unique_match(
    sigs: List[str],
    page_texts: Dict[int, str],
    page_nums: List[int],
) -> Optional[int]:
    """Try to find a signature that appears on exactly one page (exact or fuzzy)."""
    # Exact substring check
    for sig in sorted(sigs, key=len, reverse=True):
        if len(sig) < 4:
            continue
        found_on = [pn for pn in page_nums if sig in page_texts[pn]]
        if len(found_on) == 1:
            return found_on[0]
    # Fuzzy token-level check for longer signatures
    for sig in sorted(sigs, key=len, reverse=True):
        if len(sig) < 6:
            continue
        sig_tokens = sig.split()
        if len(sig_tokens) < 2:
            continue
        page_match: Dict[int, float] = {}
        for pn in page_nums:
            pt = page_texts[pn].split()
            matched = 0
            for st in sig_tokens:
                if len(st) < 3:
                    continue
                best = max((_token_sim(st, p) for p in pt
                            if abs(len(st) - len(p)) <= max(2, len(st) * 0.4)),
                           default=0.0)
                if best >= 0.8:
                    matched += 1
            page_match[pn] = matched / len(sig_tokens) if sig_tokens else 0
        sorted_pages = sorted(page_match.items(), key=lambda x: x[1], reverse=True)
        if (len(sorted_pages) >= 2
                and sorted_pages[0][1] >= 0.6
                and sorted_pages[1][1] < 0.3):
            return sorted_pages[0][0]
    return None


def _interpolate_monotonic(
    row_page: Dict[int, int], n_rows: int, page_nums: List[int],
    evidence: Optional[Dict[int, Dict[int, float]]] = None,
) -> Dict[int, int]:
    """Fill gaps in row→page mapping using evidence-based interpolation."""
    if not row_page or not page_nums:
        return row_page

    result: Dict[int, int] = {}
    anchors: List[Tuple[int, int]] = sorted(row_page.items())
    if not anchors:
        return {r: page_nums[0] for r in range(n_rows)}

    # Fill before first anchor
    first_row, first_page = anchors[0]
    for r in range(0, first_row):
        result[r] = first_page

    # Fill between anchors
    for i in range(len(anchors)):
        r_cur, p_cur = anchors[i]
        result[r_cur] = p_cur
        if i + 1 < len(anchors):
            r_next, p_next = anchors[i + 1]
            if p_cur == p_next:
                for r in range(r_cur + 1, r_next):
                    result[r] = p_cur
            else:
                # Different pages — find optimal transition using evidence
                best_t = r_cur
                if evidence:
                    best_ev = -1.0
                    for t in range(r_cur, r_next):
                        ev = 0.0
                        for ri in range(r_cur + 1, r_next):
                            scores = evidence.get(ri, {})
                            ev += scores.get(p_cur if ri <= t else p_next, 0)
                        if ev > best_ev:
                            best_ev = ev
                            best_t = t
                    if best_ev <= 0:
                        best_t = (r_cur + r_next) // 2
                else:
                    best_t = (r_cur + r_next) // 2
                for r in range(r_cur + 1, r_next):
                    result[r] = p_cur if r <= best_t else p_next

    # Fill after last anchor
    last_row, last_page = anchors[-1]
    result[last_row] = last_page
    for r in range(last_row + 1, n_rows):
        result[r] = last_page

    # Enforce monotonicity
    prev_page_idx = 0
    for r in range(n_rows):
        if r not in result:
            result[r] = page_nums[prev_page_idx]
            continue
        p = result[r]
        if p in page_nums:
            idx = page_nums.index(p)
            if idx < prev_page_idx:
                result[r] = page_nums[prev_page_idx]
            else:
                prev_page_idx = idx

    return result


def _verify_boundaries(
    row_page: Dict[int, int],
    row_cells_grouped: Dict[int, List[Dict[str, Any]]],
    page_data: Dict[int, Dict[str, Any]],
    page_nums: List[int],
    n_rows: int,
) -> Dict[int, int]:
    """Verify and fix off-by-one boundary errors using word-level + Y-coordinate evidence.

    Conservative: only shifts when there is strong match-ratio gap (>0.2) between
    the two candidate pages, and the forward/backward scan stops aggressively.
    """
    result = dict(row_page)

    boundaries: List[Tuple[int, int, int]] = []
    for r in range(n_rows - 1):
        if r in result and r + 1 in result and result[r] != result[r + 1]:
            boundaries.append((r, result[r], result[r + 1]))

    EVIDENCE_GAP = 0.2
    max_shift = min(10, max(n_rows // 5, 3))

    for bnd_last, page_a, page_b in boundaries:
        if page_a not in page_data or page_b not in page_data:
            continue
        bnd_first = bnd_last + 1

        # --- Check if "first row on page_b" is actually on page_a ---
        cells = row_cells_grouped.get(bnd_first, [])
        m_a, y_a = _row_page_evidence(
            cells, page_data[page_a]["table_words"], page_data[page_a]["bbox"])
        m_b, y_b = _row_page_evidence(
            cells, page_data[page_b]["table_words"], page_data[page_b]["bbox"])

        shift_down = False
        if m_a > m_b + EVIDENCE_GAP:
            shift_down = True
        elif m_a > 0.3 and m_b > 0.3 and y_a is not None:
            if y_a > 0.6 and (y_b is None or y_b < 0.15):
                shift_down = True

        if shift_down:
            result[bnd_first] = page_a
            for offset in range(1, max_shift):
                r_scan = bnd_first + offset
                if r_scan >= n_rows:
                    break
                sc = row_cells_grouped.get(r_scan, [])
                sa, _ = _row_page_evidence(
                    sc, page_data[page_a]["table_words"], page_data[page_a]["bbox"])
                sb, _ = _row_page_evidence(
                    sc, page_data[page_b]["table_words"], page_data[page_b]["bbox"])
                if sb > sa or sa <= 0.1:
                    break
                result[r_scan] = page_a
            continue

        # --- Check if "last row on page_a" is actually on page_b ---
        cells_last = row_cells_grouped.get(bnd_last, [])
        ml_a, yl_a = _row_page_evidence(
            cells_last, page_data[page_a]["table_words"], page_data[page_a]["bbox"])
        ml_b, yl_b = _row_page_evidence(
            cells_last, page_data[page_b]["table_words"], page_data[page_b]["bbox"])

        shift_up = False
        if ml_b > ml_a + EVIDENCE_GAP:
            shift_up = True
        elif ml_b > 0.3 and ml_a > 0.3 and yl_b is not None:
            if yl_b < 0.4 and (yl_a is None or yl_a > 0.85):
                shift_up = True

        if shift_up:
            result[bnd_last] = page_b
            for offset in range(1, max_shift):
                r_scan = bnd_last - offset
                if r_scan < 0:
                    break
                sc = row_cells_grouped.get(r_scan, [])
                sa, _ = _row_page_evidence(
                    sc, page_data[page_a]["table_words"], page_data[page_a]["bbox"])
                sb, _ = _row_page_evidence(
                    sc, page_data[page_b]["table_words"], page_data[page_b]["bbox"])
                if sa > sb or sb <= 0.1:
                    break
                result[r_scan] = page_b

    # Re-enforce monotonicity after corrections
    prev_idx = 0
    for r in range(n_rows):
        if r not in result:
            continue
        p = result[r]
        if p in page_nums:
            idx = page_nums.index(p)
            if idx < prev_idx:
                result[r] = page_nums[prev_idx]
            else:
                prev_idx = idx

    return result


# ---------------------------------------------------------------------------
# Cell → bbox mapper (approximate, from OCR word alignment)
# ---------------------------------------------------------------------------

def locate_cell_bbox(
    cell_text: str,
    page_words: List[Dict[str, Any]],
    table_bbox: List[float],
) -> Optional[List[float]]:
    """Find approximate bbox for a cell by matching its tokens to OCR words."""
    tokens = clean_text(cell_text).split()
    if not tokens:
        return None
    table_words = words_in_bbox(page_words, table_bbox, margin_x=20, margin_y=15)
    if not table_words:
        return None

    matched_bboxes = []
    used_indices = set()
    for token in tokens:
        token_lower = token.lower()
        best_idx = None
        for i, w in enumerate(table_words):
            if i in used_indices:
                continue
            if clean_text(w.get("text", "")).lower() == token_lower:
                best_idx = i
                break
        if best_idx is not None:
            matched_bboxes.append(table_words[best_idx]["bbox"])
            used_indices.add(best_idx)

    if not matched_bboxes:
        return None

    x1 = min(b[0] for b in matched_bboxes)
    y1 = min(b[1] for b in matched_bboxes)
    x2 = max(b[2] for b in matched_bboxes)
    y2 = max(b[3] for b in matched_bboxes)
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Caption / Footnote parser
# ---------------------------------------------------------------------------

class TableWrapParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._stack: List[str] = []
        self._texts: Dict[str, List[str]] = defaultdict(list)
        self._current_tag: str = ""

    def handle_starttag(self, tag, attrs):
        self._stack.append(tag)
        self._current_tag = tag

    def handle_endtag(self, tag):
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()
        self._current_tag = self._stack[-1] if self._stack else ""

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        for tag in reversed(self._stack):
            if tag in ("label", "title", "caption", "p", "fn", "table-wrap-foot"):
                self._texts[tag].append(text)
                break


def parse_caption_info(markup: str) -> Dict[str, Any]:
    """Extract structured caption info from original_markup XML/HTML."""
    if not markup:
        return {}

    result: Dict[str, Any] = {}

    # Table label (e.g., "Table 1")
    label_match = re.search(r"<label>(.*?)</label>", markup, re.DOTALL)
    if label_match:
        result["table_label"] = clean_text(label_match.group(1))

    # Caption title
    title_match = re.search(r"<title>(.*?)</title>", markup, re.DOTALL)
    if title_match:
        result["caption_title"] = clean_text(re.sub(r"<[^>]+>", "", title_match.group(1)))

    # Caption paragraphs (often contain abbreviations or footnotes)
    caption_match = re.search(r"<caption>(.*?)</caption>", markup, re.DOTALL)
    if caption_match:
        caption_body = caption_match.group(1)
        # Remove <title> from caption body (already extracted)
        caption_body = re.sub(r"<title>.*?</title>", "", caption_body, flags=re.DOTALL)
        paragraphs = re.findall(r"<p>(.*?)</p>", caption_body, re.DOTALL)
        notes = [clean_text(re.sub(r"<[^>]+>", "", p)) for p in paragraphs]
        notes = [n for n in notes if n]
        if notes:
            result["caption_notes"] = notes

    # Table-wrap-foot (footnotes)
    foot_match = re.search(r"<table-wrap-foot>(.*?)</table-wrap-foot>", markup, re.DOTALL)
    if foot_match:
        foot_body = foot_match.group(1)
        fn_blocks = re.findall(r"<fn[^>]*>(.*?)</fn>", foot_body, re.DOTALL)
        footnotes = []
        for fn in fn_blocks:
            fn_label = ""
            fn_label_match = re.search(r"<label>(.*?)</label>", fn)
            if fn_label_match:
                fn_label = clean_text(re.sub(r"<[^>]+>", "", fn_label_match.group(1)))
            fn_text = clean_text(re.sub(r"<[^>]+>", "", fn))
            if fn_text:
                footnotes.append({"label": fn_label, "text": fn_text})
        if footnotes:
            result["footnotes"] = footnotes

    # Abbreviation extraction from caption notes
    abbrevs = {}
    all_text = " ".join(result.get("caption_notes", []))
    for m in re.finditer(r"([A-Z][A-Za-z0-9]{0,8})\s*[:=]\s*([^;,]+)", all_text):
        abbr = m.group(1).strip()
        meaning = clean_text(m.group(2))
        if len(abbr) >= 2 and len(meaning) >= 3 and re.search(r"[a-zA-Z]", meaning):
            abbrevs[abbr] = meaning
    if abbrevs:
        result["abbreviations"] = abbrevs

    return result


# ---------------------------------------------------------------------------
# Table size helpers
# ---------------------------------------------------------------------------

def get_table_size(table: Dict[str, Any]) -> Tuple[int, int]:
    row_max = -1
    col_max = -1
    for cell in table.get("cells", []):
        if not isinstance(cell, dict):
            continue
        if cell.get("row_nums"):
            row_max = max(row_max, max(cell["row_nums"]))
        if cell.get("column_nums"):
            col_max = max(col_max, max(cell["column_nums"]))
    return row_max + 1, col_max + 1


def build_header_map(table: Dict[str, Any]) -> Dict[int, str]:
    headers: Dict[int, List[str]] = {}
    for cell in table.get("cells", []):
        if not isinstance(cell, dict):
            continue
        row_nums = cell.get("row_nums", [])
        col_nums = cell.get("column_nums", [])
        if not row_nums or not col_nums:
            continue
        text = clean_text(cell.get("xml_text_content") or cell.get("xml_raw_text_content"))
        if not text:
            continue
        if cell.get("is_column_header") or min(row_nums) == 0:
            for c in col_nums:
                headers.setdefault(c, []).append(text)
    merged = {}
    for c, vals in headers.items():
        seen = set()
        uniq = [v for v in vals if not (v in seen or seen.add(v))]
        merged[c] = " | ".join(uniq)
    return merged


# ---------------------------------------------------------------------------
# Main enrichment pipeline
# ---------------------------------------------------------------------------

def enrich_table(
    doc_id: str,
    table_index: int,
    table: Dict[str, Any],
    words_index: Dict[str, Path],
    image_index: Dict[str, Path],
) -> Dict[str, Any]:
    """Produce enriched annotation for one table."""
    parts = table.get("parts", [])
    cells = table.get("cells", [])
    n_rows, n_cols = get_table_size(table)
    header_map = build_header_map(table)

    # Row → page mapping
    row_page = map_rows_to_pages(cells, parts, words_index, doc_id, n_rows, n_cols)

    # Page-level row groups
    page_row_groups: Dict[int, List[int]] = defaultdict(list)
    for r, pn in sorted(row_page.items()):
        page_row_groups[pn].append(r)

    # Detect page boundary rows
    page_boundaries: List[Dict[str, Any]] = []
    sorted_rows = sorted(row_page.items())
    for i in range(len(sorted_rows) - 1):
        r_cur, p_cur = sorted_rows[i]
        r_next, p_next = sorted_rows[i + 1]
        if p_cur != p_next:
            page_boundaries.append({
                "last_row_on_page": r_cur,
                "page": p_cur,
                "first_row_on_next_page": r_next,
                "next_page": p_next,
            })

    # Caption info
    caption_info = parse_caption_info(table.get("original_markup", ""))

    # Cell bbox mapping (for a subset — first/last row per page + header)
    cell_bboxes: Dict[str, List[float]] = {}
    target_rows = set()
    target_rows.add(0)  # header
    for pn, rows_on_page in page_row_groups.items():
        if rows_on_page:
            target_rows.add(min(rows_on_page))
            target_rows.add(max(rows_on_page))
    for bd in page_boundaries:
        target_rows.add(bd["last_row_on_page"])
        target_rows.add(bd["first_row_on_next_page"])

    for cell in cells:
        if not isinstance(cell, dict):
            continue
        cell_rows = cell.get("row_nums", [])
        if not any(r in target_rows for r in cell_rows):
            continue
        text = clean_text(cell.get("xml_text_content") or cell.get("xml_raw_text_content"))
        if not text or len(text) < 2:
            continue
        # Find which page this cell is on
        cell_page = None
        for r in cell_rows:
            if r in row_page:
                cell_page = row_page[r]
                break
        if cell_page is None:
            continue
        page_id = f"{doc_id}_page_{cell_page}"
        page_words = load_words(words_index, page_id)
        part_bbox = None
        for part in parts:
            if part.get("page_num") == cell_page:
                part_bbox = part.get("bbox") or part.get("pdf_bbox")
                break
        if not part_bbox:
            continue
        bbox = locate_cell_bbox(text, page_words, part_bbox)
        if bbox:
            key = f"r{cell_rows[0]}_c{cell.get('column_nums', [0])[0]}"
            cell_bboxes[key] = bbox

    # Images and page_ids
    page_ids = []
    images = []
    for part in parts:
        pn = part.get("page_num")
        if pn is None:
            continue
        pid = f"{doc_id}_page_{pn}"
        img_path = image_index.get(f"{pid}.jpg")
        if pid not in page_ids:
            page_ids.append(pid)
            images.append(str(img_path) if img_path else "")

    return {
        "doc_id": doc_id,
        "table_index": table_index,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "page_ids": page_ids,
        "images": images,
        "header_map": {str(k): v for k, v in header_map.items()},
        "row_to_page": {str(r): pn for r, pn in row_page.items()},
        "page_row_groups": {str(pn): rows for pn, rows in page_row_groups.items()},
        "page_boundaries": page_boundaries,
        "caption_info": caption_info,
        "cell_bboxes": cell_bboxes,
    }


def process_doc(
    table_file: Path,
    words_index: Dict[str, Path],
    image_index: Dict[str, Path],
    min_table_pages: int = 2,
) -> List[Dict[str, Any]]:
    doc_id = table_file.stem.replace("_tables", "")
    data = load_json_file(table_file)
    if not isinstance(data, list):
        return []

    results = []
    for table_index, table in enumerate(data, start=1):
        parts = table.get("parts", [])
        if not isinstance(parts, list) or len(parts) < min_table_pages:
            continue

        page_ids = []
        for part in parts:
            pn = part.get("page_num")
            if pn is None:
                continue
            pid = f"{doc_id}_page_{pn}"
            img_path = image_index.get(f"{pid}.jpg")
            if img_path and pid not in page_ids:
                page_ids.append(pid)

        if len(page_ids) < min_table_pages:
            continue

        enriched = enrich_table(doc_id, table_index, table, words_index, image_index)
        results.append(enriched)

    return results


def main():
    ap = argparse.ArgumentParser(description="Enrich PubTables-v2 annotations")
    ap.add_argument("--root", required=True, help="PubTables-v2 full_documents directory")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-table-pages", type=int, default=2)
    ap.add_argument("--max-table-files", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.root)
    split_root = root / "Full Documents" / args.split
    if not split_root.exists():
        split_root = root / args.split

    table_dir = split_root / "tables"
    table_files = sorted(
        p for p in table_dir.rglob("*.json")
        if p.is_file() and "/tables/" in p.as_posix()
    )
    if args.max_table_files:
        table_files = table_files[:args.max_table_files]

    images_root = split_root / "images"
    words_root = split_root / "words"
    image_index = build_file_index(images_root, "*.jpg")
    words_index = build_file_index(words_root, "*_words.json")
    print(f"Indexed {len(image_index)} images, {len(words_index)} word files")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"enriched_{args.split}.jsonl"

    total_tables = 0
    total_rows_mapped = 0
    total_rows = 0
    total_captions = 0
    total_boundaries = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for idx, tf in enumerate(table_files, 1):
            enriched_list = process_doc(tf, words_index, image_index, args.min_table_pages)
            for enriched in enriched_list:
                f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                total_tables += 1
                total_rows += enriched["n_rows"]
                total_rows_mapped += len(enriched["row_to_page"])
                if enriched["caption_info"]:
                    total_captions += 1
                total_boundaries += len(enriched["page_boundaries"])

            if idx % 50 == 0 or idx == len(table_files):
                print(
                    f"[{idx}/{len(table_files)}] tables={total_tables} "
                    f"rows_mapped={total_rows_mapped}/{total_rows} "
                    f"captions={total_captions} boundaries={total_boundaries}"
                )

    print(f"\nSaved {total_tables} enriched tables -> {out_jsonl}")
    print(f"Row mapping coverage: {total_rows_mapped}/{total_rows} "
          f"({total_rows_mapped/max(total_rows,1)*100:.1f}%)")
    print(f"Tables with caption: {total_captions}/{total_tables}")
    print(f"Page boundaries detected: {total_boundaries}")


if __name__ == "__main__":
    main()
