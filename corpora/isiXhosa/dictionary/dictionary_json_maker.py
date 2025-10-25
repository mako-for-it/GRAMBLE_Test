#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, json, re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ---- Entry starts (uppercase & lowercase heads) -----------------------------
ENTRY_START_UPPER = re.compile(
    r"^(?P<head>[A-Z][A-Za-z\-’']*)\s*,\s*(?P<pos>(?:[a-z]+\.\s*)+)(?P<body>.*)$"
)
ENTRY_HEAD_ONLY_UPPER = re.compile(
    r"^(?P<head>[A-Z][A-Za-z\-’']*)\s*,\s*(?P<pos>(?:[a-z]+\.\s*)+)\s*$"
)

ENTRY_START_LOWER = re.compile(
    r"^(?P<head>[a-z][a-z\-’']*)\s*,\s*(?P<pos>(?:[a-z]+\.\s*)+)(?P<body>.*)$"
)
ENTRY_HEAD_ONLY_LOWER = re.compile(
    r"^(?P<head>[a-z][a-z\-’']*)\s*,\s*(?P<pos>(?:[a-z]+\.\s*)+)\s*$"
)

def match_entry(line: str):
    for rx in (ENTRY_START_UPPER, ENTRY_HEAD_ONLY_UPPER, ENTRY_START_LOWER, ENTRY_HEAD_ONLY_LOWER):
        m = rx.match(line)
        if m:
            return m, rx in (ENTRY_START_UPPER, ENTRY_START_LOWER)
    return None, False

def looks_like_entry_start(line: str) -> bool:
    m, _ = match_entry(line)
    return bool(m)

# ---- POS normalization ------------------------------------------------------
POS_SHORT = {
    "a.": "a", "adj.": "a",
    "adv.": "adv",
    "n.": "n",
    "v.": "v",
    "v. t.": "v t", "v.t.": "v t",
    "v. i.": "v i", "v.i.": "v i",
    "v. aux.": "v",
    "prep.": "prep", "conj.": "conj",
    "intj.": "intj", "pron.": "pron", "pr.": "pron",
}

def pos_tokens_from_raw(pos_raw: str) -> List[str]:
    s = re.sub(r"\s+", " ", pos_raw.strip())
    if s in POS_SHORT:
        short = POS_SHORT[s]
    else:
        short = None
        for key in sorted(POS_SHORT, key=len, reverse=True):
            if key in s:
                short = POS_SHORT[key]; break
        if short is None:  # fallback
            short = s
    toks = short.split()
    return toks if toks else [""]

# ---- Glue wrapped lines -----------------------------------------------------
def glue_blocks(lines: List[str]) -> List[str]:
    blocks, cur = [], ""
    for raw in lines:
        line = raw.rstrip()
        if not line: 
            continue
        if looks_like_entry_start(line):
            if cur:
                blocks.append(cur.strip())
            cur = line
        else:
            if cur:
                cur += " " + line.strip()
    if cur:
        blocks.append(cur.strip())
    return blocks

# ---- Fragment helpers -------------------------------------------------------
INLINE_POS_RE = re.compile(r"^(?P<pos>(?:[a-z]+\.\s*)+)\s*(?P<rest>.+)$")
PAREN_DESC_THEN_POS_RE = re.compile(r"^\((?P<desc>[^)]+)\)\s*,\s*(?P<pos>(?:[a-z]+\.\s*)+)\s*(?P<rest>.+)$")
DESC_THEN_POS_RE = re.compile(r"^(?P<desc>[^,]+?)\s*,\s*(?P<pos>(?:[a-z]+\.\s*)+)\s*(?P<rest>.+)$")
PAREN_DESC_RE = re.compile(r"^\((?P<desc>[^)]+)\)\s*,\s*(?P<rest>.+)$")

def split_by_semicolons(text: str) -> List[str]:
    t = text.strip()
    if t.endswith("."):
        t = t[:-1].rstrip()
    return [p.strip(" ;") for p in t.split(";") if p.strip(" ;")]

def split_csv_like(text: str) -> List[str]:
    return [p.strip(" ,") for p in text.split(",") if p.strip(" ,")]

# Safer "english, xhosa" detector: ONLY when left looks like plain English (no POS tokens),
# and the fragment has exactly ONE comma split we want to use.
POS_TOKEN_IN_LEFT_RE = re.compile(r"\b(?:a|adj|adv|n|v|prep|conj|intj|pron|pr)\.", re.I)
def english_xh_pair_safe(fragment: str) -> Optional[Tuple[str, str]]:
    if fragment.count(",") == 0:
        return None
    if fragment.count(",") >= 3:  # too many commas → likely not a clean pair
        return None
    left, right = fragment.rsplit(",", 1)
    left, right = left.strip(), right.strip()
    # Reject if left contains POS abbreviations like "a.", "n.", "v."
    if POS_TOKEN_IN_LEFT_RE.search(left):
        return None
    # Must look ASCII-ish with letters
    if not (all(ord(c) < 128 for c in left) and re.search(r"[A-Za-z]", left)):
        return None
    # Right must be non-empty
    if not right:
        return None
    return (left, right)

# ---- Derivative merging (lowercase head derived from previous) --------------
_DERIV_SUFFIXES = ("ed", "ing", "er", "ers", "ment", "ments")
def is_derivative_of(prev_head: str, this_head: str) -> bool:
    if not prev_head: return False
    p, t = prev_head.lower(), this_head.lower()
    if not t.startswith(p): return False
    rest = t[len(p):]
    return rest in _DERIV_SUFFIXES

# ---- Main conversion --------------------------------------------------------
def convert_blocks_to_indexed_json(blocks: List[str]) -> Dict[str, Dict]:
    output: Dict[str, Dict] = {}
    idx = 1
    last_main_key: Optional[str] = None
    last_main_head: Optional[str] = None

    def append_senses(to_key: str, senses: List[Dict]) -> None:
        obj = output[to_key]
        k = 1
        while f"sense_{k}" in obj:
            k += 1
        for s in senses:
            obj[f"sense_{k}"] = s
            k += 1

    for block in blocks:
        m, has_body = match_entry(block)
        if not m: 
            continue
        head = m.group("head")
        header_pos_tokens = pos_tokens_from_raw(m.group("pos"))
        body = (m.group("body") if has_body else "").strip()

        senses_here: List[Dict] = []

        if body:
            for frag in split_by_semicolons(body):
                # 1) Parenthetical desc + inline POS
                mdp = PAREN_DESC_THEN_POS_RE.match(frag)
                if mdp:
                    desc = mdp.group("desc").strip()
                    pos_toks = pos_tokens_from_raw(mdp.group("pos"))
                    for xh in split_csv_like(mdp.group("rest").strip().rstrip(".")):
                        for tok in pos_toks:
                            senses_here.append({"syntax": tok, "description": desc, "translation": {"xh": xh}})
                    continue

                # 2) Non-parenthetical desc + inline POS
                mdn = DESC_THEN_POS_RE.match(frag)
                if mdn:
                    desc = mdn.group("desc").strip()
                    pos_toks = pos_tokens_from_raw(mdn.group("pos"))
                    for xh in split_csv_like(mdn.group("rest").strip().rstrip(".")):
                        for tok in pos_toks:
                            senses_here.append({"syntax": tok, "description": desc, "translation": {"xh": xh}})
                    continue

                # 3) Inline POS at start (e.g., "a. -totyiwe, -xyz" or "n. uku-...")
                mip = INLINE_POS_RE.match(frag)
                if mip:
                    pos_toks = pos_tokens_from_raw(mip.group("pos"))
                    for xh in split_csv_like(mip.group("rest").strip().rstrip(".")):
                        for tok in pos_toks:
                            senses_here.append({"syntax": tok, "description": "", "translation": {"xh": xh}})
                    continue

                # 4) Parenthetical desc + list (use header POS)
                mp = PAREN_DESC_RE.match(frag)
                if mp:
                    desc = mp.group("desc").strip()
                    for xh in split_csv_like(mp.group("rest").strip().rstrip(".")):
                        for tok in header_pos_tokens:
                            senses_here.append({"syntax": tok, "description": desc, "translation": {"xh": xh}})
                    continue

                # 5) Plain list (use header POS)
                plain_items = split_csv_like(frag.rstrip("."))
                if plain_items:
                    for xh in plain_items:
                        for tok in header_pos_tokens:
                            senses_here.append({"syntax": tok, "description": "", "translation": {"xh": xh}})
                    continue

                # 6) LAST RESORT: English, Xhosa (but only if safe)
                pair = english_xh_pair_safe(frag)
                if pair:
                    en_word, xh_word = pair
                    word_key = f"word_{idx}"; idx += 1
                    output[word_key] = {
                        "word_name": en_word,
                        "sense_1": {"syntax": "", "description": "", "translation": {"xh": xh_word}},
                    }
                    continue

        # Decide where to put senses (merge lowercase derivatives into last main)
        is_lower = head[0].islower()
        should_merge = is_lower and last_main_head and is_derivative_of(last_main_head, head)

        if senses_here:
            if should_merge and last_main_key:
                append_senses(last_main_key, senses_here)
            else:
                word_key = f"word_{idx}"; idx += 1
                obj: Dict[str, Dict] = {"word_name": head}
                for i, s in enumerate(senses_here, 1):
                    obj[f"sense_{i}"] = s
                output[word_key] = obj
                last_main_key = word_key
                last_main_head = head
        else:
            # No senses parsed; still update last head for potential merge next block
            last_main_head = head

    return output

# ---- CLI --------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 xhosa_to_json.py input.txt"); sys.exit(1)
    in_path = Path(sys.argv[1]).expanduser().resolve()
    out_path = in_path.parent / "xhosa-en.json"

    with in_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    blocks = glue_blocks(lines)
    data = convert_blocks_to_indexed_json(blocks)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Wrote {len(data)} word entries → {out_path}")

if __name__ == "__main__":
    main()
