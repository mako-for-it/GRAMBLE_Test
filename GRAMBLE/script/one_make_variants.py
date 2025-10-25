#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Set
import pandas as pd
from datasets import load_dataset

WORD_RE = re.compile(r"[A-Za-z\-’']+|\d+|[^\sA-Za-z\-’'\d]")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text)

def is_word(tok: str) -> bool:
    return re.match(r"^[A-Za-z\-’']+$", tok) is not None

def strip_punct(tok: str) -> str:
    return re.sub(r"^[^\w’'-]+|[^\w’'-]+$", "", tok)

def normalize_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def load_dict(json_path: Path) -> Tuple[Dict[str, Set[str]], int]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    en2xh: Dict[str, Set[str]] = {}
    for _, obj in data.items():
        en_head = obj.get("word_name", "")
        for k, v in obj.items():
            if not k.startswith("sense_"): 
                continue
            xh = v.get("translation", {}).get("xh")
            if not xh: 
                continue
            en2xh.setdefault(normalize_key(en_head), set()).add(xh)
    max_key_len = max((len(k.split()) for k in en2xh.keys()), default=1)
    return en2xh, max_key_len

def segment_choices_en2xh(tokens: List[str], en2xh: Dict[str, Set[str]], max_key_len: int) -> List[List[str]]:
    i, N = 0, len(tokens)
    segments: List[List[str]] = []
    while i < N:
        tok = tokens[i]
        if not is_word(tok):
            segments.append([tok]); i += 1; continue
        matched = False
        for L in range(min(max_key_len, N - i), 0, -1):
            phrase = " ".join(strip_punct(t) for t in tokens[i:i+L] if is_word(t))
            key = normalize_key(phrase)
            if key in en2xh:
                segments.append(sorted(en2xh[key]))
                i += L; matched = True; break
        if not matched:
            segments.append([tokens[i]])
            i += 1
    return segments

def product_join(segments: List[List[str]], max_out: int) -> Iterable[str]:
    out_count = 0
    def rec(idx: int, acc: List[str]):
        nonlocal out_count
        if out_count >= max_out: return
        if idx == len(segments):
            s = " ".join(acc)
            s = re.sub(r"\s+([,.;:!?])", r"\1", s)
            yield s; out_count += 1; return
        for choice in segments[idx]:
            if acc and re.match(r"^[,.;:!?]$", choice):
                new_acc = acc[:-1] + [acc[-1] + choice]
            else:
                new_acc = acc + [choice]
            yield from rec(idx + 1, new_acc)
    yield from rec(0, [])

def main():
    ap = argparse.ArgumentParser(description="FLORES → variants CSV (one row per source line)")
    ap.add_argument("--dict", default="xhosa-en.json")
    ap.add_argument("--split", choices=["dev","devtest"], default="devtest")
    ap.add_argument("--out_csv", default="variants.csv")
    ap.add_argument("--max", type=int, default=2000, help="max variants per line")
    ap.add_argument("--src_lang", default="eng", help="source language column suffix in FLORES all-config")
    ap.add_argument("--tgt_lang", default="xho", help="target language column suffix in FLORES all-config")
    args = ap.parse_args()

    # Load FLORES-101 aligned split (english+xhosa on same row)
    ds = load_dataset("gsarti/flores_101", "all", split=args.split)
    src_col = f"sentence_{args.src_lang}"
    ref_col = f"sentence_{args.tgt_lang}"
    if src_col not in ds.column_names or ref_col not in ds.column_names:
        raise ValueError(f"Columns not found: {src_col}, {ref_col}")

    en2xh, max_key_len = load_dict(Path(args.dict))

    rows = []
    for ex in ds:
        src = ex[src_col]
        toks = tokenize(src)
        segs = segment_choices_en2xh(toks, en2xh, max_key_len)
        variants = list(product_join(segs, args.max))
        rows.append(variants)

    # pad jagged rows to longest width
    maxw = max((len(r) for r in rows), default=0)
    padded = [r + [""]*(maxw - len(r)) for r in rows]
    cols = [f"variant_{i}" for i in range(1, maxw+1)]
    df = pd.DataFrame(padded, columns=cols)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with {len(rows)} rows and up to {maxw} variants/row")

if __name__ == "__main__":
    main()
