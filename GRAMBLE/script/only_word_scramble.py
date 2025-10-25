#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Set

WORD_RE = re.compile(r"[A-Za-z\-’']+|\d+|[^\sA-Za-z\-’'\d]")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text)

def is_word(tok: str) -> bool:
    return re.match(r"^[A-Za-z\-’']+$", tok) is not None

def strip_punct(tok: str) -> str:
    return re.sub(r"^[^\w’'-]+|[^\w’'-]+$", "", tok)

def normalize_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def load_dict(json_path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], int]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    en2xh: Dict[str, Set[str]] = {}
    xh2en: Dict[str, Set[str]] = {}

    def add_pair(en_key: str, xh_val: str):
        en_k = normalize_key(en_key)
        if not en_k or not xh_val:
            return
        en2xh.setdefault(en_k, set()).add(xh_val)
        xh2en.setdefault(xh_val, set()).add(en_k)

    for _, obj in data.items():
        en_head = obj.get("word_name", "")
        for k, v in obj.items():
            if not k.startswith("sense_"):
                continue
            trans = v.get("translation", {})
            xh = trans.get("xh")
            if not xh:
                continue
            add_pair(en_head, xh)

    max_key_len = max((len(k.split()) for k in en2xh.keys()), default=1)
    return en2xh, xh2en, max_key_len

def segment_choices_en2xh(tokens: List[str], en2xh: Dict[str, Set[str]], max_key_len: int) -> List[List[str]]:
    i, N = 0, len(tokens)
    segments: List[List[str]] = []
    while i < N:
        tok = tokens[i]
        if not is_word(tok):
            segments.append([tok])
            i += 1
            continue
        matched = False
        for L in range(min(max_key_len, N - i), 0, -1):
            phrase = " ".join(strip_punct(t) for t in tokens[i:i+L] if is_word(t))
            phrase_norm = normalize_key(phrase)
            if phrase_norm in en2xh:
                segments.append(sorted(en2xh[phrase_norm]))
                i += L
                matched = True
                break
        if not matched:
            segments.append([tokens[i]])
            i += 1
    return segments

def segment_choices_xh2en(tokens: List[str], xh2en: Dict[str, Set[str]]) -> List[List[str]]:
    segments: List[List[str]] = []
    for tok in tokens:
        if not is_word(tok):
            segments.append([tok])
        else:
            key = strip_punct(tok)
            segments.append(sorted(xh2en.get(key, {tok})))
    return segments

def product_join(segments: List[List[str]], max_out: int) -> Iterable[str]:
    out_count = 0
    def rec(idx: int, acc: List[str]):
        nonlocal out_count
        if out_count >= max_out:
            return
        if idx == len(segments):
            sent = " ".join(acc)
            sent = re.sub(r"\s+([,.;:!?])", r"\1", sent)
            yield sent
            out_count += 1
            return
        for choice in segments[idx]:
            # inline punctuation
            if acc and re.match(r"^[,.;:!?]$", choice):
                new_acc = acc[:-1] + [acc[-1] + choice]
            else:
                new_acc = acc + [choice]
            yield from rec(idx + 1, new_acc)
    yield from rec(0, [])

def main():
    ap = argparse.ArgumentParser(description="Generate sentence variants and write CSV (one row per input line).")
    ap.add_argument("--dict", default="xhosa-en.json", help="Path to dictionary JSON")
    ap.add_argument("--direction", choices=["en2xh", "xh2en"], default="en2xh")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="Single input sentence")
    g.add_argument("--file", help="Path to a text file with one input sentence per line")
    ap.add_argument("--output", default="variants.csv", help="Output CSV file name")
    ap.add_argument("--max", type=int, default=10000, help="Max variants per input line")
    args = ap.parse_args()

    dict_path = Path(args.dict)
    en2xh, xh2en, max_key_len = load_dict(dict_path)

    def variants_for_line(line: str) -> List[str]:
        line = line.strip()
        if not line:
            return []
        toks = tokenize(line)
        if args.direction == "en2xh":
            segments = segment_choices_en2xh(toks, en2xh, max_key_len)
        else:
            segments = segment_choices_xh2en(toks, xh2en)
        return list(product_join(segments, args.max))

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if args.input is not None:
            row = variants_for_line(args.input)
            writer.writerow(row)
        else:
            with open(args.file, "r", encoding="utf-8") as fin:
                for line in fin:
                    row = variants_for_line(line)
                    writer.writerow(row)

    print(f"Wrote CSV → {out_path.resolve()}")

if __name__ == "__main__":
    main()
