#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from datasets import load_dataset
from sacrebleu import corpus_bleu, corpus_chrf

def main():
    ap = argparse.ArgumentParser(description="Evaluate BLEU and chrF++ on FLORES-101")
    ap.add_argument("--split", choices=["dev","devtest"], default="devtest")
    ap.add_argument("--hyp", default="best.tsv", help="one line per hypothesis")
    ap.add_argument("--src_lang", default="eng")
    ap.add_argument("--tgt_lang", default="xho")
    ap.add_argument("--out_json", default="metrics.json")
    args = ap.parse_args()

    ds = load_dataset("gsarti/flores_101", "all", split=args.split)
    ref_col = f"sentence_{args.tgt_lang}"
    refs = [ex[ref_col] for ex in ds]

    hyps = [line.rstrip("\n") for line in open(args.hyp, "r", encoding="utf-8")]
    if len(hyps) != len(refs):
        raise ValueError(f"hyp lines ({len(hyps)}) != ref lines ({len(refs)})")

    bleu = corpus_bleu(hyps, [refs])  # default: tokenized internally by sacrebleu
    chrf = corpus_chrf(hyps, [refs], char_order=6, word_order=2, beta=2)

    metrics = {
        "BLEU": bleu.score,
        "chrF2++": chrf.score,
        "BLEU_signature": bleu.signature,
        "chrF_signature": chrf.signature
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
