#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os
from datasets import load_dataset
from tqdm import tqdm
from sacrebleu import corpus_bleu, corpus_chrf, __version__ as sacrebleu_version

def load_refs(split: str, tgt_lang: str):
    ds = load_dataset("gsarti/flores_101", "all", split=split)
    col = f"sentence_{tgt_lang}"
    refs = []
    for ex in tqdm(ds, desc="Loading references"):
        refs.append(ex[col])
    return refs

def load_hyps(hyp_path: str, n_expected: int):
    hyps = []
    with open(hyp_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=n_expected, desc="Reading hypotheses", unit="sent"):
            line = line.rstrip("\n")
            # accept either plain text or TSV "id \t hyp"
            if "\t" in line:
                parts = line.split("\t", 1)
                hyp = parts[1]
            else:
                hyp = line
            hyps.append(hyp)
    if len(hyps) != n_expected:
        raise ValueError(f"hyp lines ({len(hyps)}) != ref lines ({n_expected})")
    return hyps

def safe_signature(score_obj, metric_name, extra=None):
    # sacrebleu >= 2.4 often lacks .signature; fall back to a minimal metadata dict
    sig = getattr(score_obj, "signature", None)
    if sig is not None:
        try:
            return str(sig)
        except Exception:
            pass
    md = {"metric": metric_name, "sacrebleu_version": sacrebleu_version}
    if extra:
        md.update(extra)
    return md

def main():
    ap = argparse.ArgumentParser(description="Evaluate BLEU and chrF++ on FLORES-101 (Xhosa)")
    ap.add_argument("--split", choices=["dev","devtest"], default="devtest")
    ap.add_argument("--hyp", default="best.tsv", help="TSV or text: either 'id<TAB>hyp' per line or just hyp")
    ap.add_argument("--tgt_lang", default="xho", help="FLORES-101 target code: e.g., xho")
    ap.add_argument("--out_json", default="metrics.json")
    args = ap.parse_args()

    refs = load_refs(args.split, args.tgt_lang)
    hyps = load_hyps(args.hyp, len(refs))

    print("Scoring…")
    bleu = corpus_bleu(hyps, [refs])  # default tokenizer "13a"
    chrf = corpus_chrf(hyps, [refs], char_order=6, word_order=2, beta=2)

    metrics = {
        "BLEU": bleu.score,
        "chrF2++": chrf.score,
        "BLEU_signature": safe_signature(bleu, "BLEU", {"tokenize": "13a"}),
        "chrF_signature": safe_signature(chrf, "chrF2++", {"char_order": 6, "word_order": 2, "beta": 2}),
        "num_sentences": len(refs),
        "split": args.split,
        "target_lang": args.tgt_lang,
        "hyp_file": os.path.basename(args.hyp),
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
