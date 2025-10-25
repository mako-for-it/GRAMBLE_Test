#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, math
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@torch.no_grad()
def ppl_for_sentences(model, tok, texts, device="cpu", max_len=512):
    scores = []
    for t in texts:
        if not t:
            scores.append(float("inf")); continue
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc["input_ids"].to(device)
        # labels = input_ids shifts internally by ignoring -100
        out = model(input_ids=input_ids, labels=input_ids)
        # cross-entropy per token
        neglog = out.loss.item()  # average nll per token in batch
        ppl = math.exp(neglog)
        scores.append(ppl)
    return scores

def main():
    ap = argparse.ArgumentParser(description="Pick best variant by LM perplexity")
    ap.add_argument("--variants_csv", default="variants.csv")
    ap.add_argument("--best_out", default="best.tsv")
    ap.add_argument("--model", required=True, help="Hugging Face model name or path (your BabyLM)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    df = pd.read_csv(args.variants_csv)
    cols = [c for c in df.columns if c.startswith("variant_")]
    if not cols:
        raise ValueError("No variant_* columns in CSV")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()

    best_lines = []
    for _, row in df.iterrows():
        candidates = [str(row[c]) for c in cols if isinstance(row[c], str) and row[c].strip() != ""]
        if not candidates:
            best_lines.append("")
            continue
        ppls = ppl_for_sentences(model, tok, candidates, device=args.device)
        best = candidates[int(min(range(len(ppls)), key=lambda i: ppls[i]))]
        best_lines.append(best)

    Path(args.best_out).write_text("\n".join(best_lines), encoding="utf-8")
    print(f"Wrote best hypotheses to {args.best_out} ({len(best_lines)} lines)")

if __name__ == "__main__":
    main()
