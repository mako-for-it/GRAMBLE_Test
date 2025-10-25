# two_pick_best_elc.py
import os
import re
import csv
import argparse
from typing import List

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from tokenizers import Tokenizer

# ELC-BERT repo modules – make sure PYTHONPATH includes the elc-bert root
#   export PYTHONPATH=/home/you/path/to/elc-bert:$PYTHONPATH
from pre_training.config import BertConfig
from models.model_elc_bert_base import Bert


def find_mask_id(tok: Tokenizer) -> int:
    """Try several common mask token strings; raise if not found."""
    candidates = ["[MASK]", "<mask>", "<MASK>", "MASK", "<M>"]
    for s in candidates:
        tid = tok.token_to_id(s)
        if tid is not None:
            return int(tid)
    raise RuntimeError(
        "Could not find a mask token id in the tokenizer. "
        f"Checked {candidates}. Add your mask token to the vocab or edit this list."
    )


def normalize_variants(df: pd.DataFrame, id_col: str, variant_id_col: str, text_col: str) -> pd.DataFrame:
    """
    Ensure long format: columns = [id_col, variant_id_col, text_col].
    If already long, pass through; otherwise melt wide variant_* columns.
    """
    # already long?
    if {id_col, variant_id_col, text_col}.issubset(df.columns):
        out = df[[id_col, variant_id_col, text_col]].copy()
    else:
        # collect variant-like columns flexibly
        var_cols = [
            c for c in df.columns
            if c != id_col and re.match(
                r"^(variant[_\-\s]?\d+|v\d+|cand(idate)?[_\-\s]?\d+)$", c, re.I
            )
        ]
        # fallback: use all non-id columns as candidates
        if not var_cols:
            var_cols = [c for c in df.columns if c != id_col]

        if not var_cols:
            raise ValueError(
                f"Couldn't find variant columns. CSV columns were: {list(df.columns)}.\n"
                f"Expect either long format with [{id_col},{variant_id_col},{text_col}] "
                f"or wide format like [{id_col}, variant_1, variant_2, ...]."
            )

        out = df.melt(
            id_vars=[id_col],
            value_vars=var_cols,
            var_name=variant_id_col,
            value_name=text_col,
        )

    # drop empty/null candidates
    out = out.dropna(subset=[text_col])
    out[text_col] = out[text_col].astype(str).str.strip()
    out = out[out[text_col] != ""].reset_index(drop=True)

    # keep variant id as string (it may be 'variant_3'); no strict numeric cast needed
    out[variant_id_col] = out[variant_id_col].astype(str)

    # final sanity
    for col in (id_col, variant_id_col, text_col):
        if col not in out.columns:
            raise ValueError(f"Column '{col}' missing after normalization. Have: {list(out.columns)}")
    return out


@torch.no_grad()
def pll_score(model, tok, text, max_len, mask_id, device):
    ids = tok.encode(text).ids[:max_len]
    if not ids:
        return -1e9

    input_ids = torch.tensor(ids, dtype=torch.long, device=device)  # [T]
    T = input_ids.size(0)
    total_logp = 0.0

    for i in range(T):
        masked = input_ids.clone()
        true_tok = int(masked[i].item())
        masked[i] = mask_id

        seq = masked.unsqueeze(1)  # [T, 1]  (model expects tokens as [T, B])

        # ✅ padding mask must be boolean and shaped [B, T] here (B=1)
        attn_mask = torch.zeros((1, T), dtype=torch.bool, device=device)

        # contextual states [T, B, H]
        contextual = model.get_contextualized(seq, attn_mask)[0]

        # logits for position i -> [1, V]
        logit_i = model.classifier(contextual[i:i+1, 0, :])

        total_logp += torch.log_softmax(logit_i, dim=-1)[0, true_tok].item()

    return float(total_logp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants_csv", required=True, help="CSV with variants (wide or long format).")
    ap.add_argument("--best_out", required=True, help="TSV to write: id<TAB>best_hyp.")
    ap.add_argument("--config_file", required=True, help="ELC-BERT config JSON.")
    ap.add_argument("--tokenizer_path", required=True, help="HF Tokenizers JSON (same used for training).")
    ap.add_argument("--checkpoint_path", required=True, help="ELC-BERT checkpoint (model.bin).")
    ap.add_argument("--text_col", default="candidate", help="Column name for the text.")
    ap.add_argument("--id_col", default="id", help="Column name for the example id.")
    ap.add_argument("--variant_id_col", default="variant", help="Column with variant id / name.")
    ap.add_argument("--seq_length", type=int, default=128, help="Truncate text to this many subwords.")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"),
                    help="Device: cuda or cpu.")
    args = ap.parse_args()

    device = torch.device(args.device)

    # tokenizer + mask id
    tok = Tokenizer.from_file(args.tokenizer_path)
    mask_id = find_mask_id(tok)

    # model
    cfg = BertConfig(args.config_file)
    model = Bert(cfg, activation_checkpointing=False)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")  # keep weights_only=False for backward compat
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(device).eval()

    # read & normalize variants
    df = pd.read_csv(args.variants_csv)
    df = normalize_variants(df, args.id_col, args.variant_id_col, args.text_col)

    # score and pick best per id (with progress bar)
    out_rows: List[dict] = []
    n_ids = df[args.id_col].nunique()
    for ex_id, grp in tqdm(df.groupby(args.id_col), total=n_ids, desc="Scoring"):
        best_row = None
        best_score = -1e18
        for _, r in grp.iterrows():
            text = str(r[args.text_col])
            s = pll_score(model, tok, text, args.seq_length, mask_id, device)
            if s > best_score:
                best_score = s
                best_row = r
        out_rows.append({
            "id": ex_id,
            "hyp": str(best_row[args.text_col]),
            "score": best_score,
            "picked_variant": str(best_row[args.variant_id_col]),
        })

    out_df = pd.DataFrame(out_rows).sort_values("id")

    # write TSV expected by your eval step: id \t hyp
    with open(args.best_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for _, r in out_df.iterrows():
            w.writerow([r["id"], r["hyp"]])

    # Optional companion CSV with scores & which variant was picked
    aux = os.path.splitext(args.best_out)[0] + ".scored.csv"
    out_df.to_csv(aux, index=False)

    print(f"Wrote {args.best_out} with {len(out_df)} rows")
    print(f"(Also wrote {aux} with PLL scores and chosen variant.)")


if __name__ == "__main__":
    main()


