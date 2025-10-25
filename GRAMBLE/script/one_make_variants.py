# one_make_variants.py
import argparse, csv, pandas as pd, itertools

def expand_tokens(tokens):
    """
    Replace this stub with your real generator.
    Each token -> list of alternatives (at least [itself]).
    Example shows a tiny toy map.
    """
    alt = {
        "I": ["I", "a", "b"],      # example
        "have": ["have", "ss"],
        "a": ["a", "djk"],
        "pen": ["pen", "shd", "sjhk"],
    }
    return [alt.get(t, [t]) for t in tokens]

def variants_for_sentence(text, max_variants=64):
    toks = text.strip().split()
    choices = expand_tokens(toks)
    # cartesian product; cap to avoid explosions
    out = []
    for combo in itertools.product(*choices):
        out.append(" ".join(combo))
        if len(out) >= max_variants:
            break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)   # e.g., flores101_eng_xho_devtest.csv
    ap.add_argument("--source_col", default="sentence_eng")  # or "sentence_xho"
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_variants", type=int, default=64)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    rows = []
    for _, r in df.iterrows():
        s = str(r[args.source_col])
        vars_ = variants_for_sentence(s, args.max_variants)
        rows.append([r["id"]] + vars_)

    # Write wide CSV: id, variant_1, variant_2, ...
    maxw = max(len(r) for r in rows)
    header = ["id"] + [f"variant_{i}" for i in range(1, maxw)]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r + [""]*(maxw-len(r)))

if __name__ == "__main__":
    main()
