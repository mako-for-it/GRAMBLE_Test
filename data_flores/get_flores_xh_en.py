# get_gsarti_flores_xh_en.py
from datasets import load_dataset
import pandas as pd

SPLIT = "devtest"  # or "dev" / "test"

# 'all' config = every language in the same table (aligned by row/id)
ds = load_dataset("gsarti/flores_101", "all", split=SPLIT)

# Keep id + English + Xhosa (column names in this repo)
pairs = ds.select_columns(["id", "sentence_eng", "sentence_xho"])
pairs.to_pandas().to_csv(f"flores101_eng_xho_{SPLIT}.csv", index=False)
print(f"wrote flores101_eng_xho_{SPLIT}.csv ({len(pairs)} rows)")

# Optional plain-text files for eval tools
with open(f"src.eng.{SPLIT}", "w", encoding="utf-8") as sf, \
     open(f"ref.xho.{SPLIT}", "w", encoding="utf-8") as rf:
    for e, x in zip(pairs["sentence_eng"], pairs["sentence_xho"]):
        sf.write(e + "\n")
        rf.write(x + "\n")
print(f"wrote src.eng.{SPLIT} and ref.xho.{SPLIT}")
