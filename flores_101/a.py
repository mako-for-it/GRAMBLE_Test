!pip install -U datasets

from datasets import load_dataset

# Load the 'all' config (gives every language aligned in the same rows)
all_devtest = load_dataset("gsarti/flores_101", "all", split="devtest")

# Keep only English and Xhosa (plus id)
pairs = all_devtest.select_columns(["id", "sentence_eng", "sentence_xho"])
pairs.to_pandas().to_csv("flores101_eng_xho_devtest.csv", index=False)
