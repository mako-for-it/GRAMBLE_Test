# check_exact_match.py
import pandas as pd
from datasets import load_dataset

split = "devtest"
tgt = "xho"

# refs
ds = load_dataset("gsarti/flores_101", "all", split=split)
refs = pd.Series([ex[f"sentence_{tgt}"] for ex in ds])

# hyps (id \t hyp)
hyps = pd.read_csv("best.devtest.tsv", sep="\t", header=None, names=["id","hyp"])
match_rate = (hyps["hyp"].reset_index(drop=True) == refs.reset_index(drop=True)).mean()
print("Exact match rate:", match_rate)


from datasets import load_dataset
ds = load_dataset("gsarti/flores_101", "all", split="devtest")
print("Sample src (eng):", ds[0]["sentence_eng"])
print("Sample ref (xho):", ds[0]["sentence_xho"])
print("Sample hyp:", open("best.devtest.tsv", encoding="utf-8").readline().split("\t",1)[1].strip())
i

mport random, json
from datasets import load_dataset
from sacrebleu import corpus_bleu

ds = load_dataset("gsarti/flores_101", "all", split="devtest")
refs = [ex["sentence_xho"] for ex in ds]
hyps = [line.rstrip("\n").split("\t",1)[1] for line in open("best.devtest.tsv", encoding="utf-8")]

random.shuffle(refs)
print("BLEU against shuffled refs:", corpus_bleu(hyps, [refs]).score)
