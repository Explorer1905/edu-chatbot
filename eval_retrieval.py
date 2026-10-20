import pandas as pd
import numpy as np
from tqdm import tqdm
from retriever import Retriever
from config import *

N_QUESTIONS = 1000
SEED = 42

qa = pd.read_csv(QA_CSV)
corpus = pd.read_csv(CORPUS_CSV)

key_to_id = {(r.subject, int(r.grade), r.Explanation): r.id for r in corpus.itertuples()}
qa["gold_id"] = [key_to_id.get((r.subject, int(r.grade), r.Explanation)) for r in qa.itertuples()]
qa = qa.dropna(subset=["gold_id"])

id_to_topic = dict(zip(corpus["id"], corpus["Topic"].str.lower().str.strip()))
id_to_meta = {r.id: (r.subject, int(r.grade)) for r in corpus.itertuples()}

sample = qa.sample(N_QUESTIONS, random_state=SEED).reset_index(drop=True)
print(f"Evaluating {len(sample)} questions on a corpus of {len(corpus)} passages\n")

R = Retriever()
modes = ["dense", "bm25", "hybrid"]
res = {m: {"hit1": [], "hit5": [], "hit10": [], "rr": [], "topic5": [], "subj5": []} for m in modes}

for row in tqdm(sample.itertuples(), total=len(sample)):
    gold = row.gold_id
    gold_topic = id_to_topic[gold]
    gold_subj, gold_grade = id_to_meta[gold]
    for mode in modes:
        ids = [p["id"] for p in R.retrieve(row.Question, k=10, mode=mode)]
        rank = ids.index(gold) + 1 if gold in ids else None
        res[mode]["hit1"].append(rank == 1)
        res[mode]["hit5"].append(bool(rank and rank <= 5))
        res[mode]["hit10"].append(bool(rank))
        res[mode]["rr"].append(1.0 / rank if rank else 0.0)
        # lenient: any top-5 passage with the same topic name
        res[mode]["topic5"].append(any(id_to_topic[i] == gold_topic for i in ids[:5]))
        # coarse: any top-5 passage from the same subject and grade
        res[mode]["subj5"].append(any(id_to_meta[i] == (gold_subj, gold_grade) for i in ids[:5]))


def ci95(x):
    x = np.array(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


print("\n===== RETRIEVAL RESULTS =====")
print(f"{'method':<8}{'Exact@1':>9}{'Exact@5':>9}{'Exact@10':>10}{'MRR':>7}{'Topic@5':>9}{'Subj+Grade@5':>14}")
for m in modes:
    r = res[m]
    print(f"{m:<8}{np.mean(r['hit1']):>9.3f}{np.mean(r['hit5']):>9.3f}{np.mean(r['hit10']):>10.3f}"
          f"{np.mean(r['rr']):>7.3f}{np.mean(r['topic5']):>9.3f}{np.mean(r['subj5']):>14.3f}")

print("\n95% confidence interval on Exact@5 (+/-):")
for m in modes:
    print(f"  {m:<8}{np.mean(res[m]['hit5']):.3f} +/- {ci95(res[m]['hit5']):.3f}")