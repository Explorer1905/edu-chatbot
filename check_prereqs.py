import pandas as pd
from collections import Counter

corpus = pd.read_csv("science_corpus.csv")

concepts = Counter()
per_row = []
for s in corpus["Prerequisites"].fillna(""):
    if not s or s.lower().startswith("no prerequisite"):
        per_row.append(0)
        continue
    items = [p.strip() for p in s.split(",") if p.strip()]
    per_row.append(len(items))
    concepts.update(i.lower() for i in items)

print("Passages with no prerequisite tag:", sum(1 for n in per_row if n == 0), "of", len(per_row))
print("Distinct prerequisite strings:", len(concepts))
print("Strings appearing only once:", sum(1 for c in concepts.values() if c == 1))
print("Strings appearing 5+ times:", sum(1 for c in concepts.values() if c >= 5))
print("Avg prerequisites per tagged passage: %.2f" % (sum(per_row) / max(1, sum(1 for n in per_row if n))))

print("\nTop 25 most common prerequisite strings:")
for c, n in concepts.most_common(25):
    print(f"  {n:>5}  {c}")

print("\n20 random rare strings (to judge how messy the naming is):")
rare = [c for c, n in concepts.items() if n == 1]
import random
random.seed(1)
for c in random.sample(rare, min(20, len(rare))):
    print("  ", c)