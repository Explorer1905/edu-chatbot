from datasets import load_dataset
import pandas as pd
from config import *

print("Downloading dataset (first time takes a while)...")
ds = load_dataset(HF_DATASET, split="train")
df = ds.to_pandas()

print("\nColumns:", list(df.columns))
print("Total rows:", len(df))
print("\nSubjects:\n", df["subject"].value_counts())

selected = df[df["subject"].isin(SUBJECTS) & df["grade"].isin(GRADES)]
print(f"\nSelected rows: {len(selected)}")
print("Selected by subject and grade:\n", selected.groupby(["subject", "grade"]).size())

uniq = selected.drop_duplicates(subset=["subject", "grade", "Explanation"])
print(f"\nUnique explanation passages: {len(uniq)}")
print("Unique topics:", selected["Topic"].nunique())

print("\nSample prerequisites:")
print(selected["Prerequisites"].value_counts().head(15))