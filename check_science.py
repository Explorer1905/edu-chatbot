from datasets import load_dataset

df = load_dataset("KadamParth/Ncert_dataset", split="train").to_pandas()
for subj in ["Science", "Chemistry", "Biology"]:
    sub = df[df["subject"] == subj]
    print(f"\n{subj}: {len(sub)} rows")
    print("  grades:", sub["grade"].value_counts().sort_index().to_dict())
    print("  unique passages:", sub["Explanation"].nunique())
    print("  sample topics:", sub["Topic"].drop_duplicates().head(5).tolist())