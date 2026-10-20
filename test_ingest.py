import pandas as pd


corpus = pd.read_csv("science_corpus.csv")
qa = pd.read_csv("science_qa.csv")
print("Passages:", len(corpus), "| QA rows:", len(qa))
print("\nPassages per subject:\n", corpus["subject"].value_counts())
print("\nPassages per grade:\n", corpus["grade"].value_counts().sort_index())

bad = corpus["Explanation"].str.contains("â€|Ã|Î¼", regex=True).sum()
bad_pre = corpus["Prerequisites"].fillna("").str.contains("â€|Ã", regex=True).sum()
print("\nPassages still with garbage chars:", bad)
print("Prerequisite strings still with garbage chars:", bad_pre)

print("\n--- 3 random passages ---")
for _, row in corpus.sample(3, random_state=1).iterrows():
    print(f"[{row['subject']} | grade {row['grade']}] {row['Topic']}")
    print(row["Explanation"])
    print("Prerequisites:", row["Prerequisites"])