import re
import pandas as pd
import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config import *


def fix_encoding(s):
    """Repair mojibake caused by UTF-8 bytes decoded as Windows-1252."""
    if not isinstance(s, str):
        return ""
    for _ in range(2):
        try:
            s = s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
    for bad, good in {"â€™": "'", "â€˜": "'", "â€œ": '"', "â€": '"',
                      "Ã—": "×", "Î¼": "μ", "Â°": "°", "Â": ""}.items():
        s = s.replace(bad, good)
    return re.sub(r"\s+", " ", s).strip()


def main():
    print("Loading dataset...")
    df = load_dataset(HF_DATASET, split="train").to_pandas()
    df = df[df["subject"].isin(SUBJECTS) & df["grade"].isin(GRADES)].copy()
    print(f"{len(df)} rows selected from subjects {SUBJECTS}")

    for col in ["Topic", "Explanation", "Question", "Answer", "Prerequisites"]:
        df[col] = df[col].map(fix_encoding)

    qa = df[["subject", "grade", "Topic", "Explanation", "Question", "Answer",
             "Difficulty", "QuestionType", "Prerequisites"]].reset_index(drop=True)
    qa.to_csv(QA_CSV, index=False)

    passages = (
        df.groupby(["subject", "grade", "Explanation"], as_index=False)
          .agg(Topic=("Topic", "first"),
               Prerequisites=("Prerequisites", "first"))
    )
    passages = passages[passages["Explanation"].str.len() > 40].reset_index(drop=True)
    passages["id"] = ["p_" + str(i) for i in passages.index]
    print(f"\n{len(passages)} unique passages in total")
    print(passages.groupby(["subject", "grade"]).size().to_string())

    passages.to_csv(CORPUS_CSV, index=False)

    # Embed with topic prepended so topic words help retrieval
    texts = (passages["Topic"] + ". " + passages["Explanation"]).tolist()
    model = SentenceTransformer(EMBED_MODEL)
    print("Embedding...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    B = 500
    for i in tqdm(range(0, len(passages), B), desc="Indexing"):
        chunk = passages.iloc[i:i + B]
        col.add(
            ids=chunk["id"].tolist(),
            documents=chunk["Explanation"].tolist(),
            embeddings=embeddings[i:i + B],
            metadatas=[{"topic": t, "grade": int(g), "subject": s, "prereqs": p}
                       for t, g, s, p in zip(chunk["Topic"], chunk["grade"],
                                             chunk["subject"], chunk["Prerequisites"])],
        )
    print(f"\nDone. {len(passages)} passages indexed.")
    print(f"Saved: {CORPUS_CSV}, {QA_CSV}, vector index in {CHROMA_DIR}/")


if __name__ == "__main__":
    main()