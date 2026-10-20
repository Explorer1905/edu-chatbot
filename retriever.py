import re
import pickle
import os
import numpy as np
import pandas as pd
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from config import *

BM25_CACHE = "bm25.pkl"

_STOP = set("""a an the of in on at to for and or is are was were be been being this that these those
it its as by with from what which who whom how why when where do does did can could should would
has have had not no into than then so such""".split())


def tokenize(text):
    toks = re.findall(r"[a-z0-9']+", str(text).lower())
    return [t for t in toks if t not in _STOP and len(t) > 1]


class Retriever:
    def __init__(self):
        self.corpus = pd.read_csv(CORPUS_CSV)
        self.model = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.col = client.get_collection(COLLECTION_NAME)

        if os.path.exists(BM25_CACHE):
            with open(BM25_CACHE, "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            print("Building BM25 index (one time)...")
            docs = (self.corpus["Topic"].fillna("") + " " + self.corpus["Explanation"].fillna(""))
            self.bm25 = BM25Okapi([tokenize(d) for d in docs])
            with open(BM25_CACHE, "wb") as f:
                pickle.dump(self.bm25, f)

        self.ids = self.corpus["id"].tolist()
        self.id_to_row = {pid: i for i, pid in enumerate(self.ids)}

    # ---------- individual retrievers: each returns a ranked list of passage ids ----------
    def dense(self, query, k=20, where=None):
        emb = self.model.encode([query], normalize_embeddings=True).tolist()
        kwargs = {"query_embeddings": emb, "n_results": k}
        if where:
            kwargs["where"] = where
        res = self.col.query(**kwargs)
        return res["ids"][0]

    def sparse(self, query, k=20, grade=None, subject=None):
        scores = self.bm25.get_scores(tokenize(query))
        if grade is not None or subject is not None:
            mask = np.ones(len(scores), dtype=bool)
            if grade is not None:
                mask &= (self.corpus["grade"].values == grade)
            if subject is not None:
                mask &= (self.corpus["subject"].values == subject)
            scores = np.where(mask, scores, -1)
        top = np.argsort(scores)[::-1][:k]
        return [self.ids[i] for i in top if scores[i] > 0]

    # ---------- fusion ----------
    @staticmethod
    def rrf(rank_lists, k=60):
        """Reciprocal Rank Fusion: score = sum 1/(k + rank)."""
        scores = {}
        for lst in rank_lists:
            for rank, pid in enumerate(lst):
                scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        return [pid for pid, _ in sorted(scores.items(), key=lambda x: -x[1])]

    def hybrid(self, query, k=20, where=None, grade=None, subject=None):
        d = self.dense(query, k=k, where=where)
        s = self.sparse(query, k=k, grade=grade, subject=subject)
        return self.rrf([d, s])

    # ---------- public: return full passage dicts ----------
    def retrieve(self, query, k=TOP_K, mode="hybrid", grade=None, subject=None):
        where = None
        conds = []
        if grade is not None:
            conds.append({"grade": grade})
        if subject is not None:
            conds.append({"subject": subject})
        if len(conds) == 1:
            where = conds[0]
        elif len(conds) > 1:
            where = {"$and": conds}

        if mode == "dense":
            ids = self.dense(query, k=k, where=where)
        elif mode == "bm25":
            ids = self.sparse(query, k=k, grade=grade, subject=subject)
        else:
            ids = self.hybrid(query, k=max(k, 20), where=where, grade=grade, subject=subject)

        out = []
        for pid in ids[:k]:
            r = self.corpus.iloc[self.id_to_row[pid]]
            out.append({
                "id": pid, "topic": r["Topic"], "text": r["Explanation"],
                "grade": int(r["grade"]), "subject": r["subject"],
                "prereqs": r["Prerequisites"],
            })
        return out


if __name__ == "__main__":
    r = Retriever()
    q = "Why does a small force on a small piston lift a heavy load?"
    for mode in ["dense", "bm25", "hybrid"]:
        print(f"\n=== {mode} ===")
        for p in r.retrieve(q, k=3, mode=mode):
            print(f"[{p['subject']} g{p['grade']}] {p['topic']}")