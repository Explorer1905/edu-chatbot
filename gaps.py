import pickle
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL

GRAPH_FILE = "curriculum_graph.pkl"
DUP_THRESHOLD = 0.90   # concepts this similar are treated as "the same idea"

with open(GRAPH_FILE, "rb") as f:
    _data = pickle.load(f)
GRAPH = _data["graph"]
CONCEPTS = _data["concepts"]

_model = SentenceTransformer(EMBED_MODEL)
_concept_emb = _model.encode(CONCEPTS, normalize_embeddings=True)
_concept_idx = {c: i for i, c in enumerate(CONCEPTS)}


def _similar_concepts(concept, threshold=DUP_THRESHOLD):
    """Concepts (including itself) that are near-duplicates of `concept`."""
    if concept not in _concept_idx:
        return {concept}
    v = _concept_emb[_concept_idx[concept]]
    sims = _concept_emb @ v
    return {CONCEPTS[i] for i, s in enumerate(sims) if s >= threshold}


def find_focus_concepts(question, top_n=2, min_sim=0.35):
    q_emb = _model.encode([question], normalize_embeddings=True)[0]
    sims = _concept_emb @ q_emb
    ranked = sorted(zip(CONCEPTS, sims), key=lambda x: -x[1])[:top_n]
    return [(c, float(s)) for c, s in ranked if s >= min_sim]


def detect_gaps(focus_concepts, known_concepts=None, max_depth=3):
    known = {k.lower().strip() for k in (known_concepts or [])}

    # exclude the focus concepts themselves AND anything near-duplicate to them
    focus_names = {c for c, _ in focus_concepts}
    excluded = set(focus_names)
    for c in focus_names:
        excluded |= _similar_concepts(c)

    gaps = {}
    for concept, _ in focus_concepts:
        if concept not in GRAPH:
            continue
        frontier = [(concept, 0)]
        seen = {concept}
        while frontier:
            node, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for pred in GRAPH.predecessors(node):
                if pred in seen:
                    continue
                seen.add(pred)
                if pred not in known and pred not in excluded:
                    w = GRAPH[pred][node]["weight"]
                    grade = GRAPH.nodes[pred]["grade"]
                    prev = gaps.get(pred)
                    if prev is None or depth + 1 < prev[0]:
                        gaps[pred] = (depth + 1, w, grade)
                frontier.append((pred, depth + 1))

    ranked = sorted(gaps.items(), key=lambda x: (x[1][0], -x[1][1]))
    return [{"concept": c, "depth": d, "weight": w, "grade": g} for c, (d, w, g) in ranked]


if __name__ == "__main__":
    tests = [
        ("Why does gradient descent-like reasoning apply to projectile motion?", []),
        ("What is Ohm's law and why does resistance depend on temperature?", ["basic circuit concepts"]),
        ("Explain electric potential in a capacitor", []),
    ]
    for q, known in tests:
        print(f"\nQ: {q}")
        focus = find_focus_concepts(q)
        print("  Focus concepts:", focus)
        gaps = detect_gaps(focus, known_concepts=known)
        print(f"  Known: {known}")
        print("  Detected gaps (nearest first):")
        for g in gaps[:6]:
            print(f"    - {g['concept']}  (depth={g['depth']}, grade={g['grade']:.0f}, support={g['weight']})")