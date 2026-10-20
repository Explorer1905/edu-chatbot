import re
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from config import *

MIN_CONCEPT_COUNT = 2
MATCH_THRESHOLD = 0.72
MIN_EDGE_WEIGHT = 2
GRAPH_FILE = "curriculum_graph.pkl"

corpus = pd.read_csv(CORPUS_CSV)
corpus["Prerequisites"] = corpus["Prerequisites"].fillna("")


def norm(s):
    """Lowercase, fix a few known glued-word glitches, collapse spaces."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_prereqs(s):
    s = s.strip()
    if not s or s.lower().startswith("no prerequisite"):
        return []
    out = []
    for p in s.split(","):
        p = re.sub(r"^\s*(and|or)\s+", "", p.strip(), flags=re.I).strip()
        p = norm(p)
        if len(p) > 2:
            out.append(p)
    return out


corpus["prereq_list"] = corpus["Prerequisites"].map(parse_prereqs)

# ---------- 1. Concept vocabulary ----------
concept_count = Counter()
concept_mention_grades = defaultdict(list)
concept_subjects = defaultdict(Counter)
for r in corpus.itertuples():
    for c in r.prereq_list:
        concept_count[c] += 1
        concept_mention_grades[c].append(int(r.grade))
        concept_subjects[c][r.subject] += 1

concepts = [c for c, n in concept_count.items() if n >= MIN_CONCEPT_COUNT]

# merge glued-word variants (e.g. 'electromagneticinduction' == 'electromagnetic induction')
squash = defaultdict(list)
for c in concepts:
    squash[c.replace(" ", "")].append(c)
merge_map = {}
for key, variants in squash.items():
    canonical = max(variants, key=lambda v: (concept_count[v], v.count(" ")))
    for v in variants:
        merge_map[v] = canonical
concepts = sorted(set(merge_map.values()))
for r_idx in corpus.index:
    corpus.at[r_idx, "prereq_list"] = sorted({merge_map.get(c, c) for c in corpus.at[r_idx, "prereq_list"]
                                              if merge_map.get(c, c) in set(concepts)})
print(f"{len(concepts)} concepts after merging spelling variants")

# ---------- 2. Map each passage topic to its nearest concept ----------
model = SentenceTransformer(EMBED_MODEL)
concept_emb = model.encode(concepts, normalize_embeddings=True, batch_size=64)

topics = corpus["Topic"].fillna("").str.lower().str.strip().tolist()
unique_topics = sorted(set(topics))
topic_emb = model.encode(unique_topics, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
sims = topic_emb @ concept_emb.T
best_idx, best_sim = sims.argmax(axis=1), sims.max(axis=1)
topic_to_concept = {t: (concepts[i], float(s))
                    for t, i, s in zip(unique_topics, best_idx, best_sim) if s >= MATCH_THRESHOLD}
mapped = sum(1 for t in topics if t in topic_to_concept)
print(f"{mapped}/{len(topics)} passages mapped ({100 * mapped / len(topics):.1f}%)")

# ---------- 3. Grade where each concept is TAUGHT (passages whose topic maps to it) ----------
taught_grades = defaultdict(list)
for r, t in zip(corpus.itertuples(), topics):
    if t in topic_to_concept:
        taught_grades[topic_to_concept[t][0]].append(int(r.grade))

# ---------- 4. Weighted prerequisite edges ----------
edge_weight = Counter()
for r, t in zip(corpus.itertuples(), topics):
    if t not in topic_to_concept:
        continue
    child = topic_to_concept[t][0]
    for p in r.prereq_list:
        if p != child:
            edge_weight[(p, child)] += 1

G = nx.DiGraph()
for c in concepts:
    if taught_grades[c]:
        grade, source = float(np.median(taught_grades[c])), "taught"
    else:
        grade, source = float(np.median(concept_mention_grades[c])), "mentioned"
    G.add_node(c, grade=grade, grade_source=source, mentions=concept_count[c],
               taught_passages=len(taught_grades[c]),
               subject=concept_subjects[c].most_common(1)[0][0])
for (p, c), w in edge_weight.items():
    if w >= MIN_EDGE_WEIGHT and p in G and c in G:
        G.add_edge(p, c, weight=w)

# ---------- 5. Make it a DAG ----------
to_remove = set()
for u, v in G.edges():
    if G.has_edge(v, u) and (v, u) not in to_remove and (u, v) not in to_remove:
        to_remove.add((v, u) if G[u][v]["weight"] >= G[v][u]["weight"] else (u, v))
G.remove_edges_from(to_remove)
n_cyc = 0
while not nx.is_directed_acyclic_graph(G):
    cyc = nx.find_cycle(G)
    lightest = min(cyc, key=lambda e: G[e[0]][e[1]]["weight"])
    G.remove_edge(lightest[0], lightest[1])
    n_cyc += 1
print(f"Removed {len(to_remove)} two-cycle edges and {n_cyc} longer-cycle edges")

print(f"\nGraph: {G.number_of_nodes()} concepts, {G.number_of_edges()} edges, DAG={nx.is_directed_acyclic_graph(G)}")
n_taught = sum(1 for _, d in G.nodes(data=True) if d["grade_source"] == "taught")
print(f"Concept grade known from teaching passages: {n_taught}/{G.number_of_nodes()}")

# ---------- 6. Curriculum-consistency check (only on edges where both grades are 'taught') ----------
ok = same = bad = skipped = 0
for u, v in G.edges():
    if G.nodes[u]["grade_source"] != "taught" or G.nodes[v]["grade_source"] != "taught":
        skipped += 1
        continue
    gu, gv = G.nodes[u]["grade"], G.nodes[v]["grade"]
    if gu < gv: ok += 1
    elif gu == gv: same += 1
    else: bad += 1
tot = max(1, ok + same + bad)
print(f"\nGrade-ordering check ({tot} edges with both grades known, {skipped} skipped):")
print(f"  prerequisite taught in an EARLIER class : {ok} ({100 * ok / tot:.1f}%)")
print(f"  taught in the SAME class                : {same} ({100 * same / tot:.1f}%)")
print(f"  taught in a LATER class (inconsistent)  : {bad} ({100 * bad / tot:.1f}%)")
print(f"  => consistent with curriculum order     : {100 * (ok + same) / tot:.1f}%")

with open(GRAPH_FILE, "wb") as f:
    pickle.dump({"graph": G, "topic_to_concept": topic_to_concept, "concepts": concepts}, f)
print(f"\nSaved {GRAPH_FILE}")

print("\nTop 15 heaviest edges:")
for u, v, d in sorted(G.edges(data=True), key=lambda x: -x[2]["weight"])[:15]:
    print(f"  {u}  -->  {v}   (w={d['weight']}, taught grade {G.nodes[u]['grade']:.0f} -> {G.nodes[v]['grade']:.0f})")