from flask import Flask, request, jsonify, send_from_directory
import system_hybrid
from gaps import GRAPH

app = Flask(__name__, static_folder="static", static_url_path="")


@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    grade = data.get("grade") or None
    subject = data.get("subject") or None
    known = data.get("known_concepts") or []

    if not question:
        return jsonify({"error": "Question is empty"}), 400

    try:
        grade = int(grade) if grade else None
    except ValueError:
        grade = None

    try:
        result = system_hybrid.answer(
            question, grade=grade, subject=subject, known_concepts=known
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    chunks = [
        {"id": c["id"], "topic": c["topic"], "grade": c["grade"], "subject": c["subject"]}
        for c in result.get("chunks", [])
    ]
    gaps = [g["concept"] for g in result.get("gaps", [])]

    # Build graph nodes/edges: question (center) -> gap concepts, question -> source topics
    nodes = [{"id": "q", "label": "Your question", "type": "question"}]
    edges = []

    seen_gap = set()
    for g in gaps:
        if g in seen_gap:
            continue
        seen_gap.add(g)
        node_id = f"gap:{g}"
        nodes.append({"id": node_id, "label": g, "type": "gap"})
        edges.append({"source": "q", "target": node_id})

    seen_src = set()
    for c in chunks:
        if c["topic"] in seen_src:
            continue
        seen_src.add(c["topic"])
        node_id = f"src:{c['topic']}"
        nodes.append({
            "id": node_id, "label": c["topic"], "type": "source",
            "grade": c["grade"], "subject": c["subject"],
        })
        edges.append({"source": "q", "target": node_id})

    return jsonify({
        "answer": result["answer"],
        "gaps": gaps,
        "sources": chunks,
        "detected_grade": chunks[0]["grade"] if chunks else None,
        "detected_subject": chunks[0]["subject"] if chunks else None,
        "graph": {"nodes": nodes, "edges": edges},
    })


@app.route("/expand_concept", methods=["POST"])
def expand_concept():
    data = request.get_json(force=True)
    concept = (data.get("concept") or "").strip().lower()

    if concept not in GRAPH:
        return jsonify({"nodes": [], "edges": []})

    nodes = [{"id": f"gap:{concept}", "label": concept, "type": "gap"}]
    edges = []

    for predecessor in list(GRAPH.predecessors(concept))[:6]:
        node_id = f"gap:{predecessor}"
        nodes.append({"id": node_id, "label": predecessor, "type": "gap"})
        edges.append({"source": f"gap:{concept}", "target": node_id})

    return jsonify({"nodes": nodes, "edges": edges})

if __name__ == "__main__":
    app.run(debug=True, port=5000)