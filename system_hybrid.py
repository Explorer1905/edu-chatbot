from llm import ask_llm
from retriever import Retriever
from gaps import find_focus_concepts, detect_gaps

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


SYSTEM = """You are a tutor answering questions using NCERT course material.
Use ONLY the provided passages for facts. If prerequisite gaps are listed, briefly
explain the most important one or two before the main answer, in simple language.
Be clear and concise."""


def answer(question, grade=None, subject=None, known_concepts=None, **kwargs):
    r = _get_retriever()
    passages = r.retrieve(question, k=5, mode="hybrid", grade=grade, subject=subject)

    focus = find_focus_concepts(question)
    gaps = detect_gaps(focus, known_concepts=known_concepts or [], max_depth=2)
    top_gaps = gaps[:3]

    context = "\n\n".join(f"[{p['id']}] ({p['subject']} grade {p['grade']}, topic: {p['topic']})\n{p['text']}"
                          for p in passages)
    gap_text = ("Likely prerequisite gaps for this student: " +
               ", ".join(g["concept"] for g in top_gaps)) if top_gaps else "No prerequisite gaps detected."

    user = f"Question: {question}\n\n{gap_text}\n\nCourse material:\n{context}"
    text = ask_llm(SYSTEM, user, max_tokens=1100)
    return {"answer": text, "chunks": passages, "gaps": top_gaps, "focus": focus}