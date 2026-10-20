from llm import ask_llm
from retriever import Retriever

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


SYSTEM = """You are a tutor answering questions using NCERT course material.
Use ONLY the provided passages. If they don't contain the answer, say so honestly.
Be clear and concise."""


def answer(question, grade=None, subject=None, **kwargs):
    r = _get_retriever()
    passages = r.retrieve(question, k=5, mode="hybrid", grade=grade, subject=subject)
    context = "\n\n".join(f"[{p['id']}] ({p['subject']} grade {p['grade']}, topic: {p['topic']})\n{p['text']}"
                          for p in passages)
    user = f"Question: {question}\n\nCourse material:\n{context}"
    text = ask_llm(SYSTEM, user, max_tokens=900)
    return {"answer": text, "chunks": passages, "gaps": []}