from llm import ask_llm

SYSTEM = """You are a helpful tutor answering a student's question.
Answer from your own general knowledge. Be clear and concise."""


def answer(question, **kwargs):
    text = ask_llm(SYSTEM, question, max_tokens=800)
    return {"answer": text, "chunks": [], "gaps": []}