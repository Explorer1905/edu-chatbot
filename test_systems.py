# test_systems.py
import system_llm_only, system_plain_rag, system_hybrid

q = "Why does resistance in a wire increase with temperature?"
for name, mod in [("LLM-only", system_llm_only), ("Plain RAG", system_plain_rag), ("Hybrid", system_hybrid)]:
    print(f"\n===== {name} =====")
    r = mod.answer(q, grade=11, subject="Physics", known_concepts=["basic circuit concepts"])
    print(r["answer"][:400])
    if r["gaps"]:
        print("Gaps:", [g["concept"] for g in r["gaps"]])