# quick_demo.py
import system_llm_only, system_plain_rag, system_hybrid

q = "Who discovered the electron?"
print("="*60)
for name, mod in [("LLM-only (no grounding)", system_llm_only),
                   ("Plain RAG", system_plain_rag),
                   ("Hybrid (RAG+OKF)", system_hybrid)]:
    r = mod.answer(q, grade=9, subject="Science")
    print(f"\n--- {name} ---")
    print(r["answer"][:300])
    if r.get("gaps"):
        print("Gaps flagged:", [g["concept"] for g in r["gaps"]])
print("="*60)