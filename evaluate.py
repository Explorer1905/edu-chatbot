# evaluate.py
import json
import time
import random
import pandas as pd
from llm import ask_llm_json
import system_llm_only, system_plain_rag, system_hybrid
from config import QA_CSV

N_QUESTIONS = 30          # start small; raise to 150-300 for your final run
SEED = 7
OUT_FILE = "eval_results.json"

SYSTEMS = {
    "llm_only": system_llm_only,
    "plain_rag": system_plain_rag,
    "hybrid": system_hybrid,
}

JUDGE_SYSTEM = """You are a strict grading assistant for a physics/science tutor.
Compare the model's answer to the reference answer for the same question.
Return JSON only: {"correctness": 1-5, "groundedness": 1-5, "reason": "short phrase"}
correctness: does the answer convey the same key facts as the reference?
groundedness: does the answer avoid making up facts not supported by course-level material?"""


def judge(question, reference, model_answer):
    user = f"Question: {question}\nReference answer: {reference}\nModel answer: {model_answer}"
    try:
        return ask_llm_json(JUDGE_SYSTEM, user, max_tokens=200)
    except Exception as e:
        print("  [judge failed]", e)
        return {"correctness": None, "groundedness": None, "reason": "judge_error"}


def gap_metrics(detected_gaps, gold_prereq_string):
    gold = set()
    if isinstance(gold_prereq_string, str) and not gold_prereq_string.lower().startswith("no prerequisite"):
        gold = {p.strip().lower() for p in gold_prereq_string.split(",") if p.strip()}
    found = {g["concept"].lower() for g in detected_gaps}
    if not gold and not found:
        return None  # nothing to score
    tp = len(gold & found)
    precision = tp / len(found) if found else 0.0
    recall = tp / len(gold) if gold else 0.0
    return {"precision": precision, "recall": recall, "gold": list(gold), "found": list(found)}


def main():
    qa = pd.read_csv(QA_CSV)
    qa = qa.dropna(subset=["Question", "Answer"])
    random.seed(SEED)
    sample = qa.sample(min(N_QUESTIONS, len(qa)), random_state=SEED).reset_index(drop=True)
    print(f"Evaluating {len(sample)} questions across {len(SYSTEMS)} systems...\n")

    rows = []
    for idx, row in sample.iterrows():
        print(f"[{idx + 1}/{len(sample)}] {row['Question'][:70]}...")
        record = {"question": row["Question"], "reference": row["Answer"],
                  "subject": row["subject"], "grade": int(row["grade"]),
                  "gold_prereqs": row["Prerequisites"]}

        for name, mod in SYSTEMS.items():
            t0 = time.time()
            try:
                out = mod.answer(row["Question"], grade=int(row["grade"]), subject=row["subject"])
            except Exception as e:
                print(f"   [{name} FAILED] {e}")
                record[name] = {"answer": None, "error": str(e)}
                continue
            elapsed = time.time() - t0

            j = judge(row["Question"], row["Answer"], out["answer"])
            gm = gap_metrics(out.get("gaps", []), row["Prerequisites"]) if name == "hybrid" else None

            record[name] = {
                "answer": out["answer"], "latency": elapsed,
                "correctness": j["correctness"], "groundedness": j["groundedness"],
                "gap_metrics": gm,
            }
            print(f"   {name:<10} correctness={j['correctness']}  groundedness={j['groundedness']}  "
                  f"({elapsed:.1f}s)")

        rows.append(record)

        # save incrementally so a crash doesn't lose progress
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    summarize(rows)


def summarize(rows):
    print("\n\n===== SUMMARY =====")
    for name in SYSTEMS:
        corr = [r[name]["correctness"] for r in rows if r.get(name, {}).get("correctness") is not None]
        ground = [r[name]["groundedness"] for r in rows if r.get(name, {}).get("groundedness") is not None]
        lat = [r[name]["latency"] for r in rows if r.get(name, {}).get("latency") is not None]
        n = len(corr)
        print(f"\n{name}  (n={n})")
        print(f"  avg correctness   : {sum(corr) / n:.2f}" if n else "  no data")
        print(f"  avg groundedness  : {sum(ground) / len(ground):.2f}" if ground else "  no data")
        print(f"  avg latency       : {sum(lat) / len(lat):.1f}s" if lat else "  no data")

    gm = [r["hybrid"]["gap_metrics"] for r in rows
          if r.get("hybrid", {}).get("gap_metrics") is not None]
    if gm:
        p = sum(g["precision"] for g in gm) / len(gm)
        rc = sum(g["recall"] for g in gm) / len(gm)
        print(f"\nGap detection vs dataset prerequisite tags (n={len(gm)}):")
        print(f"  precision: {p:.2f}   recall: {rc:.2f}")
    print(f"\nFull results saved to {OUT_FILE}")


if __name__ == "__main__":
    main()