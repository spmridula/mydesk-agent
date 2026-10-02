"""Tiny evaluation harness.  Run from project root:  python -m eval.run_eval

Reports: intent routing accuracy, top-source/tool hit rate, average latency.
NOTE: task questions write to the tasks DB; use a throwaway DB:  TASKS_DB=data/eval_tasks.db
"""
import asyncio
import json
import time
from pathlib import Path

from app.graph import run_agent

HERE = Path(__file__).parent


async def main() -> None:
    questions = json.loads((HERE / "questions.json").read_text())
    rows, intent_ok, source_ok, scored = [], 0, 0, 0
    for item in questions:
        start = time.perf_counter()
        result = await run_agent(item["q"])
        latency = time.perf_counter() - start
        got_intent = result.get("intent")
        sources = result.get("sources", [])
        i_ok = got_intent == item["intent"]
        s_ok = None
        if item["source"] is not None:
            s_ok = bool(sources) and sources[0] == item["source"]  # top-ranked source must match
            scored += 1
            source_ok += int(s_ok)
        intent_ok += int(i_ok)
        rows.append(
            {"q": item["q"], "intent": got_intent, "intent_ok": i_ok, "sources": sources,
             "source_ok": s_ok, "latency_s": round(latency, 2), "retries": result.get("rewrites", 0)}
        )
        print(f"{'OK ' if i_ok and s_ok is not False else 'BAD'} {item['q']}  ->  {got_intent} {sources} ({latency:.1f}s)")

    summary = {
        "intent_accuracy": round(intent_ok / len(questions), 2),
        "source_hit_rate": round(source_ok / scored, 2) if scored else None,
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 2),
    }
    print("\nSUMMARY:", summary)
    (HERE / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
