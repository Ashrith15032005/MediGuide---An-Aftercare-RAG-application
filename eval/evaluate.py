"""
Evaluation harness.

    python -m eval.evaluate              # safety/refusal accuracy (offline, free)
    python -m eval.evaluate --retrieval  # + retrieval hit-rate (needs API key and built index)
    python -m eval.evaluate --retrieval --threshold-sweep

Metrics
  Refusal accuracy : for each 'refuse' case the layer must block at the expected
                     level; for each 'answer' case it must NOT block.
  Retrieval hit@k  : for each 'answer' case, at least one retrieved chunk must come
                     from one of the expected sections.
  Threshold sweep  : hit-rate at several relevance thresholds, to choose
                     MIN_RELEVANCE with data instead of a guess.

Add your own cases to eval/eval_set.json (aim for 20-30 per procedure for the report).
"""
import argparse
import json
from pathlib import Path

from src.safety import check_query

CASES = json.loads((Path(__file__).parent / "eval_set.json").read_text(encoding="utf-8"))


def refusal_eval() -> bool:
    ok = 0
    failures = []
    for c in CASES:
        result = check_query(c["question"])
        if c["expect"] == "refuse":
            good = result.blocked and result.level == c["expected_level"]
        else:
            good = not result.blocked
        ok += good
        if not good:
            failures.append((c["question"], c["expect"], result.level))
    print(f"Safety layer: {ok}/{len(CASES)} correct ({ok / len(CASES):.0%})")
    for q, exp, got in failures:
        print(f"  FAIL  {q!r}: expected {exp}, layer said {got}")
    return not failures


def retrieval_eval(sweep: bool) -> None:
    from src.config import TOP_K
    from src.ingest import get_vectorstore
    from src.rag_chain import MediGuideAssistant

    store = get_vectorstore()
    answer_cases = [c for c in CASES if c["expect"] == "answer"]
    thresholds = [0.0, 0.25, 0.35, 0.45, 0.55] if sweep else [None]
    for th in thresholds:
        hits = kept_total = 0
        misses = []
        for c in answer_cases:
            bot = MediGuideAssistant(vectorstore=store, llm=_NoLLM(), k=TOP_K,
                                     min_relevance=th if th is not None else 0.35)
            docs = bot._retrieve({"question": c["question"], "primary": c["procedure"],
                                  "procedures": [c["procedure"]]})
            sections = {d.metadata["section"] for d, _ in docs}
            kept_total += len(docs)
            if sections & set(c["expected_sections"]):
                hits += 1
            else:
                misses.append((c["question"], sorted(sections)))
        label = f"threshold {th}" if th is not None else "default threshold"
        print(f"Retrieval hit@{TOP_K} ({label}): {hits}/{len(answer_cases)} "
              f"({hits / len(answer_cases):.0%}), avg chunks kept {kept_total / len(answer_cases):.1f}")
        if th is None:
            for q, secs in misses:
                print(f"  MISS  {q!r} got {secs}")


class _NoLLM:
    """Retrieval evaluation does not need a generator."""
    def __or__(self, other):
        return other

    def __ror__(self, other):
        return other


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--threshold-sweep", action="store_true")
    args = ap.parse_args()
    refusal_eval()
    if args.retrieval:
        retrieval_eval(args.threshold_sweep)
