"""
Safety / Refusal Layer
-----------------------
Runs BEFORE any query reaches the RAG pipeline. If a query looks like it
describes a symptom or complication, MediGuide must NOT attempt to answer
it from the knowledge base — it must redirect the patient to a doctor.

Deliberately rule-based (not an LLM call): fast, deterministic, auditable,
and easy to defend in a viva. A learned classifier is future work.
Patterns are loaded from data/red_flags.json so they can be edited without
touching code.
"""

import re
import json

from src import config

with open(config.RED_FLAGS_PATH, "r", encoding="utf-8") as f:
    _RED_FLAGS = json.load(f)

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _RED_FLAGS["patterns"]]
REFUSAL_MESSAGE = _RED_FLAGS["refusal_message"]


def is_symptom_query(query: str) -> bool:
    """Return True if the query matches a red-flag symptom pattern."""
    return any(pattern.search(query) for pattern in _COMPILED_PATTERNS)


def check(query: str):
    """Returns (should_refuse: bool, message_or_none: str | None)."""
    if is_symptom_query(query):
        return True, REFUSAL_MESSAGE
    return False, None


if __name__ == "__main__":
    tests = [
        "can I eat rice after my surgery",
        "I have a fever and the wound looks red",
        "how much should I walk today",
        "there is pus coming out of the incision",
        "when can I sleep on my stomach",
    ]
    for t in tests:
        refuse, _ = check(t)
        print(f"[{'REFUSE' if refuse else 'ANSWER'}] {t}")
