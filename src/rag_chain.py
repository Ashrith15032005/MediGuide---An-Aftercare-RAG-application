"""
RAG Chain (Orchestrator)
--------------------------
Core of MediGuide. Flow for every patient question:

  1. Safety check    -> refuse immediately if it's symptom-related
  2. Retrieval        -> top-k relevant chunks for THIS procedure only
  3. Patient context  -> pull in history + prescription for personalization
  4. Generation       -> LLM answers ONLY from retrieved chunks, must cite
  5. Log the turn     -> saved to patient history for future context

Generation uses the Anthropic API (ANTHROPIC_API_KEY env var). Use
use_llm=False to dry-run without an API key — useful for testing
retrieval/safety in isolation.
"""

import os
import re
import pickle

from sklearn.metrics.pairwise import cosine_similarity

from src import config, safety
from src.text_utils import tokenize_and_stem, simple_stem  # noqa: F401 (tokenize_and_stem needed to unpickle vectorizer)
from src.patient_history import PatientHistory

_index_cache = None

# Stemming already unifies word FORMS ("shower"/"showers"). This map
# handles genuine SYNONYMS, which stemming can't fix — a known,
# documented limitation of lexical (TF-IDF) retrieval. A production
# version would use sentence-transformers or an API embedding model.
QUERY_SYNONYMS = {
    "eat": ["food", "diet", "meal"],
    "food": ["eat", "diet", "meal"],
    "drink": ["fluid", "water", "hydration"],
    "walk": ["activity", "exercise", "movement"],
    "exercise": ["activity", "movement", "walk", "gym"],
    "gym": ["exercise", "activity", "strenuous"],
    "sleep": ["rest"],
    "medicine": ["medication", "drug", "tablet", "dose"],
    "medicin": ["medication", "drug", "tablet", "dose"],  # naive stemmer: "medicines" -> "medicin"
    "wound": ["incision", "cut", "stitches"],
    "shower": ["bath", "wash"],
}
_QUERY_SYNONYMS_BY_STEM = {simple_stem(k): v for k, v in QUERY_SYNONYMS.items()}


def _expand_query(query: str) -> str:
    words = re.findall(r"[a-zA-Z]+", query.lower())
    extra = []
    for w in words:
        extra.extend(_QUERY_SYNONYMS_BY_STEM.get(simple_stem(w), []))
    return query + " " + " ".join(extra)


def _load_index():
    global _index_cache
    if _index_cache is None:
        with open(config.INDEX_PATH, "rb") as f:
            _index_cache = pickle.load(f)
    return _index_cache


def list_procedures():
    idx = _load_index()
    return sorted(set(c["procedure"] for c in idx["chunks"]))


def retrieve(query: str, procedure: str, top_k: int = config.TOP_K, min_score: float = config.MIN_SCORE):
    idx = _load_index()
    vectorizer, matrix, chunks = idx["vectorizer"], idx["matrix"], idx["chunks"]

    query_vec = vectorizer.transform([_expand_query(query)])
    sims = cosine_similarity(query_vec, matrix)[0]

    scored = [(sims[i], chunks[i]) for i in range(len(chunks)) if chunks[i]["procedure"] == procedure]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(s, c) for s, c in scored[:top_k] if s >= min_score]


def _build_prompt(query: str, retrieved_chunks, patient_context: str) -> str:
    context_text = "\n\n".join(f"[Section: {c['section']}]\n{c['text']}" for _, c in retrieved_chunks)
    return f"""PATIENT CONTEXT:
{patient_context}

CONTEXT (from verified aftercare knowledge base):
{context_text}

PATIENT QUESTION: {query}

Answer the patient's question using only the CONTEXT above, personalized using the PATIENT CONTEXT where relevant."""


def _call_llm(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        system=config.SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def answer_query(patient_id: str, procedure: str, query: str, use_llm: bool = True):
    history = PatientHistory(patient_id)
    if history.data.get("procedure") != procedure:
        history.set_procedure(procedure)

    # Step 1: Safety check
    refuse, message = safety.check(query)
    if refuse:
        history.add_turn(query, message, refused=True)
        return {"refused": True, "answer": message, "sources": []}

    # Step 2: Retrieval (procedure-scoped)
    retrieved = retrieve(query, procedure)
    if not retrieved:
        fallback = ("I don't have enough verified information to answer that "
                     "confidently. Please check with your doctor or hospital.")
        history.add_turn(query, fallback, refused=False)
        return {"refused": False, "answer": fallback, "sources": []}

    # Step 3: Patient context (history + prescription) for personalization
    patient_context = history.get_context_summary()

    # Step 4: Generation
    prompt = _build_prompt(query, retrieved, patient_context)
    answer = _call_llm(prompt) if use_llm else "[DRY RUN — no LLM call made]\n\n" + prompt

    sources = [c["section"] for _, c in retrieved]
    history.add_turn(query, answer, refused=False)
    return {"refused": False, "answer": answer, "sources": sources}
