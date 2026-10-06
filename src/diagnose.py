"""
diagnose.py - find out WHY MediGuide cannot reach the model.

    python -m src.diagnose

Tests, one at a time: API key present -> embedding call -> language model call
-> vector store has data. The first line that says FAIL shows the real cause
(bad key, quota exceeded, wrong model name, index not built...).

    python -m src.diagnose --models           # which model names your key can use
    python -m src.diagnose --models --test    # ...and which of them answer right now

    python -m src.diagnose --query "when should I take my medicines?" [--procedure general_recovery]

shows the chunks retrieval returns, their scores, and whether each passes the threshold.
"""
from __future__ import annotations

import sys

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL, require_api_key


def _step(name: str, fn) -> bool:
    try:
        detail = fn()
        print(f"OK    {name}" + (f"  ({detail})" if detail else ""))
        return True
    except Exception as exc:  # we want the raw reason, whatever it is
        print(f"FAIL  {name}\n      {type(exc).__name__}: {str(exc)[:600]}")
        return False


def show_query(question: str, procedure: str) -> None:
    """Print what retrieval really returns, with scores, so MIN_RELEVANCE can be judged."""
    from collections import Counter
    from src.config import MIN_RELEVANCE, TOP_K
    from src.ingest import get_vectorstore
    from src.procedures import label_for
    store = get_vectorstore()
    metas = store._collection.get(include=["metadatas"])["metadatas"]
    print("Chunks in index per guide:", dict(Counter(m["procedure"] for m in metas)))
    query = f"{label_for(procedure)}: {question}"
    print(f"\nQuery: {query!r}\nThreshold MIN_RELEVANCE = {MIN_RELEVANCE}\n")
    for flt, title in (({"$and": [{"kind": "guidance"}, {"procedure": {"$in": [procedure]}}]}, f"Only guide '{procedure}'"),
                       (None, "Whole index (no filter)")):
        print(title)
        for doc, score in store.similarity_search_with_relevance_scores(query, k=TOP_K, filter=flt):
            mark = "KEEP" if score >= MIN_RELEVANCE else "drop"
            print(f"  {mark} {score:.3f}  {doc.metadata['procedure']} > {doc.metadata['section']}")
        print()


def list_models() -> None:
    """Models THIS key can call for text generation (use one in MEDIGUIDE_LLM_MODEL)."""
    from google import genai
    client = genai.Client(api_key=require_api_key())
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            names.append(m.name.removeprefix("models/"))
    print("Models available to your key for generateContent:")
    for n in sorted(names):
        print("  ", n)
    if "--test" in sys.argv:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("\nTrying each flash model with one tiny request (takes a minute)...")
        working = []
        for n in sorted(x for x in names if "flash" in x and not any(t in x for t in ("image", "tts", "live", "audio", "embedding"))):
            try:
                ChatGoogleGenerativeAI(model=n, max_retries=1, temperature=0).invoke("Reply with: ok")
                print(f"  OK    {n}")
                working.append(n)
            except Exception as exc:
                print(f"  FAIL  {n}  ({type(exc).__name__}: {str(exc)[:70]})")
        if working:
            print(f"\nPut this in .env:  MEDIGUIDE_LLM_MODEL={working[0]}")
            if len(working) > 1:
                print(f"and this:          MEDIGUIDE_FALLBACK_MODELS={','.join(working[1:4])}")
        return
    print("\nPick a 'flash' one and put it in .env as:  MEDIGUIDE_LLM_MODEL=<name>")


def main() -> int:
    if "--models" in sys.argv:
        list_models()
        return 0
    if "--query" in sys.argv:
        i = sys.argv.index("--query")
        proc = sys.argv[sys.argv.index("--procedure") + 1] if "--procedure" in sys.argv else "general_recovery"
        require_api_key()
        show_query(sys.argv[i + 1], proc)
        return 0
    print(f"LLM model: {LLM_MODEL} | embedding model: {EMBEDDING_MODEL}\n")
    if not _step("API key found", lambda: f"{require_api_key()[:6]}...") :
        return 1

    def embed():
        from src.ingest import get_embeddings
        return f"{len(get_embeddings().embed_query('post surgery diet'))} dimensions"

    def llm():
        from src.rag_chain import get_llm
        return repr(get_llm().invoke("Reply with the single word: ready").content)[:60]

    def index():
        from src.ingest import get_vectorstore
        n = get_vectorstore()._collection.count()
        if n == 0:
            raise RuntimeError(f"collection '{COLLECTION_NAME}' in {CHROMA_DIR} is empty - run: python -m src.ingest")
        # Every guide (including general_recovery) must be in the index, or its questions find nothing.
        from src.procedures import PROCEDURES
        store = get_vectorstore()
        missing = [k for k in PROCEDURES if not store._collection.get(where={"procedure": k}, limit=1)["ids"]]
        if missing:
            raise RuntimeError(f"index has no chunks for: {', '.join(missing)} - run: python -m src.ingest")
        return f"{n} chunks, all {len(PROCEDURES)} guides present"

    results = [_step("Embedding call", embed), _step("Language model call", llm), _step("Vector store", index)]
    print("\nAll good." if all(results) else "\nFix the first FAIL above, then retry.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())