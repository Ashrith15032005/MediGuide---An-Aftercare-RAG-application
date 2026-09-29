"""
ingest.py - build the vector store.

    python -m src.ingest             # embed chunks and store them in ChromaDB
    python -m src.ingest --dry-run   # only show the chunks (no API key needed)

What happens: chunks (see knowledge_base.py) are turned into embeddings by
Gemini and stored in ChromaDB on disk. An EMBEDDING is a list of numbers that
represents the meaning of a text; texts with similar meaning have vectors that
point in similar directions, which is how retrieval finds relevant passages.

Re-run this whenever you edit a file in data/knowledge_base/.
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, require_api_key
from src.knowledge_base import load_chunks


def get_embeddings():
    """Gemini embedding model through LangChain."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    require_api_key()
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_vectorstore(embeddings=None):
    """Open the persisted Chroma collection (cosine distance)."""
    from langchain_chroma import Chroma
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_index(reset: bool = True) -> int:
    chunks = load_chunks()
    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    store = get_vectorstore()
    ids = [f"{c.metadata['procedure']}-{i}" for i, c in enumerate(chunks)]
    store.add_documents(chunks, ids=ids)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MediGuide vector store")
    parser.add_argument("--dry-run", action="store_true", help="show chunks only")
    args = parser.parse_args()

    chunks = load_chunks()
    counts = Counter((c.metadata["procedure"], c.metadata["kind"]) for c in chunks)
    print(f"{len(chunks)} chunks prepared")
    for (proc, kind), n in sorted(counts.items()):
        print(f"  {proc:24s} {kind:9s} {n}")
    if args.dry_run:
        print("\nSample chunk:\n" + chunks[0].page_content[:400])
        return
    n = build_index()
    print(f"Indexed {n} chunks into {CHROMA_DIR}")


if __name__ == "__main__":
    main()
