"""
Ingestion Pipeline
------------------
Reads each procedure's markdown knowledge base file, chunks it by ## section,
and builds a TF-IDF retrieval index (data/index.pkl).

Run this once, and again any time you edit files in data/knowledge_base/:
    python -m src.ingest
"""

import os
import re
import pickle
from dataclasses import dataclass, asdict
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config
from src.text_utils import tokenize_and_stem, STEMMED_STOP_WORDS


@dataclass
class Chunk:
    chunk_id: str
    procedure: str
    section: str
    text: str


def chunk_markdown(filepath: str, procedure: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    parts = re.split(r"\n(?=## )", content)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("# ") and "\n## " not in part:
            continue  # skip a title-only fragment
        header_match = re.match(r"##\s+(.+)", part)
        section = header_match.group(1).strip() if header_match else "Overview"
        chunks.append(Chunk(
            chunk_id=f"{procedure}::{section}",
            procedure=procedure,
            section=section,
            text=part,
        ))
    return chunks


def build_index():
    all_chunks = []
    for fname in sorted(os.listdir(config.KB_DIR)):
        if not fname.endswith(".md"):
            continue
        procedure = fname.replace(".md", "")
        chunks = chunk_markdown(os.path.join(config.KB_DIR, fname), procedure)
        all_chunks.extend(chunks)

    texts = [c.text for c in all_chunks]
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_and_stem,
        token_pattern=None,
        stop_words=STEMMED_STOP_WORDS,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(texts)

    os.makedirs(os.path.dirname(config.INDEX_PATH), exist_ok=True)
    with open(config.INDEX_PATH, "wb") as f:
        pickle.dump({
            "chunks": [asdict(c) for c in all_chunks],
            "vectorizer": vectorizer,
            "matrix": matrix,
        }, f)

    print(f"Indexed {len(all_chunks)} chunks from "
          f"{len(set(c.procedure for c in all_chunks))} procedures.")
    for c in all_chunks:
        print(f"  - [{c.procedure}] {c.section}")


if __name__ == "__main__":
    build_index()
