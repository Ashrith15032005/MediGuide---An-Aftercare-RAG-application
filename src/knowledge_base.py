"""
knowledge_base.py - loading the curated markdown files and splitting them
into chunks.

TWO-STAGE CHUNKING (a standard RAG pattern):

  Stage 1  MarkdownHeaderTextSplitter
           Splits each file at its headings ("## Diet", "## Wound Care" ...).
           Every chunk therefore stays inside ONE topic, and the heading is
           kept as metadata. This is "structure-aware" splitting.

  Stage 2  RecursiveCharacterTextSplitter
           Any section longer than CHUNK_SIZE is cut again, preferring to
           break at paragraph, then sentence, then word boundaries, with a
           small overlap so no sentence loses its context.

Each chunk also gets a short prefix ("Procedure | Section") so the embedding
captures WHAT the text is about, and so citations can name the section.

WARNING SIGNS are deliberately kept OUT of retrieval (metadata kind =
"warning"). The assistant never interprets symptoms; instead the UI shows
the warning-sign list as fixed text, straight from the file.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, KB_DIR
from src.procedures import PROCEDURES

_HEADERS = [("#", "title"), ("##", "section")]


def kb_path(procedure_key: str):
    return KB_DIR / PROCEDURES[procedure_key].kb_file


def read_kb(procedure_key: str) -> str:
    return kb_path(procedure_key).read_text(encoding="utf-8")


def warning_signs(procedure_key: str) -> list[str]:
    """Bullet points of the '## Warning Signs' section, for static display."""
    signs, in_section = [], False
    for line in read_kb(procedure_key).splitlines():
        if line.startswith("## "):
            in_section = line[3:].strip().lower().startswith("warning")
            continue
        if in_section and line.lstrip().startswith("- "):
            signs.append(line.lstrip()[2:].strip())
    return signs


def load_chunks(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    """Read every knowledge-base file and return ready-to-embed chunks."""
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=True)
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks: list[Document] = []
    for key, proc in PROCEDURES.items():
        path = kb_path(key)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base file missing: {path}")
        # Stage 1: split by heading
        sections = header_splitter.split_text(path.read_text(encoding="utf-8"))
        # Stage 2: split long sections
        for piece in char_splitter.split_documents(sections):
            title = piece.metadata.get("title", proc.label)
            section = piece.metadata.get("section", "Overview")
            kind = "warning" if section.lower().startswith("warning") else "guidance"
            piece.page_content = f"{title} | {section}\n{piece.page_content.strip()}"
            piece.metadata = {
                "procedure": key,
                "title": title,
                "section": section,
                "kind": kind,
                "source": proc.kb_file,
            }
            chunks.append(piece)
    return chunks
