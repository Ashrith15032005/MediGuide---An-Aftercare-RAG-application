"""Ingestion tests - no API key needed because only splitting is exercised."""
from src.knowledge_base import load_chunks, warning_signs
from src.procedures import PROCEDURES


def test_every_procedure_has_chunks():
    chunks = load_chunks()
    assert {c.metadata["procedure"] for c in chunks} == set(PROCEDURES)


def test_chunk_metadata_and_prefix():
    for c in load_chunks():
        assert set(c.metadata) == {"procedure", "title", "section", "kind", "source"}
        assert c.page_content.startswith(f"{c.metadata['title']} | {c.metadata['section']}")


def test_chunks_respect_size_limit():
    # size limit + prefix allowance
    assert all(len(c.page_content) <= 700 + 120 for c in load_chunks())


def test_warning_sections_are_excluded_from_guidance():
    chunks = load_chunks()
    warn = [c for c in chunks if c.metadata["kind"] == "warning"]
    assert len(warn) == len(PROCEDURES)
    assert all(c.metadata["section"].lower().startswith("warning") for c in warn)
    assert not any("warning" in c.metadata["section"].lower()
                   for c in chunks if c.metadata["kind"] == "guidance")


def test_warning_signs_parsed_for_display():
    for key in PROCEDURES:
        signs = warning_signs(key)
        assert len(signs) >= 5
        assert all(not s.startswith("-") for s in signs)


def test_no_section_is_split_across_topics():
    # Stage 1 guarantee: each chunk belongs to exactly one section heading.
    for c in load_chunks():
        assert "\n## " not in c.page_content
