import pytest

from src.procedures import PROCEDURES, match_all, match_procedure


@pytest.mark.parametrize("text,key", [
    ("I had my appendix taken out last week", "appendectomy"),
    ("laparoscopic appendicectomy", "appendectomy"),
    ("apendectomy", "appendectomy"),                       # typo tolerance
    ("C-section delivery", "c_section"),
    ("I had a caesarean", "c_section"),
    ("cesarean section two weeks ago", "c_section"),
    ("I have diabetes and high BP", "diabetes_hypertension"),
    ("type 2 diabetes", "diabetes_hypertension"),
    ("angioplasty with a stent", "cardiac_recovery"),
    ("CABG bypass surgery", "cardiac_recovery"),
    ("recovering from a heart attack", "cardiac_recovery"),
])
def test_free_text_matching(text, key):
    result = match_procedure(text)
    assert result is not None and result.key == key


@pytest.mark.parametrize("text", ["hello there", "I broke my leg", "knee replacement", ""])
def test_unsupported_returns_none(text):
    assert match_procedure(text) is None


def test_all_registry_files_exist():
    from src.knowledge_base import kb_path
    for key in PROCEDURES:
        assert kb_path(key).exists()


def test_match_all_finds_multiple_topics():
    keys = [m.key for m in match_all("heart bypass surgery and diabetes")]
    assert "cardiac_recovery" in keys and "diabetes_hypertension" in keys
