"""Checks the REAL Chroma filter syntax and the score API used by the
assistant, with offline fake embeddings (no API key). Retrieval QUALITY needs
real embeddings and is measured by eval/evaluate.py instead."""
import warnings

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.knowledge_base import load_chunks
from src.patient import PatientProfile
from src.rag_chain import MediGuideAssistant


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    chunks = load_chunks()
    vs = Chroma(collection_name="mediguide_test", embedding_function=DeterministicFakeEmbedding(size=64),
                persist_directory=str(tmp_path_factory.mktemp("chroma")),
                collection_metadata={"hnsw:space": "cosine"})
    vs.add_documents(chunks, ids=[f"{c.metadata['procedure']}-{i}" for i, c in enumerate(chunks)])
    return vs


def _assistant(store):
    # min_relevance very low so the fake embeddings never trigger the fallback
    return MediGuideAssistant(vectorstore=store, llm=FakeListChatModel(responses=["ok"] * 10),
                              min_relevance=-10.0, k=6)


def test_filter_limits_results_to_one_procedure_and_guidance(store):
    bot = _assistant(store)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        docs = bot._retrieve({"question": "diet", "primary": "c_section",
                              "procedures": ["c_section"]})
    assert docs
    assert {d.metadata["procedure"] for d, _ in docs} == {"c_section"}
    assert all(d.metadata["kind"] == "guidance" for d, _ in docs)


def test_secondary_topics_are_included(store):
    bot = _assistant(store)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        docs = bot._retrieve({"question": "what should I eat", "primary": "c_section",
                              "procedures": ["c_section", "diabetes_hypertension"]})
    assert {d.metadata["procedure"] for d, _ in docs} <= {"c_section", "diabetes_hypertension"}


def test_full_answer_path_against_real_chroma(store):
    bot = _assistant(store)
    profile = PatientProfile.from_lists("diabetes", procedure_key="appendectomy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ans = bot.answer("what can I eat?", "appendectomy", profile=profile)
    assert ans.text == "ok" and ans.sources
    assert {s.procedure for s in ans.sources} <= {"appendectomy", "diabetes_hypertension"}
