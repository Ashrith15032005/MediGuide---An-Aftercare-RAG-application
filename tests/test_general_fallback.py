"""Surgeries without a specific guide fall back to the general recovery guide."""
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.knowledge_base import warning_signs
from src.patient import PatientProfile
from src.procedures import GENERAL_KEY, SPECIFIC_KEYS, match_all
from src.rag_chain import MediGuideAssistant


class FakeStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_relevance_scores(self, query, k=4, filter=None):
        self.calls.append(filter)
        d = Document(page_content="General Recovery After Surgery | Diet\nEat light foods.",
                     metadata={"procedure": GENERAL_KEY, "title": "General", "section": "Diet",
                               "kind": "guidance", "source": "g.md"})
        return [(d, 0.9)]


class BrokenLLM(FakeListChatModel):
    def _call(self, *a, **k):
        raise RuntimeError("429 quota exceeded")


def test_general_is_never_auto_matched():
    for text in ("knee replacement", "gallbladder removal", "hernia repair"):
        assert all(m.key != GENERAL_KEY for m in match_all(text))
    assert GENERAL_KEY not in SPECIFIC_KEYS


def test_general_guide_has_warning_signs():
    assert len(warning_signs(GENERAL_KEY)) >= 5


def test_general_procedure_is_answerable_and_scoped():
    store = FakeStore()
    bot = MediGuideAssistant(vectorstore=store, llm=FakeListChatModel(responses=["Eat light food. [S1]"]))
    profile = PatientProfile(procedure_key=GENERAL_KEY, procedure_description="knee replacement")
    ans = bot.answer("diet", GENERAL_KEY, profile=profile)
    assert not ans.error and ans.text.startswith("Eat light")
    assert {"procedure": {"$in": [GENERAL_KEY]}} in store.calls[0]["$and"]


def test_llm_failure_exposes_real_reason():
    bot = MediGuideAssistant(vectorstore=FakeStore(), llm=BrokenLLM(responses=["x"]))
    ans = bot.answer("diet", GENERAL_KEY)
    assert ans.error and "429 quota exceeded" in ans.error_detail