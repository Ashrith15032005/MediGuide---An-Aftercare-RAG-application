"""End-to-end pipeline tests using fake components (no network, no API key)."""
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.patient import PatientProfile
from src.rag_chain import NO_CONTEXT_MSG, MediGuideAssistant


class FakeStore:
    """Mimics the Chroma method used by the assistant and records the call."""

    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def similarity_search_with_relevance_scores(self, query, k=4, filter=None):
        self.calls.append({"query": query, "k": k, "filter": filter})
        return self.hits


class CountingLLM(FakeListChatModel):
    calls: int = 0

    def _call(self, *args, **kwargs):
        self.calls += 1
        return super()._call(*args, **kwargs)


def _doc(section="Diet", proc="appendectomy"):
    return Document(page_content=f"Appendectomy Recovery | {section}\nEat light foods such as rice and dal.",
                    metadata={"procedure": proc, "title": "Appendectomy Recovery", "section": section,
                              "kind": "guidance", "source": "x.md"})


def make(hits, reply="Rice is fine. [S1]"):
    store, llm = FakeStore(hits), CountingLLM(responses=[reply])
    return MediGuideAssistant(vectorstore=store, llm=llm, min_relevance=0.35), store, llm


def test_normal_question_is_answered_with_sources():
    bot, store, llm = make([(_doc(), 0.8)])
    ans = bot.answer("can I eat rice?", "appendectomy")
    assert not ans.refused and ans.text == "Rice is fine. [S1]"
    assert ans.sources[0].label == "S1" and ans.sources[0].section == "Diet"
    assert llm.calls == 1


def test_symptom_question_never_reaches_retrieval_or_llm():
    bot, store, llm = make([(_doc(), 0.9)])
    ans = bot.answer("my wound is red and swollen", "appendectomy")
    assert ans.refused and ans.safety_level == "refer"
    assert store.calls == [] and llm.calls == 0


def test_emergency_is_flagged():
    bot, store, llm = make([(_doc(), 0.9)])
    ans = bot.answer("I have chest pain", "cardiac_recovery")
    assert ans.refused and ans.safety_level == "emergency" and llm.calls == 0


def test_low_relevance_falls_back_without_calling_llm():
    bot, store, llm = make([(_doc(), 0.10)])
    ans = bot.answer("can I bring a pet parrot?", "appendectomy")
    assert ans.text == NO_CONTEXT_MSG and ans.sources == [] and llm.calls == 0


def test_retrieval_is_scoped_to_procedure_and_excludes_warning_chunks():
    bot, store, _ = make([(_doc(), 0.8)])
    bot.answer("can I eat rice?", "appendectomy")
    flt = store.calls[0]["filter"]["$and"]
    assert {"kind": "guidance"} in flt
    assert {"procedure": {"$in": ["appendectomy"]}} in flt


def test_patient_conditions_widen_retrieval():
    bot, store, _ = make([(_doc(), 0.8)])
    profile = PatientProfile.from_lists("type 2 diabetes", procedure_key="c_section")
    bot.answer("what should I eat?", "c_section", profile=profile)
    flt = store.calls[0]["filter"]["$and"]
    assert {"procedure": {"$in": ["c_section", "diabetes_hypertension"]}} in flt


def test_llm_failure_returns_safe_message():
    class Boom(FakeListChatModel):
        def _call(self, *a, **k):
            raise RuntimeError("quota exceeded")
    bot = MediGuideAssistant(vectorstore=FakeStore([(_doc(), 0.9)]), llm=Boom(responses=["x"]))
    ans = bot.answer("can I eat rice?", "appendectomy")
    assert ans.error and "could not reach" in ans.text


def test_unknown_procedure_rejected():
    bot, *_ = make([])
    try:
        bot.answer("hello", "knee_replacement")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_history_is_accepted():
    bot, *_ = make([(_doc(), 0.8)])
    ans = bot.answer("and dal?", "appendectomy",
                     history=[("user", "can I eat rice?"), ("assistant", "Yes. [S1]")])
    assert not ans.refused
