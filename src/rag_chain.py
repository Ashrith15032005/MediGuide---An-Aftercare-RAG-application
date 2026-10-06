"""
rag_chain.py - the RAG pipeline, written with LangChain Expression Language
(LCEL).

Request flow for one patient question:

    question
       |
       v
   [1] Safety layer (plain Python, src/safety.py)        -> refuse, or continue
       |
       v
   [2] LCEL chain
         retrieve  : vector search in ChromaDB, filtered to the patient's
                     procedure (+ topics implied by their conditions),
                     dropping chunks below a relevance threshold
         branch    : nothing relevant found -> honest "not in my guide" reply
                     (the LLM is NOT called, so it cannot invent an answer)
         generate  : prompt | Gemini | string parser
       |
       v
   Answer(text, sources)

LCEL KEY IDEAS (useful for the viva):
  * A "Runnable" is anything with .invoke(). Prompts, models, parsers,
    lambdas and whole chains are all Runnables.
  * The "|" operator pipes the output of one Runnable into the next:
        prompt | llm | parser
  * RunnablePassthrough.assign(x=...) keeps the incoming dict and adds a key.
  * RunnableBranch picks a path based on a condition, like if/else.
  * Because every step is a Runnable, the same chain can be invoked, batched
    or streamed without code changes, and tests can swap in fake components.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

from src.config import FALLBACK_MODELS, HISTORY_TURNS, LLM_MODEL, LLM_TEMPERATURE, MIN_RELEVANCE, TOP_K, require_api_key
from src.patient import PatientProfile
from src.prescriptions import medicines_context
from src.procedures import GENERAL_KEY, PROCEDURES, label_for
from src.safety import check_query

log = logging.getLogger("mediguide")

NO_CONTEXT_MSG = (
    "I could not find this in the aftercare guide for your procedure, so I do not "
    "want to guess. Please ask your doctor or care team, or try rephrasing your "
    "question about diet, activity, wound care, daily routine or follow-up."
)
LLM_ERROR_MSG = (
    "I could not reach the language model just now, so I cannot answer safely. "
    "Please try again in a moment. If your question is urgent, contact your doctor."
)

SYSTEM_PROMPT = """You are MediGuide, a post-discharge aftercare assistant. You give general \
lifestyle and logistics guidance (diet, activity, daily routine, wound care basics, follow-up \
planning) to patients recovering at home.

Rules you must always follow:
1. Answer ONLY from the numbered guide excerpts provided. If they do not contain the answer, \
say you do not have that information and suggest asking the doctor or care team. Never use \
outside medical knowledge to fill gaps.
2. Never diagnose, never interpret symptoms, never say whether something is normal or serious. \
If the patient describes a symptom, tell them to contact their doctor.
3. Never advise starting, stopping, skipping or changing a medicine or dose, and do not discuss \
side effects or interactions; refer those to the doctor or pharmacist. You may read back what the \
patient's uploaded prescription lists, and must say it was extracted automatically and may \
contain errors.
4. Use the patient background only to make general guidance fit them (for example, mention a \
listed allergy or an existing condition where the excerpts allow it). Do not infer new medical \
facts about the patient.
5. If the excerpts conflict with the patient's own discharge instructions, tell them to follow \
their doctor's instructions.
6. Cite the excerpts you used like [S1], [S2] at the end of the relevant sentence.
7. If the primary procedure is the general recovery guide, the excerpts are NOT specific to the \
patient's operation. Give only the general advice they contain, and for anything specific to the \
named procedure (lifting limits, movement restrictions, dressings, timelines) say the guide does \
not cover it and the surgeon must advise.
8. Be brief, calm and plain-spoken: short paragraphs or a short list. No emojis. Do not repeat \
the disclaimer in every answer."""

HUMAN_PROMPT = """Primary procedure or condition: {primary_label}

Patient background:
{patient_context}

Prescription information:
{prescription_context}

Guide excerpts:
{context}

Patient question: {question}"""


@dataclass
class Source:
    label: str          # "S1"
    procedure: str
    section: str
    snippet: str
    score: float


@dataclass
class Answer:
    text: str
    refused: bool = False
    safety_level: str = "ok"
    safety_reason: str = ""
    sources: list = field(default_factory=list)
    error: bool = False
    error_detail: str = ""   # real exception (type + message) when error is True


def format_docs(docs_with_scores) -> str:
    """Number the retrieved chunks so the model can cite them as [S1], [S2]."""
    blocks = []
    for i, (doc, _score) in enumerate(docs_with_scores, start=1):
        blocks.append(f"[S{i}] ({doc.metadata.get('title', '')} - {doc.metadata.get('section', '')})\n"
                      f"{doc.page_content}")
    return "\n\n".join(blocks)


def _to_messages(history) -> list:
    """history: list of (role, text) with role in {'user', 'assistant'}."""
    msgs = []
    for role, text in history[-2 * HISTORY_TURNS:]:
        msgs.append(HumanMessage(text) if role == "user" else AIMessage(text))
    return msgs


def get_llm():
    """Main model with automatic retries, plus backup models for 503 'high demand' spikes."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    require_api_key()

    def make(name: str):
        return ChatGoogleGenerativeAI(model=name, temperature=LLM_TEMPERATURE, max_retries=4)

    main = make(LLM_MODEL)
    backups = [make(m) for m in FALLBACK_MODELS if m != LLM_MODEL]
    return main.with_fallbacks(backups) if backups else main


class MediGuideAssistant:
    """Ties together safety layer, retrieval and generation.

    vectorstore and llm can be injected, which is how the tests run the whole
    pipeline with fake components and no network.
    """

    def __init__(self, vectorstore=None, llm=None, k: int = TOP_K, min_relevance: float = MIN_RELEVANCE):
        if vectorstore is None:
            from src.ingest import get_vectorstore
            vectorstore = get_vectorstore()
        self.vectorstore = vectorstore
        self.llm = llm if llm is not None else get_llm()
        self.k = k
        self.min_relevance = min_relevance
        self.chain = self._build_chain()

    @staticmethod
    def _primary_label(procedure: str, profile: PatientProfile) -> str:
        """For the general fallback guide, tell the model what the patient actually had."""
        if procedure == GENERAL_KEY and profile.procedure_description.strip():
            return (f"{label_for(procedure)}. The patient describes it as: "
                    f"\"{profile.procedure_description.strip()[:200]}\"")
        return label_for(procedure)

    # -- retrieval ----------------------------------------------------------
    @staticmethod
    def _search_filter(procedures: list[str]) -> dict:
        # Only guidance chunks (never warning-sign chunks) from the patient's topics.
        return {"$and": [{"kind": "guidance"}, {"procedure": {"$in": procedures}}]}

    def _retrieve(self, inputs: dict) -> list:
        """Return [(Document, score)] above the relevance threshold."""
        query = f"{label_for(inputs['primary'])}: {inputs['question']}"
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=self.k, filter=self._search_filter(inputs["procedures"]))
        return [(d, s) for d, s in results if s >= self.min_relevance]

    # -- LCEL chain ---------------------------------------------------------
    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", HUMAN_PROMPT),
        ])
        generate = prompt | self.llm | StrOutputParser()

        with_docs = RunnablePassthrough.assign(docs=RunnableLambda(self._retrieve))
        with_context = RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: format_docs(x["docs"]) or "(no relevant guide excerpts found)"))

        answer_step = RunnableBranch(
            # Condition 1: nothing relevant retrieved -> refuse to guess.
            # (If the patient uploaded a prescription, still call the model so it can read
            # back what the prescription lists; the prompt forbids inventing anything else.)
            (lambda x: not x["docs"] and not x["has_medicines"], RunnableLambda(lambda x: NO_CONTEXT_MSG)),
            # Default: generate a grounded answer.
            generate,
        )
        return with_docs | with_context | RunnablePassthrough.assign(answer=answer_step)

    # -- public API ---------------------------------------------------------
    def answer(self, question: str, procedure: str, profile: PatientProfile | None = None,
               medicines: list | None = None, history: list | None = None) -> Answer:
        if procedure not in PROCEDURES:
            raise ValueError(f"Unknown procedure '{procedure}'")

        # Step 1: safety layer. Blocked questions never reach retrieval or the LLM.
        safety = check_query(question)
        if safety.blocked:
            return Answer(text=safety.message, refused=True,
                          safety_level=safety.level, safety_reason=safety.reason)

        profile = profile or PatientProfile(procedure_key=procedure)
        # Personalised retrieval: add topics implied by the patient's conditions.
        procedures = [procedure] + [p for p in profile.secondary_procedures() if p != procedure]

        payload = {
            "question": question,
            "primary": procedure,
            "primary_label": self._primary_label(procedure, profile),
            "procedures": procedures,
            "patient_context": profile.to_context(),
            "prescription_context": medicines_context(medicines or []),
            "has_medicines": bool(medicines),
            "history": _to_messages(history or []),
        }
        # Step 2: retrieval + generation.
        try:
            out = self.chain.invoke(payload)
        except Exception as exc:  # network, quota, bad key, model errors
            # Keep the real cause: the patient sees a calm message, but the app and
            # the terminal show what actually failed (bad key, quota, model name...).
            log.exception("MediGuide pipeline failed")
            return Answer(text=LLM_ERROR_MSG, error=True,
                          error_detail=f"{type(exc).__name__}: {str(exc)[:400]}")

        sources = [
            Source(label=f"S{i}", procedure=d.metadata.get("procedure", ""),
                   section=d.metadata.get("section", ""),
                   snippet=d.page_content.split("\n", 1)[-1][:280], score=round(float(s), 3))
            for i, (d, s) in enumerate(out["docs"], start=1)
        ]
        return Answer(text=out["answer"].strip(), sources=sources)


def list_procedures() -> list[str]:
    return list(PROCEDURES)