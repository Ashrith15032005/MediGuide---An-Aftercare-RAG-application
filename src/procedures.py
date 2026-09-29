"""
procedures.py - the registry of supported procedures/conditions, plus a
matcher that turns a patient's own words into one of them.

Why this exists: patients should be able to describe their procedure freely
("I had my appendix taken out last week") instead of choosing from a
dropdown. The matcher is deliberately simple and deterministic (phrase
matching plus a light typo tolerance) so it is testable and needs no API
call. Anything it cannot match is reported honestly as "not supported" rather
than guessed.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Procedure:
    key: str          # used as metadata in the vector store
    label: str        # shown to the patient
    kb_file: str      # file inside data/knowledge_base/
    aliases: tuple    # phrases patients might use


PROCEDURES: dict[str, Procedure] = {
    p.key: p
    for p in (
        Procedure(
            "appendectomy", "Appendectomy (appendix removal)", "appendectomy.md",
            ("appendectomy", "appendicectomy", "appendix", "appendix removal",
             "appendix removed", "appendix surgery", "appendicitis",
             "laparoscopic appendectomy"),
        ),
        Procedure(
            "c_section", "C-section (caesarean delivery)", "c_section.md",
            ("c section", "csection", "caesarean", "cesarean", "caesarean section",
             "cesarean section", "caesarean delivery", "cesarean delivery", "lscs",
             "lower segment caesarean", "surgical delivery", "baby by operation"),
        ),
        Procedure(
            "diabetes_hypertension", "Diabetes and hypertension management",
            "diabetes_hypertension.md",
            ("diabetes", "diabetic", "type 2 diabetes", "type 1 diabetes", "sugar",
             "blood sugar", "high sugar", "hypertension", "high blood pressure",
             "high bp", "blood pressure", "bp", "hba1c", "insulin"),
        ),
        Procedure(
            "cardiac_recovery", "Cardiac recovery (after a heart procedure or event)",
            "cardiac_recovery.md",
            ("cardiac", "heart attack", "myocardial infarction", "angioplasty", "stent",
             "stenting", "bypass", "cabg", "heart surgery", "open heart",
             "heart bypass", "coronary", "cardiac rehab", "heart operation"),
        ),
    )
}


def list_procedures() -> list[str]:
    return list(PROCEDURES)


def label_for(key: str) -> str:
    return PROCEDURES[key].label if key in PROCEDURES else key


@dataclass(frozen=True)
class MatchResult:
    key: str
    score: int
    matched: tuple


_WORD_RX = re.compile(r"[a-z0-9]+")


def _normalise(text: str) -> str:
    # "c-section" and "C section" and "csection" should all look alike.
    text = text.lower().replace("-", " ")
    return " ".join(_WORD_RX.findall(text))


def _score(text: str, tokens: list[str], proc: Procedure) -> MatchResult:
    score, matched = 0, []
    for alias in proc.aliases:
        # Exact whole-phrase hit: longer phrases are stronger evidence.
        if re.search(rf"\b{re.escape(alias)}\b", text):
            score += 3 * len(alias.split())
            matched.append(alias)
            continue
        # Typo tolerance for long single words only ("apendectomy").
        if " " not in alias and len(alias) >= 7:
            if difflib.get_close_matches(alias, tokens, n=1, cutoff=0.84):
                score += 2
                matched.append(alias + "~")
    return MatchResult(proc.key, score, tuple(matched))


def match_all(text: str, min_score: int = 3) -> list[MatchResult]:
    """Every procedure that matches the text, best first."""
    norm = _normalise(text)
    tokens = norm.split()
    results = [_score(norm, tokens, p) for p in PROCEDURES.values()]
    return sorted((r for r in results if r.score >= min_score),
                  key=lambda r: r.score, reverse=True)


def match_procedure(text: str) -> MatchResult | None:
    """Best single match, or None if nothing in the registry fits."""
    results = match_all(text)
    if not results:
        return None
    # A tie with no clear winner is ambiguous; let the UI ask the patient.
    if len(results) > 1 and results[0].score == results[1].score:
        return None
    return results[0]
