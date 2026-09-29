"""
safety.py - the rule-based safety layer.

DESIGN DECISION (worth stating in the report and viva):
This layer is plain Python and deliberately sits OUTSIDE LangChain and
OUTSIDE the LLM. Reasons:

1. Determinism   - the same input always gives the same decision, so it can
                   be unit-tested exhaustively (tests/test_safety.py).
2. Reliability   - a language model can be talked out of a refusal; a regex
                   cannot. Safety must not depend on model behaviour.
3. Cost/latency  - refusals never call the API at all.
4. Auditability  - every decision names the rule that fired.

The layer runs BEFORE retrieval. If it fires, the question never reaches
the LLM.

Levels (most to least severe):
    emergency - possible medical emergency or self-harm; tell the user to
                get emergency help now.
    urgent    - a number the user mentioned (BP, sugar, temperature, pulse)
                is outside a safe range; tell them to contact their doctor
                the same day.
    refer     - symptom, diagnosis or medication-change question; MediGuide
                only answers lifestyle/logistics questions, so redirect.
    ok        - safe to pass to the RAG pipeline.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from src.config import RED_FLAGS_PATH

LEVELS = ("ok", "refer", "urgent", "emergency")
_RANK = {level: i for i, level in enumerate(LEVELS)}

EMERGENCY_MSG = (
    "What you describe may need urgent medical attention. Please call your "
    "local emergency number now (for example 112 in India or 911 in the US) "
    "or go to the nearest emergency department. Do not wait for an online "
    "answer."
)
SELF_HARM_MSG = (
    "I am concerned about what you wrote. If you might act on thoughts of "
    "harming yourself, please contact your local emergency services or a "
    "crisis line in your country right away, and ask someone you trust to "
    "stay with you. Please also tell your doctor or care team how you are "
    "feeling."
)
SYMPTOM_MSG = (
    "I can only help with general recovery guidance such as diet, activity, "
    "daily routine and follow-up planning. Questions about symptoms, how you "
    "feel, or whether something is normal need your doctor or care team. "
    "Please contact them directly, and check the warning signs listed in the "
    "sidebar. If it feels severe or is getting worse quickly, seek emergency "
    "care."
)
DIAGNOSIS_MSG = (
    "I cannot diagnose conditions or explain what is causing a problem. "
    "Please speak to your doctor or care team about this."
)
MEDICATION_MSG = (
    "Decisions about starting, stopping, skipping or changing a medicine, "
    "and questions about side effects or interactions, must come from your "
    "doctor or pharmacist. Please contact them. I can only show you what "
    "your uploaded prescription lists."
)


@dataclass(frozen=True)
class SafetyResult:
    level: str
    reason: str = ""
    message: str = ""
    matched: tuple = ()

    @property
    def blocked(self) -> bool:
        return self.level != "ok"


OK = SafetyResult("ok")


# ---------------------------------------------------------------------------
# Loading and matching
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _config() -> dict:
    with open(RED_FLAGS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _compile(phrases: list[str]) -> list[tuple[str, re.Pattern]]:
    """Turn phrases into regexes. A trailing '$' means whole-word match."""
    compiled = []
    for raw in phrases:
        exact = raw.endswith("$")
        phrase = raw[:-1] if exact else raw
        body = r"\s+".join(re.escape(part) for part in phrase.split())
        pattern = rf"\b{body}\b" if exact else rf"\b{body}"
        compiled.append((phrase, re.compile(pattern, re.IGNORECASE)))
    return compiled


@lru_cache(maxsize=None)
def _patterns(key: str) -> list[tuple[str, re.Pattern]]:
    return _compile(_config().get(key, []))


def _first_hits(text: str, key: str, limit: int = 3) -> tuple:
    hits = [phrase for phrase, rx in _patterns(key) if rx.search(text)]
    return tuple(hits[:limit])


def _normalise(text: str) -> str:
    # Curly apostrophes -> straight, so "can't" matches.
    return text.replace("\u2019", "'").replace("\u2018", "'")


# ---------------------------------------------------------------------------
# Numeric checks
# ---------------------------------------------------------------------------
# Blood pressure written as "190/125" or "190 over 125".
#  - The lookbehind/lookahead stop dates ("25/12/2026") and fractions being
#    read as BP.
#  - Range validation stops "3/4 cup" (too small) and "10/12" (too small).
#  - This was a real bug in an earlier version: a plain \d+/\d+ pattern
#    treated dates and fractions as readings.
_BP_RX = re.compile(r"(?<![\d/.])(\d{2,3})\s*(?:/|over)\s*(\d{2,3})(?![\d/])", re.IGNORECASE)

# Blood sugar needs a context word or the mg/dL unit, so "sugar 2 spoons" or
# "25 g sugar" is not treated as a reading.
_SUGAR_CTX_RX = re.compile(
    r"\b(?:sugar|glucose|fbs|rbs|ppbs|fasting|random)\b"
    r"(?:\s+(?:level|levels|reading|value|is|was|of|at|now|today))*\s*[:=]?\s*"
    r"(?:is|was|at|of)?\s*(\d{2,3})(?!\d)(?!\s*(?:g\b|gm\b|gms\b|grams?\b|mg\b(?!\s*/?\s*dl)|%|ml\b|kcal|cal))",
    re.IGNORECASE,
)
_SUGAR_UNIT_RX = re.compile(r"(?<![\d.])(\d{2,3})\s*mg\s*/?\s*dl\b", re.IGNORECASE)

_TEMP_CTX_RX = re.compile(
    r"\b(?:temp(?:erature)?|fever)\b\s*(?:is|was|of|at|reading|:)?\s*(\d{2,3}(?:\.\d)?)\s*(?:°|º|degrees?|deg)?\s*([fc])?\b",
    re.IGNORECASE,
)
_TEMP_UNIT_RX = re.compile(r"(?<![\d.])(\d{2,3}(?:\.\d)?)\s*(?:°|º)\s*([fc])\b", re.IGNORECASE)

_PULSE_RX = re.compile(
    r"\b(?:pulse|heart\s*rate|hr)\b\s*(?:is|was|of|at|:)?\s*(\d{2,3})(?!\d)", re.IGNORECASE
)
_BPM_RX = re.compile(r"(?<![\d.])(\d{2,3})\s*bpm\b", re.IGNORECASE)


def _t(name: str):
    return _config()["thresholds"][name]


def _worst(results: list[SafetyResult]) -> SafetyResult:
    return max(results, key=lambda r: _RANK[r.level]) if results else OK


def check_bp(systolic: int, diastolic: int) -> SafetyResult:
    """Classify a blood pressure reading. Invalid readings return ok."""
    if not (60 <= systolic <= 300 and 30 <= diastolic <= 200 and systolic > diastolic):
        return OK
    reading = f"{systolic}/{diastolic}"
    if systolic >= _t("bp_systolic_emergency") or diastolic >= _t("bp_diastolic_emergency"):
        return SafetyResult(
            "emergency", f"bp_emergency:{reading}",
            f"A blood pressure of {reading} is in a range that can be dangerous. "
            "If you have chest pain, breathlessness, severe headache, weakness or "
            "confusion, call emergency services now. Otherwise re-check after "
            "resting for 5 minutes and, if it is still this high, seek urgent "
            "medical care.", (reading,))
    if (systolic >= _t("bp_systolic_urgent") or diastolic >= _t("bp_diastolic_urgent")
            or systolic < _t("bp_systolic_low")):
        return SafetyResult(
            "urgent", f"bp_urgent:{reading}",
            f"A blood pressure of {reading} is outside the range your care team "
            "would normally want to hear about promptly. Please contact your "
            "doctor or clinic today.", (reading,))
    return OK


def check_sugar(value: int) -> SafetyResult:
    """Classify a blood glucose reading in mg/dL."""
    if not 20 <= value <= 700:
        return OK
    if value >= _t("sugar_emergency_high") or value < _t("sugar_emergency_low"):
        return SafetyResult(
            "emergency", f"sugar_emergency:{value}",
            f"A blood sugar of {value} mg/dL is in a dangerous range. Seek "
            "emergency medical care now. If it is low and you are awake and able "
            "to swallow, follow the low-sugar plan your doctor gave you while "
            "help is on the way.", (str(value),))
    if value >= _t("sugar_urgent_high") or value < _t("sugar_urgent_low"):
        return SafetyResult(
            "urgent", f"sugar_urgent:{value}",
            f"A blood sugar of {value} mg/dL is outside the safe range. Please "
            "contact your doctor today and follow the plan they gave you for "
            "high or low readings.", (str(value),))
    return OK


def check_temperature(value: float, unit: str | None = None) -> SafetyResult:
    unit = (unit or ("c" if value < 50 else "f")).lower()
    celsius = value if unit == "c" else (value - 32) * 5 / 9
    if not 30 <= celsius <= 45:
        return OK
    shown = f"{value:g} {unit.upper()}"
    if celsius >= _t("temp_emergency_c"):
        return SafetyResult("emergency", f"temp_emergency:{shown}",
                            f"A temperature of {shown} is dangerously high. Seek emergency care now.",
                            (shown,))
    if celsius >= _t("temp_urgent_c"):
        return SafetyResult("urgent", f"temp_urgent:{shown}",
                            f"A temperature of {shown} counts as a fever after a procedure. "
                            "Please contact your doctor today.", (shown,))
    return OK


def check_pulse(value: int) -> SafetyResult:
    if not 20 <= value <= 250:
        return OK
    if value >= _t("pulse_emergency_high") or value < _t("pulse_emergency_low"):
        return SafetyResult("emergency", f"pulse_emergency:{value}",
                            f"A pulse of {value} per minute needs urgent assessment. "
                            "Seek emergency care now.", (str(value),))
    if value > _t("pulse_urgent_high") or value < _t("pulse_urgent_low"):
        return SafetyResult("urgent", f"pulse_urgent:{value}",
                            f"A pulse of {value} per minute is outside the usual range. "
                            "Please contact your doctor today.", (str(value),))
    return OK


def check_vitals(bp_systolic: int = 0, bp_diastolic: int = 0, blood_sugar: int = 0) -> list[SafetyResult]:
    """Check stored intake readings. Zero means 'not measured'. Returns only
    the readings that need attention."""
    results = []
    if bp_systolic and bp_diastolic:
        results.append(check_bp(bp_systolic, bp_diastolic))
    if blood_sugar:
        results.append(check_sugar(blood_sugar))
    return [r for r in results if r.blocked]


def _numeric_results(text: str) -> list[SafetyResult]:
    results: list[SafetyResult] = []
    for m in _BP_RX.finditer(text):
        results.append(check_bp(int(m.group(1)), int(m.group(2))))
    for rx in (_SUGAR_CTX_RX, _SUGAR_UNIT_RX):
        for m in rx.finditer(text):
            results.append(check_sugar(int(m.group(1))))
    for rx in (_TEMP_CTX_RX, _TEMP_UNIT_RX):
        for m in rx.finditer(text):
            results.append(check_temperature(float(m.group(1)), m.group(2)))
    for rx in (_PULSE_RX, _BPM_RX):
        for m in rx.finditer(text):
            results.append(check_pulse(int(m.group(1))))
    return [r for r in results if r.blocked]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def check_query(text: str) -> SafetyResult:
    """Decide whether a patient message may proceed to retrieval.

    Order matters: the most severe categories are checked first.
    """
    text = _normalise(text or "")

    hits = _first_hits(text, "self_harm")
    if hits:
        return SafetyResult("emergency", "self_harm", SELF_HARM_MSG, hits)

    hits = _first_hits(text, "medical_emergency")
    if hits:
        return SafetyResult("emergency", "medical_emergency", EMERGENCY_MSG, hits)

    numeric = _numeric_results(text)
    if numeric:
        return _worst(numeric)

    hits = _first_hits(text, "medication_change_patterns")
    if hits:
        return SafetyResult("refer", "medication_change", MEDICATION_MSG, hits)

    hits = _first_hits(text, "diagnosis_phrases")
    if hits:
        return SafetyResult("refer", "diagnosis", DIAGNOSIS_MSG, hits)

    hits = _first_hits(text, "symptom_terms")
    if hits:
        return SafetyResult("refer", "symptom", SYMPTOM_MSG, hits)

    return OK
