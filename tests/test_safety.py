"""Safety layer tests. These run offline and are the most important tests in
the project: the layer must refuse what it should AND must not refuse
harmless lifestyle questions."""
import pytest

from src.safety import check_bp, check_query, check_sugar, check_temperature, check_vitals


@pytest.mark.parametrize("text", [
    "I have chest pain", "I can't breathe properly", "I cannot breathe", "I fainted this morning",
    "there is heavy bleeding", "my face is drooping", "I think I am having a heart attack",
    "I am vomiting blood", "sudden weakness on one side",
])
def test_medical_emergencies(text):
    r = check_query(text)
    assert r.level == "emergency" and r.reason == "medical_emergency"


@pytest.mark.parametrize("text", ["I want to die", "thinking about suicide", "I might hurt myself"])
def test_self_harm_is_emergency(text):
    r = check_query(text)
    assert r.level == "emergency" and r.reason == "self_harm"


@pytest.mark.parametrize("text", [
    "my wound is red and swollen", "I have a fever", "I feel dizzy", "there is pus at the stitches",
    "I am vomiting", "is this normal?", "I have a cough", "my leg is swollen", "it hurts when I walk",
])
def test_symptoms_are_referred(text):
    r = check_query(text)
    assert r.level == "refer" and r.reason == "symptom"


@pytest.mark.parametrize("text", [
    "can I stop taking my tablets", "what are the side effects of metformin",
    "should I skip a dose", "can I double the dose", "is it safe to take with alcohol? interact",
    "how many units of insulin should I take",
])
def test_medication_changes_are_referred(text):
    r = check_query(text)
    assert r.level == "refer" and r.reason == "medication_change"


def test_diagnosis_is_referred():
    r = check_query("what is wrong with me?")
    assert r.level == "refer" and r.reason == "diagnosis"


@pytest.mark.parametrize("text", [
    "can I eat rice?", "when can I take a bath", "what is a low sodium diet",
    "how far should I walk each day", "can I climb stairs", "when can I drive",
    "what foods help with constipation", "how much water should I drink",
    "when is my follow-up visit", "can I lift my baby", "how do I sleep comfortably",
])
def test_lifestyle_questions_pass(text):
    assert check_query(text).level == "ok", text


# ---- numeric thresholds (regression tests for the BP regex bug) ----------
@pytest.mark.parametrize("text,level", [
    ("my BP is 190/125", "emergency"),
    ("BP 190 over 120 today", "emergency"),
    ("bp reading 165/105 this morning", "urgent"),
    ("my pressure was 85/55", "urgent"),
    ("120/80 is my usual reading", "ok"),
    ("140/90 today", "ok"),
])
def test_blood_pressure_numbers(text, level):
    assert check_query(text).level == level


@pytest.mark.parametrize("text", [
    "my review is on 25/12/2026", "I need 3/4 cup of rice", "add 1/2 teaspoon salt",
    "appointment 10/12", "eat 200/300 grams",
])
def test_dates_and_fractions_are_not_blood_pressure(text):
    assert check_query(text).level == "ok", text


@pytest.mark.parametrize("text,level", [
    ("blood sugar 320", "urgent"), ("my sugar is 55 mg/dl", "urgent"),
    ("glucose reading 45", "emergency"), ("sugar level 110", "ok"), ("fasting sugar 95", "ok"),
])
def test_blood_sugar_numbers(text, level):
    assert check_query(text).level == level


@pytest.mark.parametrize("text", ["can I have sugar 2 spoons in tea", "I take 25 g sugar daily", "sugar 30 grams is too much?"])
def test_sugar_quantities_are_not_readings(text):
    assert check_query(text).level == "ok", text


def test_temperature_and_pulse():
    assert check_query("temperature 101 F").level == "urgent"
    assert check_query("temperature 38.5").level == "urgent"
    assert check_query("temperature 98.6 F").level == "ok"
    assert check_query("pulse 130").level == "urgent"
    assert check_query("my pulse is 72").level == "ok"


def test_most_severe_numeric_wins():
    assert check_query("BP 165/105 and sugar 320 and BP 190/125").level == "emergency"


def test_curly_apostrophe():
    assert check_query("I can\u2019t breathe").level == "emergency"


def test_check_functions_directly():
    assert check_bp(190, 125).level == "emergency"
    assert check_bp(125, 80).level == "ok"
    assert check_bp(30, 20).level == "ok"           # implausible reading ignored
    assert check_sugar(320).level == "urgent"
    assert check_temperature(102, "f").level == "urgent"


def test_check_vitals_reports_only_problems():
    assert check_vitals(120, 80, 100) == []
    out = check_vitals(185, 125, 320)
    assert {r.level for r in out} == {"emergency", "urgent"}
    assert check_vitals(0, 0, 0) == []


# Regressions found by eval/evaluate.py: harmless questions that contain a
# symptom-like WORD ("blood pressure", "coughing") must not be refused.
@pytest.mark.parametrize("text", [
    "How should I take my blood pressure at home?",
    "What foods help lower blood pressure?",
    "How do I protect my chest after surgery when coughing?",
    "How do I check my blood sugar level at home?",
])
def test_symptom_words_in_harmless_context_pass(text):
    assert check_query(text).level == "ok", text


@pytest.mark.parametrize("text", ["there is blood on my dressing", "I have a cough", "my cough is getting worse"])
def test_real_symptoms_still_refused(text):
    assert check_query(text).blocked, text
