"""
Tests for the safety/refusal layer.
Run with: pytest tests/test_safety.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.safety import check, is_symptom_query


def test_normal_diet_question_is_answered():
    refuse, _ = check("can I eat rice after my surgery")
    assert refuse is False


def test_normal_activity_question_is_answered():
    refuse, _ = check("how much should I walk today")
    assert refuse is False


def test_fever_triggers_refusal():
    refuse, msg = check("I have a fever and the wound looks red")
    assert refuse is True
    assert msg is not None


def test_pus_triggers_refusal():
    refuse, _ = check("there is pus coming out of the incision")
    assert refuse is True


def test_sleep_question_is_answered():
    refuse, _ = check("when can I sleep on my stomach")
    assert refuse is False


def test_chest_pain_triggers_refusal():
    assert is_symptom_query("I'm having chest pain and difficulty breathing") is True


def test_emergency_keyword_triggers_refusal():
    assert is_symptom_query("is this an emergency") is True
