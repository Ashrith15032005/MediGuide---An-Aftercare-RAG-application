from datetime import date

from src.patient import PatientProfile, PatientStore


def test_context_contains_key_facts():
    p = PatientProfile.from_lists("diabetes, hypertension", "penicillin", age=54, bp_systolic=132,
                                  bp_diastolic=84, procedure_key="cardiac_recovery",
                                  discharge_date="2026-09-20")
    ctx = p.to_context()
    assert "Age: 54" in ctx and "penicillin" in ctx and "132/84" in ctx
    assert p.days_since_discharge(date(2026, 9, 29)) == 9
    assert "Days since discharge: " in ctx


def test_empty_profile():
    assert PatientProfile().to_context() == "No patient background provided."


def test_secondary_procedures_from_conditions():
    p = PatientProfile.from_lists("Type 2 diabetes, high blood pressure", procedure_key="c_section")
    assert p.secondary_procedures() == ["diabetes_hypertension"]
    p2 = PatientProfile.from_lists("diabetes", procedure_key="diabetes_hypertension")
    assert p2.secondary_procedures() == []


def test_store_roundtrip(tmp_path):
    store = PatientStore(tmp_path)
    p = PatientProfile.from_lists("asthma", "", patient_id="a/b c", name="Test", age=30)
    store.save(p, [{"name": "Metformin"}])
    loaded, meds = store.load("a/b c")
    assert loaded.name == "Test" and loaded.conditions == ["asthma"] and meds == [{"name": "Metformin"}]
    store.delete("a/b c")
    assert store.load("a/b c") is None


def test_bad_discharge_date_is_ignored():
    assert PatientProfile(discharge_date="not-a-date").days_since_discharge() is None
