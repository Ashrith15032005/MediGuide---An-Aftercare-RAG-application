from src.prescriptions import Medicine, medicines_context, parse_medicines

SAMPLE = """City Hospital
Dr. Rao MBBS  Reg No 12345
Rx
1. Tab. Paracetamol 500mg TDS x 5 days after food
2) Cap Pantoprazole 40 mg OD before food for 14 days
Metformin 500mg 1-0-1 for 30 days
Inj Enoxaparin 40 mg OD
Date: 12/05/2026
Review after 1 week
"""


def test_parses_prefixed_and_unprefixed_lines():
    meds = parse_medicines(SAMPLE, source="rx1.png")
    names = [m.name for m in meds]
    assert names == ["Tab Paracetamol", "Cap Pantoprazole", "Metformin", "Inj Enoxaparin"]


def test_fields_extracted():
    meds = {m.name: m for m in parse_medicines(SAMPLE)}
    p = meds["Tab Paracetamol"]
    assert (p.dose, p.frequency, p.duration, p.instructions) == ("500mg", "TDS", "5 days", "after food")
    assert meds["Metformin"].frequency == "1-0-1"
    assert meds["Cap Pantoprazole"].duration == "14 days"


def test_header_and_date_lines_ignored():
    names = [m.name for m in parse_medicines(SAMPLE)]
    assert not any("Hospital" in n or "Review" in n or "Date" in n for n in names)


def test_empty_text():
    assert parse_medicines("") == []
    assert medicines_context([]) == "No prescription uploaded."


def test_context_lists_medicines_and_warns_about_ocr():
    ctx = medicines_context([Medicine("Metformin", "500mg", "BD").to_dict()])
    assert "Metformin" in ctx and "may contain errors" in ctx
