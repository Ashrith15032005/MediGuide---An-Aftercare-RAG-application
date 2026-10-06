"""PDF tables come out of the text layer one cell per line; the parser must handle that."""
from src.prescriptions import parse_medicines

TABLE_TEXT = """SAMPLE TEST PRESCRIPTION
Patient: Rahul Sharma    Age: 45
Medicine
Dose
Frequency
Duration
Instructions
Paracetamol
500 mg
Every 6–8 hours as needed
5 days
Take after food
Pantoprazole
40 mg
Once daily
7 days
30 min before breakfast
Prescriber: Dr. A. Mehta
This document is fictional.
"""


def test_cell_per_line_table_is_parsed():
    meds = parse_medicines(TABLE_TEXT)
    assert [m.name for m in meds] == ["Paracetamol", "Pantoprazole"]
    assert meds[0].dose == "500mg" and meds[0].frequency.startswith("Every 6")
    assert meds[1].duration == "7 days" and meds[1].instructions == "30 min before breakfast"


def test_footer_is_not_read_as_a_medicine():
    assert all("Prescriber" not in m.name for m in parse_medicines(TABLE_TEXT))