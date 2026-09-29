"""
patient.py - patient health background and local storage.

The intake form fills a PatientProfile. It is used in two ways:
  1. As text context in the prompt (age, allergies, conditions ...).
  2. To widen retrieval: a C-section patient who also has diabetes gets
     diabetes guidance retrieved alongside C-section guidance
     (see PatientProfile.secondary_procedures).

Storage is one JSON file per patient under data/patients/ (git-ignored).
This is fine for a single-machine demo. A real deployment would need an
encrypted database, authentication and consent handling - listed as future
work in the README.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from src.config import PATIENT_DIR
from src.procedures import match_all


def _split_list(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,\n;]", text or "") if p.strip()]


@dataclass
class PatientProfile:
    patient_id: str = "demo_patient"
    name: str = ""
    age: int = 0
    sex: str = ""
    conditions: list = field(default_factory=list)   # past/current conditions
    allergies: list = field(default_factory=list)
    other_medications: str = ""
    bp_systolic: int = 0        # 0 means "not measured"
    bp_diastolic: int = 0
    blood_sugar: int = 0        # mg/dL, 0 means "not measured"
    discharge_date: str = ""    # ISO date, optional
    procedure_key: str = ""
    procedure_description: str = ""

    # -- helpers -----------------------------------------------------------
    @classmethod
    def from_lists(cls, conditions_text: str = "", allergies_text: str = "", **kw):
        return cls(conditions=_split_list(conditions_text),
                   allergies=_split_list(allergies_text), **kw)

    def days_since_discharge(self, today: date | None = None) -> int | None:
        if not self.discharge_date:
            return None
        try:
            d = date.fromisoformat(self.discharge_date)
        except ValueError:
            return None
        return max(0, ((today or date.today()) - d).days)

    def secondary_procedures(self) -> list[str]:
        """Other supported topics implied by the patient's conditions."""
        keys = []
        for cond in self.conditions:
            for m in match_all(cond):
                if m.key != self.procedure_key and m.key not in keys:
                    keys.append(m.key)
        return keys

    def to_context(self) -> str:
        """Plain-text summary injected into the prompt."""
        lines = []
        if self.age:
            lines.append(f"Age: {self.age}")
        if self.sex:
            lines.append(f"Sex: {self.sex}")
        days = self.days_since_discharge()
        if days is not None:
            lines.append(f"Days since discharge: {days}")
        if self.conditions:
            lines.append("Existing conditions: " + ", ".join(self.conditions))
        if self.allergies:
            lines.append("Allergies: " + ", ".join(self.allergies))
        if self.other_medications:
            lines.append("Other medicines reported: " + self.other_medications)
        if self.bp_systolic and self.bp_diastolic:
            lines.append(f"Last blood pressure reading: {self.bp_systolic}/{self.bp_diastolic}")
        if self.blood_sugar:
            lines.append(f"Last blood sugar reading: {self.blood_sugar} mg/dL")
        return "\n".join(lines) if lines else "No patient background provided."


def _safe_id(patient_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", patient_id) or "demo_patient"


class PatientStore:
    """Tiny JSON-file store: one file per patient."""

    def __init__(self, directory: Path = PATIENT_DIR):
        self.directory = Path(directory)

    def _path(self, patient_id: str) -> Path:
        return self.directory / f"{_safe_id(patient_id)}.json"

    def save(self, profile: PatientProfile, medicines: list[dict] | None = None) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"profile": asdict(profile), "medicines": medicines or []}
        self._path(profile.patient_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, patient_id: str) -> tuple[PatientProfile, list[dict]] | None:
        path = self._path(patient_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        known = PatientProfile.__dataclass_fields__
        profile = PatientProfile(**{k: v for k, v in data["profile"].items() if k in known})
        return profile, data.get("medicines", [])

    def delete(self, patient_id: str) -> None:
        self._path(patient_id).unlink(missing_ok=True)
