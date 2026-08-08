"""
Patient History Module
-----------------------
Keeps a lightweight per-patient record for the current session: selected
procedure, health background, past Q&A turns, and extracted prescription
data. This is what makes answers feel personalized rather than starting
fresh every time.

Beta scope: file-based (JSON), single session per patient_id. Swapping
this for a real database with patient login is future work — the
interface (get_context_summary / add_turn) will not need to change.
"""

import os
import json
from datetime import datetime, timezone

from src import config


class PatientHistory:
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.path = os.path.join(config.SESSIONS_DIR, f"{patient_id}.json")
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {
                "patient_id": self.patient_id,
                "procedure": None,
                "health_background": None,
                "prescription": None,
                "prescriptions": [],
                "turns": [],
            }

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def set_procedure(self, procedure: str):
        self.data["procedure"] = procedure
        self._save()

    def set_health_background(self, background: dict):
        self.data["health_background"] = background
        self._save()

    def add_prescription(self, prescription: dict):
        """Append a prescription (patient may upload more than one)."""
        if "prescriptions" not in self.data:
            self.data["prescriptions"] = []
        self.data["prescriptions"].append(prescription)
        self._save()

    def set_prescription(self, prescription: dict):
        # Kept for backward compatibility; prefer add_prescription.
        self.data["prescription"] = prescription
        self._save()

    def add_turn(self, query: str, answer: str, refused: bool = False):
        self.data["turns"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "answer": answer,
            "refused": refused,
        })
        self._save()

    def get_context_summary(self, max_turns: int = 3) -> str:
        parts = []
        if self.data.get("procedure"):
            parts.append(f"Patient's procedure: {self.data['procedure']}")

        bg = self.data.get("health_background")
        if bg:
            bg_bits = []
            if bg.get("age"):
                bg_bits.append(f"age {bg['age']}")
            if bg.get("blood_pressure"):
                bg_bits.append(f"blood pressure {bg['blood_pressure']}")
            if bg.get("existing_conditions"):
                bg_bits.append(f"existing conditions: {bg['existing_conditions']}")
            if bg.get("allergies"):
                bg_bits.append(f"allergies: {bg['allergies']}")
            if bg.get("past_surgeries"):
                bg_bits.append(f"past surgeries: {bg['past_surgeries']}")
            if bg_bits:
                parts.append("Patient health background: " + "; ".join(bg_bits))

        all_meds = []
        for rx in self.data.get("prescriptions", []):
            all_meds.extend(rx.get("medicines", []))
        if self.data.get("prescription"):
            all_meds.extend(self.data["prescription"].get("medicines", []))
        if all_meds:
            parts.append("Patient's prescribed medicines: " + ", ".join(all_meds))

        for t in self.data["turns"][-max_turns:]:
            if not t["refused"]:
                parts.append(f'Previously asked: "{t["query"]}"')
        return "\n".join(parts) if parts else "No prior context."