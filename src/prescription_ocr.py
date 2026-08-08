"""
Prescription Scan Module
--------------------------
Patient uploads a photo/scan of their prescription. Runs OCR (Tesseract)
to extract raw text, then applies a light heuristic parser to pull out
medicine names/dosages. Result is stored in patient history and injected
into future RAG prompts for personalization.

Beta scope: heuristic line-based extraction, works best on clearly
printed prescriptions. Handwritten ones are noisy — a vision-LLM or
trained NER model is listed as future work.
"""

import re
from PIL import Image
import pytesseract

COMMON_DRUG_HINTS = [
    "mg", "tablet", "tab", "cap", "capsule", "syrup", "mcg", "ml",
    "amoxicillin", "paracetamol", "metformin", "ibuprofen", "azithromycin",
    "pantoprazole", "cetirizine", "amlodipine", "atorvastatin", "insulin",
]


def extract_text(image_path: str) -> str:
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)


def parse_medicines(raw_text: str):
    medicines = []
    dosage_pattern = re.compile(r"\b\d+\s?(mg|ml|mcg)\b", re.IGNORECASE)
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if dosage_pattern.search(line) or any(hint in lower for hint in COMMON_DRUG_HINTS):
            medicines.append(line)
    return medicines


def process_prescription(image_path: str) -> dict:
    raw_text = extract_text(image_path)
    return {
        "raw_text": raw_text.strip(),
        "medicines": parse_medicines(raw_text),
    }
