"""
prescriptions.py - read prescriptions from images and PDFs.

Pipeline:  file bytes -> text (OCR or PDF text layer) -> medicine list.

* Images (png/jpg)  : Pillow preprocessing + Tesseract OCR (pytesseract).
* PDFs              : text layer via pypdf; if the PDF is a scan (almost no
                      text), pages are rendered with pypdfium2 and OCR'd.
* Parsing           : heuristic, line based. Works best on printed
                      prescriptions; handwriting is noisy.

Because OCR makes mistakes, the UI shows the parsed table to the patient for
correction. Medicines are used ONLY as reference context: MediGuide never
advises changing them (that is enforced by the safety layer and the prompt).
"""
from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass, field


class PrescriptionError(RuntimeError):
    """Raised when a file cannot be read (missing OCR engine, bad file ...)."""


@dataclass
class Medicine:
    name: str
    dose: str = ""
    frequency: str = ""
    duration: str = ""
    instructions: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        parts = [self.name]
        for label, value in (("dose", self.dose), ("frequency", self.frequency),
                             ("duration", self.duration), ("notes", self.instructions)):
            if value:
                parts.append(f"{label}: {value}")
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def _ocr_image(image) -> str:
    try:
        import pytesseract
        from PIL import ImageOps
    except ImportError as exc:  # pragma: no cover
        raise PrescriptionError("pytesseract and Pillow are required for scanning.") from exc
    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    try:
        return pytesseract.image_to_string(gray, config="--psm 6")
    except pytesseract.TesseractNotFoundError as exc:
        raise PrescriptionError(
            "Tesseract OCR is not installed. Install it (Mac: 'brew install tesseract') "
            "or enter medicines manually."
        ) from exc


def extract_text(data: bytes, filename: str) -> str:
    """Return the text of an uploaded prescription (image or PDF)."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return _extract_pdf(data)
    if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
        from PIL import Image
        try:
            return _ocr_image(Image.open(io.BytesIO(data)))
        except PrescriptionError:
            raise
        except Exception as exc:
            raise PrescriptionError(f"Could not open image '{filename}': {exc}") from exc
    raise PrescriptionError(f"Unsupported file type: {filename}")


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        raise PrescriptionError(f"Could not read PDF: {exc}") from exc
    if len(text.strip()) >= 30:
        return text
    # Scanned PDF: render each page and OCR it.
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover
        raise PrescriptionError("pypdfium2 is required to scan image-only PDFs.") from exc
    pdf = pdfium.PdfDocument(data)
    pages = []
    for i in range(len(pdf)):
        pil_image = pdf[i].render(scale=2.5).to_pil()
        pages.append(_ocr_image(pil_image))
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_FORM = r"(?:tab(?:let)?s?|cap(?:sule)?s?|syp|syrup|inj(?:ection)?|oint(?:ment)?|drops?|susp(?:ension)?|gel|cream|sachet|inh(?:aler)?)"
_LINE_START_RX = re.compile(rf"^\s*(?:\d+\s*[\.\)\-]\s*)?(?P<form>{_FORM})\b\.?\s*(?P<rest>.+)$", re.IGNORECASE)
_DOSE_RX = re.compile(r"(\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ug|gm|g|ml|iu|units?|%))\b", re.IGNORECASE)
_FREQ_RX = re.compile(
    r"\b(od|bd|bid|tds|tid|qid|qds|hs|sos|prn|stat|once\s+(?:a\s+|per\s+)?daily|twice\s+(?:a\s+|per\s+)?daily|"
    r"thrice\s+(?:a\s+|per\s+)?daily|(?:one|two|three|four|\d)\s+times\s+(?:a\s+|per\s+)?(?:day|daily)|"
    r"every\s+\d+(?:\s*[-\u2013]\s*\d+)?\s*(?:hours?|hrs?)|\d\s*-\s*\d\s*-\s*\d(?:\s*-\s*\d)?)\b",
    re.IGNORECASE,
)
_DURATION_RX = re.compile(r"(?:x|for)?\s*(\d+)\s*(days?|weeks?|months?|wks?)\b", re.IGNORECASE)
_TIMING_RX = re.compile(r"\b(before\s+food|after\s+food|with\s+food|empty\s+stomach|at\s+bedtime|at\s+night|ac|pc)\b", re.IGNORECASE)
_SKIP_RX = re.compile(r"\b(dr\.?|reg\.?\s*no|hospital|clinic|patient|name|age|date|address|phone|signature|mbbs|md)\b", re.IGNORECASE)


def _clean_name(raw: str) -> str:
    # Keep the leading run of letters/spaces/hyphens before the first digit or dose.
    m = re.match(r"([A-Za-z][A-Za-z\-\s]*)", raw.strip())
    name = (m.group(1) if m else raw).strip(" -.,")
    return re.sub(r"\s+", " ", name).title()


_HEADER_WORDS = {
    "name": ("medicine", "medicines", "drug", "medication", "name", "drug name", "medicine name"),
    "dose": ("dose", "dosage", "strength"),
    "frequency": ("frequency", "freq", "how often"),
    "duration": ("duration", "days", "period"),
    "instructions": ("instructions", "instruction", "remarks", "notes", "directions", "advice"),
}


def _header_field(line: str) -> str | None:
    word = line.strip().strip(":").lower()
    for field_name, words in _HEADER_WORDS.items():
        if word in words:
            return field_name
    return None


def _parse_table_cells(lines: list[str], source: str) -> list[Medicine]:
    """Tables extracted from PDFs often come out one cell per line:

        Medicine / Dose / Frequency / Duration / Instructions     (header cells)
        Paracetamol / 500 mg / Every 6-8 hours / 5 days / After food   (one row)

    Find a run of header cells, then read the following lines in groups of that size.
    A row is accepted only if its dose cell looks like a dose, which also stops the
    scan at footer lines such as 'Prescriber: ...'.
    """
    for start in range(len(lines)):
        columns = []
        i = start
        while i < len(lines) and (f := _header_field(lines[i])) and f not in columns:
            columns.append(f)
            i += 1
        if len(columns) < 3 or "name" not in columns or "dose" not in columns:
            continue
        medicines = []
        width = len(columns)
        while i + width <= len(lines):
            row = dict(zip(columns, lines[i:i + width]))
            if not _DOSE_RX.fullmatch(row["dose"].strip().replace("\u00a0", " ")) and not _DOSE_RX.search(row["dose"]):
                break
            medicines.append(Medicine(
                name=_clean_name(row["name"]), dose=re.sub(r"\s+", "", row["dose"]),
                frequency=row.get("frequency", ""), duration=row.get("duration", ""),
                instructions=row.get("instructions", ""), source=source))
            i += width
        if medicines:
            return medicines
    return []


def parse_medicines(text: str, source: str = "") -> list[Medicine]:
    """Extract medicines from prescription text (table cells first, then line heuristics)."""
    cells = [ln.strip() for ln in text.splitlines() if ln.strip()]
    table = _parse_table_cells(cells, source)
    if table:
        return table
    medicines: list[Medicine] = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 4:
            continue
        m = _LINE_START_RX.match(line)
        if m:
            rest = m.group("rest")
            form = m.group("form").rstrip(".").title()
        else:
            # Fallback: a line with BOTH a dose and a frequency is very likely a medicine.
            if _SKIP_RX.search(line) or not (_DOSE_RX.search(line) and _FREQ_RX.search(line)):
                continue
            rest, form = line, ""
        name = _clean_name(rest)
        if len(name) < 3:
            continue
        dose = _DOSE_RX.search(rest)
        freq = _FREQ_RX.search(rest)
        dur = _DURATION_RX.search(rest)
        timing = _TIMING_RX.search(rest)
        medicines.append(Medicine(
            name=f"{form} {name}".strip() if form else name,
            dose=re.sub(r"\s+", "", dose.group(1)) if dose else "",
            frequency=re.sub(r"\s+", " ", freq.group(1)) if freq else "",
            duration=f"{dur.group(1)} {dur.group(2)}" if dur else "",
            instructions=timing.group(1) if timing else "",
            source=source,
        ))
    return medicines


def read_prescription(data: bytes, filename: str) -> tuple[str, list[Medicine]]:
    """Convenience wrapper: bytes -> (raw text, parsed medicines)."""
    text = extract_text(data, filename)
    return text, parse_medicines(text, source=filename)


def medicines_context(medicines: list[dict] | list[Medicine]) -> str:
    """Text block for the prompt."""
    if not medicines:
        return "No prescription uploaded."
    lines = []
    for m in medicines:
        med = m if isinstance(m, Medicine) else Medicine(**{k: v for k, v in m.items() if k in Medicine.__dataclass_fields__})
        if med.name:
            lines.append("- " + med.describe())
    return ("Extracted automatically from uploaded prescriptions, may contain errors:\n"
            + "\n".join(lines)) if lines else "No prescription uploaded."