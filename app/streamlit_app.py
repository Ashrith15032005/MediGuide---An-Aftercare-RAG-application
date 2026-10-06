"""
MediGuide - Streamlit app.

Run:  streamlit run app/streamlit_app.py

Flow (four steps, tracked in st.session_state["stage"]):
    1. intake        health background form
    2. procedure     patient describes the procedure in their own words
    3. prescriptions optional upload of one or more prescriptions (image/PDF)
    4. chat          grounded Q&A with sources

Streamlit reruns this whole script on every interaction; anything that must
survive a rerun lives in st.session_state.
"""
import html
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.config import ConfigError
from src.knowledge_base import warning_signs
from src.patient import PatientProfile, PatientStore
from src.prescriptions import PrescriptionError, read_prescription
from src.procedures import GENERAL_KEY, SPECIFIC_KEYS, label_for, match_all
from src.safety import check_query, check_vitals

st.set_page_config(page_title="MediGuide", page_icon=None, layout="wide")

STEPS = ["Health background", "Your procedure", "Prescriptions", "Ask MediGuide"]
STAGES = ["intake", "procedure", "prescriptions", "chat"]
CONDITION_OPTIONS = ["Diabetes", "High blood pressure", "Heart disease", "Kidney disease",
                     "Asthma or COPD", "Thyroid disorder", "None of these"]

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    h1, h2, h3 {font-weight: 600; letter-spacing: -0.01em;}
    .mg-sub {color: #5b6672; margin-top: -0.6rem; margin-bottom: 1.2rem;}
    .mg-step {padding: 4px 0; color: #8a94a0;}
    .mg-step-active {padding: 4px 0; font-weight: 600; color: #0f5c8c;}
    .mg-note {border-left: 3px solid #0f5c8c; padding: 0.5rem 0.9rem; background: rgba(15,92,140,0.06);
              border-radius: 2px; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "stage": "intake", "profile": PatientProfile(), "medicines": [],
        "history": [], "messages": [], "seen_files": set(), "vitals_alerts": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def goto(stage: str) -> None:
    st.session_state.stage = stage
    st.rerun()


def persist() -> None:
    PatientStore().save(st.session_state.profile, st.session_state.medicines)


@st.cache_resource(show_spinner="Loading the aftercare guide")
def get_assistant():
    from src.rag_chain import MediGuideAssistant
    return MediGuideAssistant()


def show_vitals_alerts() -> None:
    """Readings entered at intake that are outside a safe range stay visible."""
    for level, message in st.session_state.vitals_alerts:
        (st.error if level == "emergency" else st.warning)(message)


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### MediGuide")
        st.caption("Post-discharge care assistant")
        current = STAGES.index(st.session_state.stage)
        for i, label in enumerate(STEPS):
            cls = "mg-step-active" if i == current else "mg-step"
            st.markdown(f"<div class='{cls}'>{i + 1}. {label}</div>", unsafe_allow_html=True)
        profile: PatientProfile = st.session_state.profile
        if profile.procedure_key:
            st.divider()
            st.markdown("**Procedure**")
            st.write(label_for(profile.procedure_key))
            with st.expander("Warning signs: contact your doctor"):
                for sign in warning_signs(profile.procedure_key):
                    st.markdown(f"- {sign}")
                st.caption("If any of these is severe, or you are unsure, seek emergency care.")
        if st.session_state.stage == "chat":
            st.divider()
            if st.button("Edit my details", width="stretch"):
                goto("intake")
        if st.button("Start over", width="stretch"):
            for key in ("stage", "profile", "medicines", "history", "messages", "seen_files", "vitals_alerts"):
                st.session_state.pop(key, None)
            st.rerun()
        st.divider()
        st.caption("MediGuide gives general aftercare information only. It does not diagnose, "
                   "treat or replace your doctor.")


# ---------------------------------------------------------------------------
# Step 1: intake
# ---------------------------------------------------------------------------
def intake_step() -> None:
    st.header("Health background")
    st.markdown("<div class='mg-sub'>This helps tailor general guidance. It stays on this computer.</div>",
                unsafe_allow_html=True)
    p: PatientProfile = st.session_state.profile
    with st.form("intake"):
        c1, c2, c3 = st.columns([2, 1, 1])
        name = c1.text_input("Full name", p.name)
        age = c2.number_input("Age", 0, 120, p.age, step=1)
        sex = c3.selectbox("Sex", ["", "Female", "Male", "Other"],
                           index=["", "Female", "Male", "Other"].index(p.sex) if p.sex in ("", "Female", "Male", "Other") else 0)
        preselected = [c for c in CONDITION_OPTIONS if c in p.conditions]
        chosen = st.multiselect("Existing or past conditions", CONDITION_OPTIONS, default=preselected)
        other = st.text_input("Other conditions (comma separated)",
                              ", ".join(c for c in p.conditions if c not in CONDITION_OPTIONS))
        c4, c5 = st.columns(2)
        allergies = c4.text_input("Allergies (comma separated)", ", ".join(p.allergies))
        other_meds = c5.text_input("Other medicines you take", p.other_medications)
        st.markdown("**Most recent readings, if you have them (leave 0 if not measured)**")
        c6, c7, c8, c9 = st.columns(4)
        sys_bp = c6.number_input("BP systolic", 0, 300, p.bp_systolic)
        dia_bp = c7.number_input("BP diastolic", 0, 200, p.bp_diastolic)
        sugar = c8.number_input("Blood sugar (mg/dL)", 0, 700, p.blood_sugar)
        discharge = c9.date_input("Discharge date", value=date.fromisoformat(p.discharge_date) if p.discharge_date else None)
        submitted = st.form_submit_button("Continue", type="primary")

    if submitted:
        if not name.strip():
            st.error("Please enter your name.")
            return
        if bool(sys_bp) != bool(dia_bp):
            st.error("Enter both systolic and diastolic blood pressure, or leave both as 0.")
            return
        if sys_bp and sys_bp <= dia_bp:
            st.error("Systolic pressure should be higher than diastolic. Please re-check the values.")
            return
        conditions = [c for c in chosen if c != "None of these"] + [
            c.strip() for c in other.split(",") if c.strip()]
        new = PatientProfile.from_lists(
            "", "", patient_id=p.patient_id, name=name.strip(), age=int(age), sex=sex,
            other_medications=other_meds.strip(), bp_systolic=int(sys_bp), bp_diastolic=int(dia_bp),
            blood_sugar=int(sugar), discharge_date=discharge.isoformat() if discharge else "",
            procedure_key=p.procedure_key, procedure_description=p.procedure_description)
        new.conditions = conditions
        new.allergies = [a.strip() for a in allergies.split(",") if a.strip()]
        st.session_state.profile = new
        # Check the readings just entered; tell the patient right away if needed.
        # Stored (not shown here) because st.rerun() below would wipe them.
        st.session_state.vitals_alerts = [
            (i.level, i.message) for i in check_vitals(new.bp_systolic, new.bp_diastolic, new.blood_sugar)]
        persist()
        st.session_state.stage = "procedure"
        st.rerun()


# ---------------------------------------------------------------------------
# Step 2: procedure in the patient's own words
# ---------------------------------------------------------------------------
def procedure_step() -> None:
    st.header("Your procedure")
    show_vitals_alerts()
    st.markdown("<div class='mg-sub'>Describe what you had done, in your own words. Any surgery or "
                "procedure is fine.</div>", unsafe_allow_html=True)
    p: PatientProfile = st.session_state.profile
    text = st.text_area("For example: I had my appendix taken out last week, or a C-section, or a knee "
                        "replacement, or my gallbladder removed.",
                        p.procedure_description, height=110)

    # Not disabled when empty: a text area only commits on blur, so a disabled-until-filled
    # button would swallow the first click.
    if st.button("Continue with this description", type="primary"):
        if not text.strip():
            st.error("Please describe your surgery or procedure first.")
            return
        p.procedure_description = text.strip()
        matches = [m for m in match_all(text) if m.key in SPECIFIC_KEYS]
        top = [m for m in matches if matches and m.score == matches[0].score]
        if len(top) == 1:
            p.procedure_key = top[0].key
            st.session_state.pop("procedure_choices", None)
        elif len(top) > 1:
            # Two guides fit equally well: let the patient pick between just those.
            p.procedure_key = ""
            st.session_state.procedure_choices = [m.key for m in top]
        else:
            # No specific guide: use the general recovery guide, with their own words as context.
            p.procedure_key = GENERAL_KEY
            st.session_state.pop("procedure_choices", None)

    choices = st.session_state.get("procedure_choices")
    if choices:
        picked = st.radio("Your description fits more than one guide. Which is your main reason for care?",
                          choices, index=None, format_func=label_for)
        if picked:
            p.procedure_key = picked
            st.session_state.pop("procedure_choices", None)
            st.rerun()
    elif p.procedure_key == GENERAL_KEY:
        st.info("I do not have a guide written specifically for this procedure, so I will use the general "
                "recovery guide. It covers wound care, rest, diet, medicines and follow-up in general terms. "
                "For anything specific to your operation (lifting limits, movement rules, timelines), "
                "please follow your surgeon's instructions.")
    elif p.procedure_key:
        st.success(f"Matched: {label_for(p.procedure_key)}")

    c1, c2 = st.columns([1, 6])
    if c1.button("Back"):
        goto("intake")
    if c2.button("Continue", disabled=not p.procedure_key, type="primary"):
        persist()
        goto("prescriptions")


# ---------------------------------------------------------------------------
# Step 3: prescriptions
# ---------------------------------------------------------------------------
def prescriptions_step() -> None:
    st.header("Prescriptions")
    show_vitals_alerts()
    st.markdown("<div class='mg-sub'>Optional. Upload one or more prescriptions (photo or PDF). "
                "Medicines are used as reference only.</div>", unsafe_allow_html=True)
    files = st.file_uploader("Upload prescriptions", type=["png", "jpg", "jpeg", "pdf"],
                             accept_multiple_files=True)
    for f in files or []:
        marker = (f.name, f.size)
        if marker in st.session_state.seen_files:
            continue
        try:
            with st.spinner(f"Reading {f.name}"):
                _, meds = read_prescription(f.getvalue(), f.name)
        except PrescriptionError as exc:
            st.error(f"{f.name}: {exc}")
            continue
        st.session_state.seen_files.add(marker)
        if meds:
            st.session_state.medicines.extend(m.to_dict() for m in meds)
            st.success(f"{f.name}: found {len(meds)} medicine(s).")
        else:
            st.warning(f"{f.name}: no medicines could be recognised. Add them manually in the table below.")

    st.markdown("**Medicines list**")
    st.caption("Automatic reading can make mistakes. Check the table against your prescription, "
               "correct it, or add rows.")
    columns = ["name", "dose", "frequency", "duration", "instructions", "source"]
    df = pd.DataFrame(st.session_state.medicines, columns=columns)
    edited = st.data_editor(df, num_rows="dynamic", width="stretch", hide_index=True,
                            column_config={"source": st.column_config.TextColumn(disabled=True)})
    st.session_state.medicines = [
        {k: ("" if pd.isna(v) else str(v)) for k, v in row.items()}
        for row in edited.to_dict("records") if str(row.get("name") or "").strip()
    ]

    c1, c2 = st.columns([1, 6])
    if c1.button("Back"):
        goto("procedure")
    if c2.button("Continue", type="primary"):
        persist()
        goto("chat")


# ---------------------------------------------------------------------------
# Step 4: chat
# ---------------------------------------------------------------------------
def render_answer_extras(msg: dict) -> None:
    if msg.get("sources"):
        with st.expander("Sources"):
            for s in msg["sources"]:
                st.markdown(f"**{s['label']}**  {label_for(s['procedure'])}, {s['section']}")
                st.caption(s["snippet"])


def chat_step() -> None:
    p: PatientProfile = st.session_state.profile
    st.header("Ask MediGuide")
    show_vitals_alerts()
    heading = (f"Recovery after: {p.procedure_description[:120]} (general guide)"
               if p.procedure_key == GENERAL_KEY and p.procedure_description else label_for(p.procedure_key))
    st.markdown(f"<div class='mg-sub'>{html.escape(heading)}</div>", unsafe_allow_html=True)
    st.markdown("<div class='mg-note'>I can help with diet, activity, daily routine, wound care basics and "
                "follow-up planning. For symptoms, medicine changes or anything worrying, contact your "
                "doctor. In an emergency, call your local emergency number.</div>", unsafe_allow_html=True)
    st.write("")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])
            render_answer_extras(msg)

    question = st.chat_input("Ask about your recovery")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.write(question)
    # Safety answers need no API key or vector store, so show them even if the
    # assistant cannot start.
    safety = check_query(question)
    if safety.blocked:
        with st.chat_message("assistant"):
            (st.error if safety.level == "emergency" else st.warning)(safety.message)
        st.session_state.messages.append(
            {"role": "assistant", "text": safety.message, "level": safety.level, "sources": []})
        return
    try:
        assistant = get_assistant()
    except ConfigError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Could not start the assistant: {exc}. Have you run 'python -m src.ingest'?")
        return

    with st.chat_message("assistant"):
        with st.spinner("Checking the guide"):
            ans = assistant.answer(question, p.procedure_key, profile=p,
                                   medicines=st.session_state.medicines,
                                   history=st.session_state.history)
        if ans.error:
            st.error(ans.text)
            with st.expander("Technical details (for the developer)"):
                st.code(ans.error_detail or "no detail captured")
                st.caption("Run 'python -m src.diagnose' in the project folder to test the API key, "
                           "embedding model and language model one by one.")
        elif ans.safety_level == "emergency":
            st.error(ans.text)
        elif ans.safety_level in ("urgent", "refer"):
            st.warning(ans.text)
        else:
            st.write(ans.text)
        msg = {"role": "assistant", "text": ans.text, "level": "error" if ans.error else ans.safety_level,
               "sources": [s.__dict__ for s in ans.sources]}
        render_answer_extras(msg)
    st.session_state.messages.append(msg)
    if not ans.refused and not ans.error:
        st.session_state.history.extend([("user", question), ("assistant", ans.text)])


def main() -> None:
    init_state()
    sidebar()
    {"intake": intake_step, "procedure": procedure_step,
     "prescriptions": prescriptions_step, "chat": chat_step}[st.session_state.stage]()


main()