"""
MediGuide -- Streamlit Web Application
--------------------------------------
Run from the project root with:
    streamlit run app/streamlit_app.py

Flow:
    1. Patient Intake: Health background (age, sex, conditions, allergies, BP, sugar, discharge date)
       and procedure description/selection, plus prescription upload (image/PDF OCR).
    2. Interactive Chat: Grounded aftercare assistant powered by LangChain LCEL & Gemini,
       with deterministic rule-based safety screening.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import GOOGLE_API_KEY_ENV
from src.patient import PatientProfile, PatientStore
from src.prescriptions import Medicine, read_prescription
from src.procedures import PROCEDURES, label_for, match_procedure
from src.rag_chain import MediGuideAssistant

st.set_page_config(
    page_title="MediGuide - Post-Discharge Care Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, accessible UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .safety-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .safety-emergency {
        background-color: #FEE2E2;
        border-left: 5px solid #DC2626;
        color: #991B1B;
    }
    .safety-urgent {
        background-color: #FEF3C7;
        border-left: 5px solid #D97706;
        color: #92400E;
    }
    .safety-refer {
        background-color: #DBEAFE;
        border-left: 5px solid #2563EB;
        color: #1E40AF;
    }
    .source-badge {
        display: inline-block;
        background-color: #F3F4F6;
        border: 1px solid #E5E7EB;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin-right: 0.3rem;
        margin-top: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "patient_id" not in st.session_state:
    st.session_state.patient_id = "patient_demo"
if "profile" not in st.session_state:
    st.session_state.profile = PatientProfile(patient_id=st.session_state.patient_id)
if "medicines" not in st.session_state:
    st.session_state.medicines = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "intake_complete" not in st.session_state:
    st.session_state.intake_complete = False

patient_store = PatientStore()


def load_assistant() -> MediGuideAssistant:
    return MediGuideAssistant()


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🩺 Patient Dashboard")
    st.caption(f"Session ID: `{st.session_state.patient_id}`")

    if st.session_state.intake_complete and st.session_state.profile.procedure_key:
        proc_key = st.session_state.profile.procedure_key
        st.success(f"**Procedure:** {label_for(proc_key)}")

        with st.expander("👤 Health Background", expanded=False):
            p = st.session_state.profile
            if p.age:
                st.write(f"- **Age:** {p.age}")
            if p.sex:
                st.write(f"- **Sex:** {p.sex}")
            if p.conditions:
                st.write(f"- **Conditions:** {', '.join(p.conditions)}")
            if p.allergies:
                st.write(f"- **Allergies:** {', '.join(p.allergies)}")
            if p.bp_systolic and p.bp_diastolic:
                st.write(f"- **BP:** {p.bp_systolic}/{p.bp_diastolic}")
            if p.blood_sugar:
                st.write(f"- **Blood Sugar:** {p.blood_sugar} mg/dL")
            if p.days_since_discharge() is not None:
                st.write(f"- **Days since discharge:** {p.days_since_discharge()}")

        if st.session_state.medicines:
            with st.expander(f"💊 Prescriptions ({len(st.session_state.medicines)})", expanded=False):
                for m in st.session_state.medicines:
                    med = m if isinstance(m, Medicine) else Medicine(**m)
                    st.write(f"- **{med.name}** {med.dose} ({med.frequency})")

        st.divider()
        if st.button("✏️ Edit Intake & Prescriptions", use_container_width=True):
            st.session_state.intake_complete = False
            st.rerun()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("Complete the intake form to start tailored recovery chat.")

    st.divider()
    has_api_key = bool(os.environ.get(GOOGLE_API_KEY_ENV))
    if has_api_key:
        st.caption("✅ Google Gemini API key configured")
    else:
        st.caption("⚠️ `GOOGLE_API_KEY` or `GEMINI_API_KEY` missing. Please set it in your environment or .env file.")


# -----------------------------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">MediGuide</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Grounded post-discharge aftercare guidance. Not a substitute for emergency care or medical diagnosis.</div>',
    unsafe_allow_html=True,
)

if not st.session_state.intake_complete:
    # -------------------------------------------------------------------------
    # STEP 1: PATIENT INTAKE FORM
    # -------------------------------------------------------------------------
    st.subheader("📋 Step 1: Health Background & Intake")
    st.write(
        "Please provide your procedure details and health background. MediGuide uses this context "
        "to personalize recovery information (e.g. dietary precautions or exercise pace)."
    )

    with st.form("patient_intake_form"):
        st.markdown("#### 1. Procedure Information")
        proc_mode = st.radio(
            "How would you like to specify your procedure?",
            ["Choose from list", "Describe in your own words"],
            horizontal=True,
        )

        procedure_key = ""
        procedure_desc = ""
        if proc_mode == "Choose from list":
            proc_options = list(PROCEDURES.keys())
            procedure_key = st.selectbox(
                "Select procedure / condition:",
                options=proc_options,
                format_func=label_for,
            )
        else:
            procedure_desc = st.text_input(
                "Describe what procedure or surgery you had (e.g., 'had my appendix removed', 'c-section delivery'):",
                value="",
            )

        st.markdown("#### 2. General Health Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            age_val = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.profile.age or 0)
        with c2:
            sex_val = st.selectbox("Sex", ["", "Female", "Male", "Other"], index=0)
        with c3:
            discharge_date_val = st.date_input("Discharge Date (optional)", value=None)

        c4, c5 = st.columns(2)
        with c4:
            conditions_input = st.text_area(
                "Existing conditions (comma-separated):",
                value=", ".join(st.session_state.profile.conditions),
                placeholder="e.g. Type 2 Diabetes, Hypertension, Asthma",
            )
        with c5:
            allergies_input = st.text_area(
                "Known allergies (comma-separated):",
                value=", ".join(st.session_state.profile.allergies),
                placeholder="e.g. Penicillin, Sulfa drugs, Peanuts",
            )

        c6, c7 = st.columns(2)
        with c6:
            bp_input = st.text_input("Recent Blood Pressure (e.g. 120/80):", placeholder="120/80")
        with c7:
            sugar_input = st.number_input("Recent Blood Sugar (mg/dL):", min_value=0, max_value=800, value=0)

        st.markdown("#### 3. Prescriptions & Discharge Medications (Optional)")
        uploaded_files = st.file_uploader(
            "Upload photos or PDFs of your prescriptions:",
            type=["png", "jpg", "jpeg", "pdf"],
            accept_multiple_files=True,
        )

        submit_intake = st.form_submit_button("Continue to Recovery Assistant ➡️", use_container_width=True)

    if submit_intake:
        # Resolve procedure
        final_proc_key = procedure_key
        if proc_mode == "Describe in your own words" and procedure_desc:
            match = match_procedure(procedure_desc)
            if match:
                final_proc_key = match.key
                st.success(f"Matched procedure: **{label_for(final_proc_key)}**")
            else:
                st.error(
                    f"Could not match '{procedure_desc}' to a supported procedure. "
                    f"Supported topics: {', '.join(label_for(k) for k in PROCEDURES)}"
                )
                st.stop()

        # Parse BP
        bp_sys, bp_dia = 0, 0
        if bp_input and "/" in bp_input:
            try:
                parts = bp_input.split("/")
                bp_sys, bp_dia = int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                pass

        # Update profile
        prof = PatientProfile.from_lists(
            conditions_text=conditions_input,
            allergies_text=allergies_input,
            patient_id=st.session_state.patient_id,
            age=int(age_val),
            sex=sex_val,
            bp_systolic=bp_sys,
            bp_diastolic=bp_dia,
            blood_sugar=int(sugar_input),
            discharge_date=discharge_date_val.isoformat() if discharge_date_val else "",
            procedure_key=final_proc_key,
            procedure_description=procedure_desc,
        )
        st.session_state.profile = prof

        # Process prescriptions
        parsed_medicines = []
        if uploaded_files:
            with st.spinner("Processing prescription files with OCR..."):
                for uploaded in uploaded_files:
                    try:
                        _, meds = read_prescription(uploaded.getvalue(), uploaded.name)
                        for m in meds:
                            parsed_medicines.append(m.to_dict())
                    except Exception as e:
                        st.warning(f"Could not scan {uploaded.name}: {e}")

        st.session_state.medicines = parsed_medicines
        patient_store.save(prof, parsed_medicines)
        st.session_state.intake_complete = True
        st.rerun()

else:
    # -------------------------------------------------------------------------
    # STEP 2: CHAT INTERFACE
    # -------------------------------------------------------------------------
    st.info(
        f"**Active Procedure:** {label_for(st.session_state.profile.procedure_key)} | "
        "Ask about diet, activity, recovery routine, wound care basics, or follow-up timelines."
    )

    # Display conversation messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Verified Knowledge Base Sources", expanded=False):
                    for s in msg["sources"]:
                        st.markdown(
                            f"- **{s.get('label', 'Source')}**: `{s.get('procedure', '')}` > `{s.get('section', '')}` "
                            f"(relevance: {s.get('score', 0):.2f})\n"
                            f"  > *\"{s.get('snippet', '')}\"*"
                        )

    # Chat input
    user_query = st.chat_input("Ask a question about your diet, wound care, activity, or follow-up...")
    if user_query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Reviewing aftercare guidelines..."):
                try:
                    assistant = load_assistant()
                    # Convert history
                    hist = [
                        (m["role"], m["content"])
                        for m in st.session_state.messages[:-1]
                        if m["role"] in ("user", "assistant")
                    ]
                    ans = assistant.answer(
                        question=user_query,
                        procedure=st.session_state.profile.procedure_key,
                        profile=st.session_state.profile,
                        medicines=st.session_state.medicines,
                        history=hist,
                    )

                    if ans.refused:
                        level_class = f"safety-{ans.safety_level}"
                        st.markdown(
                            f'<div class="safety-card {level_class}">'
                            f'<strong>[{ans.safety_level.upper()}] Safety Guidance:</strong><br>{ans.text}</div>',
                            unsafe_allow_html=True,
                        )
                        reply_text = ans.text
                        sources_data = []
                    elif ans.error:
                        st.error(ans.text)
                        reply_text = ans.text
                        sources_data = []
                    else:
                        st.markdown(ans.text)
                        sources_data = [
                            {
                                "label": s.label,
                                "procedure": s.procedure,
                                "section": s.section,
                                "snippet": s.snippet,
                                "score": s.score,
                            }
                            for s in ans.sources
                        ]
                        if sources_data:
                            with st.expander("📚 Verified Knowledge Base Sources", expanded=False):
                                for s in sources_data:
                                    st.markdown(
                                        f"- **{s['label']}**: `{s['procedure']}` > `{s['section']}` "
                                        f"(relevance: {s['score']:.2f})\n"
                                        f"  > *\"{s['snippet']}\"*"
                                    )
                        reply_text = ans.text

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply_text,
                        "sources": sources_data,
                    })
                except Exception as ex:
                    err_msg = f"An error occurred: {ex}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
