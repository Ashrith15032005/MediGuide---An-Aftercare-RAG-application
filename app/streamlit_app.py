"""
MediGuide -- Streamlit Demo App
--------------------------------
Run from the project root with:
    streamlit run app/streamlit_app.py

Requires ANTHROPIC_API_KEY in the environment for live answers. Works
without it too (uncheck "Use live LLM") to demo retrieval/safety/history.

Flow:
    1. Patient Intake  -- basic health background (age, BP, existing
       conditions, allergies, past surgeries) and prescription upload(s)
    2. Chat             -- grounded Q&A, personalized using the intake data
"""

import os
import sys
import uuid

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.rag_chain import list_procedures, answer_query
from src.patient_history import PatientHistory
from src.prescription_ocr import process_prescription

st.set_page_config(page_title="MediGuide", page_icon=None, layout="centered")

if "patient_id" not in st.session_state:
    st.session_state.patient_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "intake_complete" not in st.session_state:
    st.session_state.intake_complete = False

history = PatientHistory(st.session_state.patient_id)

st.title("MediGuide")
st.caption("A grounded assistant for post-discharge aftercare questions. Not a diagnostic tool.")

# --------------------------------------------------------------------------
# STEP 1: PATIENT INTAKE
# --------------------------------------------------------------------------
if not st.session_state.intake_complete:
    st.subheader("Patient Intake")
    st.write(
        "Before we begin, please share a little about your health background. "
        "This helps MediGuide tailor its answers to you specifically."
    )

    with st.form("intake_form"):
        procedures = list_procedures()
        procedure = st.selectbox(
            "Select your procedure / condition",
            options=procedures,
            format_func=lambda p: config.PROCEDURE_DISPLAY_NAMES.get(p, p),
        )

        col1, col2 = st.columns(2)
        with col1:
            age = st.text_input("Age")
            blood_pressure = st.text_input("Typical blood pressure (e.g. 120/80)")
        with col2:
            existing_conditions = st.text_input(
                "Existing conditions (e.g. diabetes, hypertension, asthma)"
            )
            allergies = st.text_input("Known allergies")

        past_surgeries = st.text_area("Past surgeries or major medical history (optional)")

        st.divider()
        st.write("Upload prescription(s)")
        uploaded_files = st.file_uploader(
            "Upload photos of all your current prescriptions",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        submitted = st.form_submit_button("Continue to MediGuide")

    if submitted:
        history.set_procedure(procedure)
        history.set_health_background({
            "age": age,
            "blood_pressure": blood_pressure,
            "existing_conditions": existing_conditions,
            "allergies": allergies,
            "past_surgeries": past_surgeries,
        })

        if uploaded_files:
            with st.spinner("Reading prescriptions..."):
                for i, uploaded in enumerate(uploaded_files):
                    temp_path = f"/tmp/{st.session_state.patient_id}_rx_{i}.png"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded.getbuffer())
                    result = process_prescription(temp_path)
                    history.add_prescription(result)

        st.session_state.procedure = procedure
        st.session_state.intake_complete = True
        st.rerun()

# --------------------------------------------------------------------------
# STEP 2: CHAT
# --------------------------------------------------------------------------
else:
    with st.sidebar:
        st.header("Patient Summary")
        st.write("Procedure:", config.PROCEDURE_DISPLAY_NAMES.get(
            st.session_state.procedure, st.session_state.procedure
        ))

        bg = history.data.get("health_background") or {}
        if any(bg.values()):
            st.write("Health background:")
            if bg.get("age"):
                st.write("- Age:", bg["age"])
            if bg.get("blood_pressure"):
                st.write("- Blood pressure:", bg["blood_pressure"])
            if bg.get("existing_conditions"):
                st.write("- Existing conditions:", bg["existing_conditions"])
            if bg.get("allergies"):
                st.write("- Allergies:", bg["allergies"])
            if bg.get("past_surgeries"):
                st.write("- Past history:", bg["past_surgeries"])

        num_rx = len(history.data.get("prescriptions", []))
        if num_rx:
            st.write(f"Prescriptions on file: {num_rx}")
            with st.expander("View extracted medicines"):
                for rx in history.data["prescriptions"]:
                    for m in rx.get("medicines", []):
                        st.write("-", m)

        st.divider()
        use_llm = st.checkbox(
            "Use live LLM (requires ANTHROPIC_API_KEY)",
            value=bool(os.environ.get("ANTHROPIC_API_KEY")),
        )
        st.caption(f"Session ID: {st.session_state.patient_id}")

        if st.button("Restart intake"):
            st.session_state.intake_complete = False
            st.session_state.messages = []
            st.rerun()

    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                st.caption("Sources: " + ", ".join(msg["sources"]))

    query = st.chat_input("Ask a question about your diet, activity, sleep, or care...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = answer_query(
                    st.session_state.patient_id,
                    st.session_state.procedure,
                    query,
                    use_llm=use_llm,
                )
            st.write(result["answer"])
            if result["sources"]:
                st.caption("Sources: " + ", ".join(result["sources"]))
            if result["refused"]:
                st.warning("Safety layer triggered -- redirected to a doctor instead of answering.")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })