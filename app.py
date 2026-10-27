import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import evaluate_resume

load_dotenv()

st.set_page_config(page_title="AI Resume Screening Assistant", page_icon="📄", layout="wide")

st.title("📄 AI Resume Screening Assistant")
st.caption("LangChain + RAG + FAISS + OpenAI + Streamlit")

with st.sidebar:
    st.header("Configuration")
    model = st.text_input("OpenAI Model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    st.info("Candidate evidence is retrieved from each uploaded resume separately.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing. Add it to the .env file in the project root.")
    st.stop()

job_description = st.text_area(
    "Job Description",
    height=220,
    placeholder="Paste the complete Job Description here...",
)

uploaded_files = st.file_uploader(
    "Upload one or more Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button("🔍 Evaluate Candidates", type="primary", use_container_width=True):
    if not job_description.strip():
        st.warning("Please enter a Job Description.")
        st.stop()
    if not uploaded_files:
        st.warning("Please upload at least one PDF resume.")
        st.stop()

    results = []
    progress = st.progress(0)

    for i, uploaded_file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            pdf_path = tmp.name

        try:
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                evaluation = evaluate_resume(pdf_path, job_description, model=model)
            results.append((uploaded_file.name, evaluation))
        except Exception as exc:
            st.error(f"Failed to analyze {uploaded_file.name}: {exc}")
        finally:
            Path(pdf_path).unlink(missing_ok=True)

        progress.progress((i + 1) / len(uploaded_files))

    if not results:
        st.stop()

    st.subheader("Candidate Evaluation")

    ranking = sorted(results, key=lambda x: x[1].match_score, reverse=True)
    ranking_rows = [
        {
            "Candidate": name,
            "Match Score": evaluation.match_score,
            "Recommendation": evaluation.hiring_recommendation,
        }
        for name, evaluation in ranking
    ]
    st.dataframe(ranking_rows, use_container_width=True, hide_index=True)

    if len(ranking) > 1:
        st.success(f"Top-ranked candidate: {ranking[0][0]} ({ranking[0][1].match_score}/100)")

    for name, evaluation in results:
        with st.expander(f"{name} — {evaluation.match_score}/100 — {evaluation.hiring_recommendation}", expanded=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Match Score", f"{evaluation.match_score}/100")
                st.write("**Recommendation**")
                st.write(evaluation.hiring_recommendation)
            with c2:
                st.write("**Candidate Summary**")
                st.write(evaluation.candidate_summary)

            st.write("### Matching Skills")
            st.write(", ".join(evaluation.matching_skills) if evaluation.matching_skills else "None identified")

            st.write("### Missing Skills")
            st.write(", ".join(evaluation.missing_skills) if evaluation.missing_skills else "None identified")

            col1, col2 = st.columns(2)
            with col1:
                st.write("### Strengths")
                for item in evaluation.strengths:
                    st.write(f"- {item}")
            with col2:
                st.write("### Weaknesses")
                for item in evaluation.weaknesses:
                    st.write(f"- {item}")

            st.write("### Recommendation Justification")
            st.write(evaluation.recommendation_justification)

            st.write("### Resume Evidence")
            for item in evaluation.evidence:
                st.write(f"- {item}")

            st.write("### Structured Output")
            st.json(evaluation.model_dump())
