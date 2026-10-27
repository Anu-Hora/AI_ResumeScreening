# AI Resume Screening Assistant

An AI-powered resume screening application built with LangChain, RAG, FAISS, OpenAI, and Streamlit.

## Features

- Upload one or more PDF resumes
- Enter a Job Description
- Extract PDF content using `PyPDFLoader`
- Split resume text with `RecursiveCharacterTextSplitter`
- Create embeddings using OpenAI embeddings
- Store each candidate's chunks in a FAISS vector database
- Retrieve JD-relevant resume evidence
- Evaluate candidates with GPT-4o-mini
- Return structured results using Pydantic
- Compare multiple candidates by match score
- Identify matching and missing skills
- Generate strengths, weaknesses, summary, and hiring recommendation

## RAG design

Each uploaded resume gets its own temporary FAISS vector store. This prevents evidence from one candidate being retrieved while evaluating another candidate.

The Job Description is used as the retrieval query and evaluation criteria. Candidate facts must come from retrieved resume evidence.

## Setup with Anaconda

Recommended: use a dedicated Conda environment.

```bash
conda create -n resume_ai python=3.11 -y
conda activate resume_ai
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add your OpenAI API key.

## Run

```bash
conda activate resume_ai
cd /d D:\EICT_AI_DS_ML\Python\Practice_work\Capstone3\AI_Resume_Screening_Assistant
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## Assignment test cases

### Test Case 1 — Single candidate

JD: Data Scientist requiring Python, SQL, pandas, scikit-learn, machine learning, statistics and communication.

Upload: Resume A.

Expected demonstration: match score, matching skills, missing skills, summary, strengths, weaknesses and recommendation.

### Test Case 2 — Candidate comparison

Use the same Data Scientist JD.

Upload: Resume A and Resume B.

Expected demonstration: both candidates are evaluated independently and ranked by match score.

### Test Case 3 — Missing skill analysis

Use a JD requiring Python, SQL, AWS, Spark, ML and Docker.

Upload: Resume C.

Expected demonstration: skills present in the retrieved resume evidence are matched; skills not supported by the resume are identified as missing/unknown.

### Additional demonstrations

- Recommend the best candidate among multiple resumes.
- Generate a hiring recommendation with evidence-based justification.

## Important limitation

This application is an AI-assisted screening tool, not an autonomous hiring decision system. Human review should be used for final hiring decisions. The model is instructed not to invent candidate qualifications and to ground candidate-specific claims in retrieved resume evidence.
