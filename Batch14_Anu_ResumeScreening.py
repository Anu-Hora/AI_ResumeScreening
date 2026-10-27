import os
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()


class ResumeEvaluation(BaseModel):
    match_score: int = Field(ge=0, le=100, description="Overall match score from 0 to 100")
    candidate_summary: str
    matching_skills: List[str]
    missing_skills: List[str]
    strengths: List[str]
    weaknesses: List[str]
    hiring_recommendation: str
    recommendation_justification: str
    evidence: List[str] = Field(description="Short evidence statements grounded only in retrieved resume text")


def load_resume(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def build_retriever(pdf_path: str, k: int = 8):
    documents = load_resume(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def evaluate_resume(pdf_path: str, job_description: str, model: str = "gpt-4o-mini") -> ResumeEvaluation:
    retriever = build_retriever(pdf_path)
    retrieved_docs = retriever.invoke(job_description)

    context = "\n\n--- RESUME EVIDENCE ---\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
You are a resume screening assistant.

Evaluate ONE candidate against the supplied Job Description.

IMPORTANT RULES:
1. Candidate facts must come ONLY from the RESUME EVIDENCE below.
2. Do not invent skills, experience, education, certifications, employers, projects,
   years of experience, or achievements that are not supported by the evidence.
3. The Job Description is evaluation criteria, not evidence about the candidate.
4. If a skill is not found in the resume evidence, treat it as missing/unknown rather
   than assuming the candidate has it.
5. Give a match score from 0-100 based on the evidence available in the resume.
6. Keep evidence concise and directly traceable to the retrieved resume content.
7. Hiring recommendation should be one of: Strong Hire, Hire, Consider, Reject.

JOB DESCRIPTION:
{job_description}

RESUME EVIDENCE:
{context}
"""

    llm = ChatOpenAI(model=model, temperature=0)
    structured_llm = llm.with_structured_output(ResumeEvaluation)
    return structured_llm.invoke(prompt)
