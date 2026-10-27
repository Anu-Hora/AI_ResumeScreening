# Demonstration Test Cases

## Test Case 1: Data Scientist — Resume A

**JD**

Data Scientist with Python, SQL, pandas, scikit-learn, statistics, machine learning, data visualization and strong communication skills.

**Action**

Upload `resume_A.pdf` and run evaluation.

**Record**

- Match Score
- Matching Skills
- Missing Skills
- Candidate Summary
- Strengths
- Weaknesses
- Recommendation
- Evidence

## Test Case 2: Compare Resume A vs Resume B

Use the same JD as Test Case 1 and upload both PDFs.

**Expected demonstration**

The application produces an independent RAG evaluation for each candidate and displays a ranking table.

## Test Case 3: Missing Skills — Resume C

**JD**

Machine Learning Engineer requiring Python, SQL, AWS, Docker, Kubernetes, Spark, MLflow and model deployment.

Upload `resume_C.pdf`.

**Expected demonstration**

The application highlights the skills supported by resume evidence and identifies requirements that are not supported by the retrieved evidence.
