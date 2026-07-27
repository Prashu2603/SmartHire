"""Resume text parsing utilities.

Supports parsing resume text from:
- Raw text input (pasted resume content)
- PDF files (using pdfplumber)
"""

import re
from typing import Optional


def parse_resume_text(text: str) -> str:
    """Clean and normalize pasted resume text.

    Parameters
    ----------
    text : str
        Raw resume text pasted by the user.

    Returns
    -------
    str
        Cleaned resume text ready for feature extraction.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove common resume formatting artifacts
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def parse_resume_pdf(file) -> str:
    """Extract text from a PDF resume file using pdfplumber.

    Parameters
    ----------
    file : file-like or str
        A file path (str) or a file-like object (e.g., Streamlit UploadedFile)
        pointing to a PDF file.

    Returns
    -------
    str
        Extracted text from the PDF.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF parsing. "
            "Install it with: pip install pdfplumber"
        )

    text_parts = []

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n".join(text_parts)


def extract_skills_from_text(text: str) -> list:
    """Extract potential skill keywords from resume text.

    Uses a curated list of common technical and soft skills
    to identify skills mentioned in the resume.

    Parameters
    ----------
    text : str
        Resume text (raw or cleaned).

    Returns
    -------
    list of str
        Sorted list of detected skills.
    """
    if not text:
        return []

    text_lower = text.lower()

    # Comprehensive skill keywords list
    skill_keywords = sorted(set([
        # Programming Languages
        "python", "java", "javascript", "c++", "c#", "r", "sql", "php",
        "ruby", "go", "scala", "swift", "kotlin", "typescript", "matlab",
        "sas", "perl", "rust", "haskell", "lua",
        # Web Technologies
        "html", "css", "react", "angular", "vue", "node.js", "django",
        "flask", "fastapi", "spring", "express", "bootstrap", "jquery",
        "rest api", "graphql",
        # Data Science & ML
        "machine learning", "deep learning", "natural language processing",
        "nlp", "computer vision", "tensorflow", "keras", "pytorch",
        "scikit-learn", "sklearn", "xgboost", "pandas", "numpy",
        "scipy", "matplotlib", "seaborn", "plotly", "tableau",
        "data analysis", "data visualization", "statistical modeling",
        "regression", "classification", "clustering", "neural network",
        "convolutional neural network", "cnn", "recurrent neural network",
        "rnn", "lstm", "transformer", "bert", "gpt",
        # Data Engineering
        "hadoop", "spark", "hive", "kafka", "airflow", "etl",
        "data pipeline", "big data", "sql", "nosql", "mongodb",
        "postgresql", "mysql", "oracle", "cassandra", "redis",
        "elasticsearch", "databricks", "snowflake", "redshift",
        # Cloud & DevOps
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "jenkins", "git", "github", "gitlab", "ci/cd", "terraform",
        "ansible", "linux", "bash", "shell scripting",
        # Business & Management
        "project management", "agile", "scrum", "jira", "leadership",
        "communication", "team management", "strategic planning",
        "business analysis", "requirements gathering", "stakeholder management",
        "budgeting", "forecasting",
        # Accounting & Finance
        "accounting", "financial analysis", "budgeting", "forecasting",
        "accounts payable", "accounts receivable", "general ledger",
        "tax preparation", "audit", "compliance", "gaap", "ifrs",
        "quickbooks", "sap", "erp",
        # Engineering
        "cad", "autocad", "solidworks", "matlab", "simulink",
        "circuit design", "pcb design", "embedded systems",
        "iot", "robotics", "plc", "scada",
        # Soft Skills
        "problem solving", "critical thinking", "teamwork",
        "time management", "attention to detail", "adaptability",
        "creativity", "analytical thinking",
        # Marketing
        "digital marketing", "seo", "sem", "social media marketing",
        "content marketing", "email marketing", "google analytics",
        "crm", "salesforce", "market research",
    ]))

    detected = []
    for skill in skill_keywords:
        # Token boundaries prevent false positives such as the language "r"
        # matching every ordinary word that contains that letter.
        pattern = rf"(?<!\w){re.escape(skill)}(?!\w)"
        if re.search(pattern, text_lower):
            detected.append(skill)

    return sorted(set(detected))
