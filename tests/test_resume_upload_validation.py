"""Security and content tests for PDF resume uploads."""

from io import BytesIO

import matplotlib.pyplot as plt
import pytest
from docx import Document

from src.parsing.resume_parser import (
    ResumeValidationError,
    detect_candidate_name,
    validate_resume_docx_upload,
    validate_resume_pdf_upload,
    validate_resume_text,
    validate_resume_upload,
)


VALID_RESUME = """Prasanth Veluri
prasanth@example.com | +91 98765 43210

Professional Summary
Data analyst with 2 years of experience building business dashboards.

Education
B.Tech in Computer Science, 2023

Technical Skills
Python, SQL, Pandas, Tableau and machine learning

Experience
Data Analyst, Example Technologies, 2023 - Present

Projects
Built a customer churn classification dashboard using Python and SQL.
"""


class UploadedBytes(BytesIO):
    """Small stand-in for Streamlit's UploadedFile."""

    def __init__(self, data: bytes, name: str, mime_type: str):
        super().__init__(data)
        self.name = name
        self.type = mime_type


def make_text_pdf(text: str) -> bytes:
    """Create a real text-based PDF entirely in memory."""
    output = BytesIO()
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(0.08, 0.94, text, va="top", fontsize=10)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output.getvalue()


def make_blank_pdf() -> bytes:
    output = BytesIO()
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output.getvalue()


def make_docx(text: str) -> bytes:
    output = BytesIO()
    document = Document()
    for line in text.splitlines():
        document.add_paragraph(line)
    document.save(output)
    return output.getvalue()


def test_accepts_genuine_resume_pdf():
    upload = UploadedBytes(make_text_pdf(VALID_RESUME), "resume.pdf", "application/pdf")
    extracted = validate_resume_pdf_upload(upload)
    assert "prasanth@example.com" in extracted.lower()
    assert "Technical Skills" in extracted


def test_accepts_genuine_resume_docx():
    upload = UploadedBytes(
        make_docx(VALID_RESUME),
        "resume.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    extracted = validate_resume_docx_upload(upload)
    assert "prasanth@example.com" in extracted.lower()
    assert "Technical Skills" in extracted


def test_dispatches_supported_upload_types():
    pdf = UploadedBytes(make_text_pdf(VALID_RESUME), "resume.pdf", "application/pdf")
    docx = UploadedBytes(
        make_docx(VALID_RESUME),
        "resume.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert "Professional Summary" in validate_resume_upload(pdf)
    assert "Professional Summary" in validate_resume_upload(docx)


def test_rejects_unsupported_upload_type():
    upload = UploadedBytes(b"plain text", "resume.txt", "text/plain")
    with pytest.raises(ResumeValidationError, match="PDF or DOCX"):
        validate_resume_upload(upload)


def test_rejects_file_renamed_to_docx():
    upload = UploadedBytes(
        b"not a real document",
        "resume.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    with pytest.raises(ResumeValidationError, match="not a real DOCX"):
        validate_resume_docx_upload(upload)


def test_rejects_non_pdf_extension():
    upload = UploadedBytes(make_text_pdf(VALID_RESUME), "resume.jpg", "image/jpeg")
    with pytest.raises(ResumeValidationError, match="Only PDF files"):
        validate_resume_pdf_upload(upload)


def test_rejects_wrong_mime_even_with_pdf_extension():
    upload = UploadedBytes(make_text_pdf(VALID_RESUME), "resume.pdf", "image/png")
    with pytest.raises(ResumeValidationError, match="not PDF"):
        validate_resume_pdf_upload(upload)


def test_rejects_file_renamed_to_pdf():
    upload = UploadedBytes(b"not a real pdf", "resume.pdf", "application/pdf")
    with pytest.raises(ResumeValidationError, match="not a real PDF"):
        validate_resume_pdf_upload(upload)


def test_rejects_blank_pdf():
    upload = UploadedBytes(make_blank_pdf(), "blank.pdf", "application/pdf")
    with pytest.raises(ResumeValidationError, match="blank|readable text"):
        validate_resume_pdf_upload(upload)


def test_rejects_certificate_document():
    certificate = """CERTIFICATE OF COMPLETION
This is to certify that Prasanth Veluri completed a Python course.
prasanth@example.com
Education Skills Experience Projects
Issued on 20 January 2026.
"""
    with pytest.raises(ResumeValidationError, match="does not appear to be a resume"):
        validate_resume_text(certificate)


def test_rejects_resume_without_email_or_phone():
    no_contact = VALID_RESUME.replace(
        "prasanth@example.com | +91 98765 43210", "Hyderabad, India"
    )
    with pytest.raises(ResumeValidationError, match="email address or phone"):
        validate_resume_text(no_contact)


def test_rejects_excessively_long_pasted_text():
    with pytest.raises(ResumeValidationError, match="100,000 characters"):
        validate_resume_text(VALID_RESUME + (" experience" * 20_000))


def test_detects_candidate_name_when_present():
    assert detect_candidate_name(VALID_RESUME) == "Prasanth Veluri"
