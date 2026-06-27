"""RAG document loading utilities.

Each loader converts low-level parser failures (corrupt files, encryption,
encoding problems) into a ``ValueError`` with an actionable message so callers
can surface a clean 4xx instead of leaking a parser stack trace.
"""

from pathlib import Path

import docx2txt  # type: ignore[import-untyped]
from langchain_core.documents import Document
from loguru import logger
from pypdf import PdfReader
from pypdf.errors import PdfReadError


def _load_pdf(path: Path) -> list[Document]:
    """Load all pages from a PDF file as individual Documents.

    Handles encrypted PDFs (attempts an empty-password decrypt) and converts
    malformed-PDF errors into a ValueError.
    """
    try:
        reader = PdfReader(str(path))
        # Some PDFs are encrypted with an empty owner password; try to unlock.
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001 - normalise to ValueError below
                msg = "PDF is password-protected and cannot be read."
                raise ValueError(msg) from exc
    except PdfReadError as exc:
        msg = "PDF file is corrupt or not a valid PDF."
        raise ValueError(msg) from exc

    documents = []
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - skip unreadable pages, keep going
            logger.warning("Skipping unreadable PDF page {}: {}", page_num, exc)
            text = ""
        documents.append(
            Document(
                page_content=text,
                metadata={"source": str(path), "page": page_num},
            )
        )
    return documents


def _load_docx(path: Path) -> list[Document]:
    """Load the full text from a DOCX file as a single Document."""
    try:
        text = docx2txt.process(str(path)) or ""
    except Exception as exc:  # noqa: BLE001 - normalise parser errors
        msg = "DOCX file is corrupt or not a valid Word document."
        raise ValueError(msg) from exc
    return [Document(page_content=text, metadata={"source": str(path)})]


def _load_txt(path: Path) -> list[Document]:
    """Load the full text from a plain text file as a single Document.

    Falls back from UTF-8 to a lenient decode so files with stray non-UTF-8
    bytes still load instead of raising UnicodeDecodeError.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("TXT file {} is not valid UTF-8; decoding leniently", path.name)
        text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": str(path)})]


def load_documents(path: Path) -> list[Document]:
    """Load supported documents from a file path.

    Raises ValueError for unsupported types, unreadable files, or documents that
    contain no extractable text.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        documents = _load_pdf(path)
    elif suffix == ".docx":
        documents = _load_docx(path)
    elif suffix == ".txt":
        documents = _load_txt(path)
    else:
        msg = f"Unsupported file type: {suffix}"
        raise ValueError(msg)

    if not documents:
        msg = "Document contains no readable text"
        raise ValueError(msg)
    combined_text = "\n".join(doc.page_content.strip() for doc in documents)
    if not combined_text.strip():
        msg = "Document contains only whitespace"
        raise ValueError(msg)
    return documents
