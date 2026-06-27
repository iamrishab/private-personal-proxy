"""Text chunking for RAG ingestion.

Uses a recursive splitter with an explicit separator hierarchy so splits prefer
natural boundaries (paragraphs, then lines, then sentences) before falling back
to mid-word cuts. Tiny fragments left over from splitting are dropped so the
vector store is not polluted with low-signal chunks.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Separator hierarchy, most-structural first. The splitter tries each in order,
# so paragraph and line breaks are preferred over sentence and word boundaries.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

# Chunks shorter than this (after stripping) carry too little context to be
# useful for retrieval and are discarded.
_MIN_CHUNK_CHARS = 40


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 77,
) -> list[Document]:
    """Split documents into overlapping, boundary-aware chunks for embedding.

    Parent-document metadata (source, page, etc.) is preserved on every chunk by
    the splitter. Whitespace-only and very short trailing fragments are removed.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=_SEPARATORS,
        keep_separator=True,
    )
    chunks = splitter.split_documents(documents)
    # Drop empty or trivially short fragments that add noise but no signal.
    return [chunk for chunk in chunks if len(chunk.page_content.strip()) >= _MIN_CHUNK_CHARS]
