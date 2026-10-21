"""Read supported formats into ordered text with source locations.

Layout is used for extraction and citation only. No heading, font or page boundary
is supplied to the semantic graph planner as a grouping instruction.
"""

import html
import unicodedata
from pathlib import Path

from legal_rag.schemas import ParsedDocument, Span
from legal_rag.settings import Settings

SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".txt"}


def parse_document(path: Path, settings: Settings) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported format: {suffix}")
    if suffix in {".txt", ".md"}:
        text = unicodedata.normalize("NFC", path.read_text(encoding="utf-8-sig"))
        if not text.strip():
            raise ValueError("The document has no readable text.")
        if len(text) > settings.max_characters:
            raise ValueError("The document exceeds the configured character limit.")
        return ParsedDocument(
            text=text,
            spans=[Span(start=0, end=len(text), locator="text")],
            parser="utf-8",
        )
    if settings.parser == "plain":
        if suffix != ".pdf":
            raise ValueError(
                "The plain extraction backend supports PDF, TXT and Markdown only."
            )
        return parse_plain_pdf(path, settings)
    return parse_docling(path, settings)


def parse_plain_pdf(path: Path, settings: Settings) -> ParsedDocument:
    """Explicit text-only backend for diagnostics, without layout analysis or OCR."""
    import pymupdf

    pieces, spans, warnings = [], [], []
    position = 0
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError("Password-protected PDFs must be unlocked before upload.")
        if len(pdf) > settings.max_pages:
            raise ValueError("The PDF exceeds the configured page limit.")
        for number in range(1, len(pdf) + 1):
            page = pdf[number - 1]
            text = unicodedata.normalize("NFC", str(page.get_text("text")))
            if not text.strip():
                warnings.append(f"Page {number} has no readable text; OCR is required.")
                continue
            pieces.append(text)
            spans.append(
                Span(
                    start=position,
                    end=position + len(text),
                    page=number,
                    locator=f"page:{number}",
                )
            )
            position += len(text) + 1
        pages = len(pdf)
    return assemble(pieces, spans, pages, warnings, "pymupdf-text", settings)


def parse_docling(path: Path, settings: Settings) -> ParsedDocument:
    """Delegate format-specific conversion and OCR to Docling."""
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc.items.table.table import TableItem

    converter = DocumentConverter()
    result = converter.convert(path, max_num_pages=settings.max_pages)
    status = str(getattr(result.status, "value", result.status)).lower()
    if "success" not in status:
        raise ValueError(f"Docling did not complete conversion: {status}")
    document = result.document
    pieces, spans, warnings = [], [], []
    position = 0
    for item, _level in document.iterate_items():
        label = str(getattr(item, "label", ""))
        if "page_header" in label or "page_footer" in label:
            continue
        if isinstance(item, TableItem):
            text = item.export_to_markdown(doc=document)
        else:
            text = getattr(item, "text", "")
        text = unicodedata.normalize("NFC", text or "").strip()
        if not text:
            continue
        pieces.append(text)
        locations = list(getattr(item, "prov", []))
        if not locations:
            spans.append(
                Span(start=position, end=position + len(text), locator=item.self_ref)
            )
        for location in locations:
            spans.append(
                Span(
                    start=position,
                    end=position + len(text),
                    page=location.page_no,
                    locator=item.self_ref,
                )
            )
        position += len(text) + 1
    pages = len(getattr(document, "pages", {}))
    read_pages = {span.page for span in spans if span.page}
    for number in range(1, pages + 1):
        if number not in read_pages:
            warnings.append(
                f"No text was extracted from page {number}; inspect the source."
            )
    confidence = getattr(result, "confidence", None)
    grade = str(getattr(confidence, "low_grade", "")).lower()
    if "poor" in grade or "fair" in grade:
        warnings.append(f"Docling reported low conversion confidence: {grade}.")
    for error in getattr(result, "errors", []):
        warnings.append(str(error))
    return assemble(pieces, spans, pages, warnings, "docling", settings)


def assemble(
    pieces: list[str],
    spans: list[Span],
    pages: int,
    warnings: list[str],
    parser: str,
    settings: Settings,
) -> ParsedDocument:
    text = "\n".join(pieces)
    if not text.strip():
        raise ValueError(
            "No readable text was extracted. Check OCR language and source quality."
        )
    if len(text) > settings.max_characters:
        raise ValueError("The document exceeds the configured character limit.")
    if "\ufffd" in text or "\x00" in text:
        warnings.append(
            "Extracted text contains replacement or null characters; review is required."
        )
    return ParsedDocument(
        text=text, spans=spans, pages=pages, warnings=warnings, parser=parser
    )


def preview_html(parsed: ParsedDocument, name: str) -> str:
    """Render inert text with source-position anchors for non-PDF citations."""
    positions = sorted({0, len(parsed.text), *(span.start for span in parsed.spans)})
    parts = []
    for start, end in zip(positions, positions[1:]):
        parts.append(
            f'<span id="source-{start}">{html.escape(parsed.text[start:end])}</span>'
        )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<style>body{max-width:900px;margin:32px auto;padding:0 20px;font:15px/1.7 system-ui}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}span:target{background:#dbeafe}</style>"
        f"<title>{html.escape(name)}</title><h1>{html.escape(name)}</h1>"
        f"<pre>{''.join(parts)}</pre></html>"
    )
