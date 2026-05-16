"""
Produce bad chunks using default Docling settings.

Demonstrates known failure modes:
  1. attention_is_all_you_need.pdf  — two-column column mixing
  2. bert_paper.pdf                 — two-column column mixing (second example)
  3. nist_sp800_53.pdf              — header/footer contamination
  4. q4_financial_report.pdf        — table splitting across pages

Chunks are saved as JSON to:
  basics/docling_lightrag_raganything/output/docling/bad_chunks/

Run:
  python basics/docling_lightrag_raganything/produce_bad_chunks.py
"""

import json
import sys
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

DOCS_DIR = Path(__file__).parent / "documents"
OUT_DIR = Path(__file__).parent / "output" / "docling" / "bad_chunks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Default converter — no accuracy tuning, no fixes
converter = DocumentConverter()

# Default chunker — no token ceiling, merges peers naively
chunker = HybridChunker()

DOCUMENTS = [
    {
        "file": "attention_is_all_you_need.pdf",
        "failure": "two_column_mixing",
        "description": (
            "Two-column NeurIPS paper. Chunks near column boundaries mix sentences "
            "from the left and right columns, producing incoherent text."
        ),
    },
    {
        "file": "bert_paper.pdf",
        "failure": "two_column_mixing",
        "description": (
            "Two-column NAACL paper. Same column-mixing failure as above — "
            "confirms this is systematic, not document-specific."
        ),
    },
    {
        "file": "nist_sp800_53.pdf",
        "failure": "header_footer_contamination",
        "description": (
            "Long NIST government publication. Running page headers and footers "
            "(e.g. 'NIST SP 800-53 REV. 5', page numbers) are injected into content chunks."
        ),
    },
    {
        "file": "q4_financial_report.pdf",
        "failure": "table_splitting",
        "description": (
            "Q4 business review with financial tables. Tables that span multiple "
            "pages are split into disconnected fragments, losing row context."
        ),
    },
]


def chunk_document(doc_meta: dict) -> list[dict]:
    path = DOCS_DIR / doc_meta["file"]
    if not path.exists():
        print(f"  [skip] {doc_meta['file']} not found — run download_demo_documents.py first")
        return []

    print(f"  Converting {doc_meta['file']} ...")
    result = converter.convert(str(path))
    doc = result.document

    print(f"  Chunking ...")
    chunks = list(chunker.chunk(dl_doc=doc))
    print(f"  Produced {len(chunks)} chunks")

    serialised = []
    for i, chunk in enumerate(chunks):
        text = chunk.text if hasattr(chunk, "text") else str(chunk)
        meta = chunk.meta.export_json_dict() if hasattr(chunk, "meta") and chunk.meta else {}
        serialised.append(
            {
                "index": i,
                "text": text,
                "char_count": len(text),
                "meta": meta,
            }
        )
    return serialised


def detect_anomalies(chunks: list[dict], failure_type: str) -> list[dict]:
    """Tag chunks that exhibit the expected failure pattern."""
    flagged = []
    for chunk in chunks:
        text = chunk["text"]
        flags = []

        if failure_type == "two_column_mixing":
            # Heuristic: abrupt sentence breaks, very short sentences mid-paragraph
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            short_lines = [l for l in lines if 5 < len(l) < 50 and not l.endswith((".", ",", ":", ";", ")"))]
            if len(short_lines) >= 2:
                flags.append("possible_column_interleave: multiple short disconnected lines")

        elif failure_type == "header_footer_contamination":
            # Heuristic: chunk contains very short standalone lines that look like headers
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            noise_lines = [l for l in lines if len(l) < 60 and l.isupper()]
            if noise_lines:
                flags.append(f"possible_header_footer: {noise_lines[:2]}")

        elif failure_type == "table_splitting":
            # Heuristic: chunk looks like a table fragment (has | separators or starts mid-row)
            if "|" in text and chunk["char_count"] < 300:
                flags.append("possible_table_fragment: short chunk with pipe separators")

        if flags:
            flagged.append({**chunk, "anomaly_flags": flags})

    return flagged


def main() -> None:
    print(f"Output directory: {OUT_DIR}\n")
    summary = []

    for doc_meta in DOCUMENTS:
        stem = Path(doc_meta["file"]).stem
        print(f"\n{'='*60}")
        print(f"Document: {doc_meta['file']}")
        print(f"Failure:  {doc_meta['failure']}")
        print(f"{'='*60}")

        chunks = chunk_document(doc_meta)
        if not chunks:
            continue

        flagged = detect_anomalies(chunks, doc_meta["failure"])

        output = {
            "document": doc_meta["file"],
            "failure_type": doc_meta["failure"],
            "description": doc_meta["description"],
            "total_chunks": len(chunks),
            "flagged_chunks": len(flagged),
            "all_chunks": chunks,
            "flagged_anomalies": flagged,
        }

        out_path = OUT_DIR / f"{stem}_bad_chunks.json"
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"  Saved {len(chunks)} chunks → {out_path.name}")
        print(f"  Flagged {len(flagged)} anomalous chunks")

        summary.append(
            {
                "document": doc_meta["file"],
                "failure_type": doc_meta["failure"],
                "total_chunks": len(chunks),
                "flagged_anomalies": len(flagged),
                "output_file": out_path.name,
            }
        )

    # Write summary
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for s in summary:
        print(
            f"  {s['document']:<40} "
            f"chunks={s['total_chunks']:>4}  "
            f"flagged={s['flagged_anomalies']:>3}  "
            f"({s['failure_type']})"
        )
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
