"""
Produce improved chunks by applying targeted fixes for each Docling failure mode.

Fixes applied:
  1. Two-column mixing     → TableFormerMode.ACCURATE + do_cell_matching=False
  2. Header/footer noise   → Strip PAGE_HEADER / PAGE_FOOTER elements before chunking
  3. Table splitting        → ACCURATE mode + do_cell_matching=False + repeat_table_header

Chunks are saved as JSON to:
  basics/docling_lightrag_raganything/output/docling/good_chunks/

Run:
  python basics/docling_lightrag_raganything/produce_good_chunks.py
"""

import json
from pathlib import Path

from docling.datamodel.base_models import DocItemLabel
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker

DOCS_DIR = Path(__file__).parent / "documents"
OUT_DIR = Path(__file__).parent / "output" / "docling" / "good_chunks"
BAD_DIR = Path(__file__).parent / "output" / "docling" / "bad_chunks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Fix 1: Accurate pipeline — better layout analysis, table structure
# -------------------------------------------------------------------
def make_accurate_converter() -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = False
    return DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=opts)}
    )


# -------------------------------------------------------------------
# Fix 2: Strip page headers and footers from the DoclingDocument tree
# -------------------------------------------------------------------
NOISE_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}

def strip_headers_footers(doc) -> int:
    """Remove PAGE_HEADER and PAGE_FOOTER items from the document body. Returns count removed."""
    to_remove = []
    for item, _ in doc.iterate_items():
        if hasattr(item, "label") and item.label in NOISE_LABELS:
            to_remove.append(item)

    for item in to_remove:
        try:
            doc.body.children = [
                ref for ref in doc.body.children if ref.cref != item.self_ref
            ]
        except Exception:
            pass
    return len(to_remove)


# -------------------------------------------------------------------
# Fix 3: HybridChunker configured to keep table headers in context
# -------------------------------------------------------------------
def make_chunker(repeat_table_header: bool = True) -> HybridChunker:
    return HybridChunker(
        repeat_table_header=repeat_table_header,
        merge_peers=True,
    )


# -------------------------------------------------------------------
# Serialise chunks to dicts
# -------------------------------------------------------------------
def serialise(chunks: list) -> list[dict]:
    result = []
    for i, chunk in enumerate(chunks):
        text = chunk.text if hasattr(chunk, "text") else str(chunk)
        meta = chunk.meta.export_json_dict() if hasattr(chunk, "meta") and chunk.meta else {}
        result.append({"index": i, "text": text, "char_count": len(text), "meta": meta})
    return result


# -------------------------------------------------------------------
# Load bad-chunk counts for comparison
# -------------------------------------------------------------------
def load_bad_summary() -> dict:
    summary_path = BAD_DIR / "summary.json"
    if not summary_path.exists():
        return {}
    data = json.loads(summary_path.read_text())
    return {d["document"]: d for d in data}


# -------------------------------------------------------------------
# Per-document processing with the right fix
# -------------------------------------------------------------------
DOCUMENTS = [
    {
        "file": "attention_is_all_you_need.pdf",
        "failure": "two_column_mixing",
        "fix": "ACCURATE pipeline + do_cell_matching=False",
    },
    {
        "file": "bert_paper.pdf",
        "failure": "two_column_mixing",
        "fix": "ACCURATE pipeline + do_cell_matching=False",
    },
    {
        "file": "nist_sp800_53.pdf",
        "failure": "header_footer_contamination",
        "fix": "Strip PAGE_HEADER/PAGE_FOOTER elements before chunking",
    },
    {
        "file": "q4_financial_report.pdf",
        "failure": "table_splitting",
        "fix": "ACCURATE pipeline + repeat_table_header=True",
    },
]


def process(doc_meta: dict, converter: DocumentConverter, bad_summary: dict) -> dict | None:
    path = DOCS_DIR / doc_meta["file"]
    if not path.exists():
        print(f"  [skip] {doc_meta['file']} not found")
        return None

    print(f"  Converting ...")
    result = converter.convert(str(path))
    doc = result.document

    removed = 0
    if doc_meta["failure"] == "header_footer_contamination":
        removed = strip_headers_footers(doc)
        print(f"  Removed {removed} header/footer elements")

    chunker = make_chunker(repeat_table_header=(doc_meta["failure"] == "table_splitting"))
    chunks = list(chunker.chunk(dl_doc=doc))
    print(f"  Produced {len(chunks)} chunks (fix: {doc_meta['fix']})")

    bad = bad_summary.get(doc_meta["file"], {})
    comparison = {
        "bad_total_chunks": bad.get("total_chunks", "n/a"),
        "bad_flagged_anomalies": bad.get("flagged_anomalies", "n/a"),
        "good_total_chunks": len(chunks),
        "headers_footers_removed": removed,
    }

    return {
        "document": doc_meta["file"],
        "failure_type": doc_meta["failure"],
        "fix_applied": doc_meta["fix"],
        "comparison": comparison,
        "chunks": serialise(chunks),
    }


def main() -> None:
    print(f"Output directory: {OUT_DIR}\n")
    converter = make_accurate_converter()
    bad_summary = load_bad_summary()
    summary = []

    for doc_meta in DOCUMENTS:
        stem = Path(doc_meta["file"]).stem
        print(f"\n{'='*60}")
        print(f"Document: {doc_meta['file']}")
        print(f"Failure:  {doc_meta['failure']}")
        print(f"Fix:      {doc_meta['fix']}")
        print(f"{'='*60}")

        output = process(doc_meta, converter, bad_summary)
        if output is None:
            continue

        out_path = OUT_DIR / f"{stem}_good_chunks.json"
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"  Saved → {out_path.name}")

        summary.append(
            {
                "document": doc_meta["file"],
                "failure_type": doc_meta["failure"],
                "fix_applied": doc_meta["fix"],
                **output["comparison"],
                "output_file": out_path.name,
            }
        )

    # Write summary
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print("COMPARISON: bad vs good chunks")
    print(f"{'='*60}")
    for s in summary:
        bad_c = s.get("bad_total_chunks", "?")
        good_c = s.get("good_total_chunks", "?")
        bad_f = s.get("bad_flagged_anomalies", "?")
        removed = s.get("headers_footers_removed", 0)
        print(f"  {s['document']}")
        print(f"    chunks: {bad_c} → {good_c}  |  flagged: {bad_f} → (see output)  |  noise removed: {removed}")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
