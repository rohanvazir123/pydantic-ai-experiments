# Copyright 2024 The Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compare standard Docling pipeline vs VLM pipeline for image-heavy PDFs.

Standard pipeline:  text + layout extraction only — figures become placeholders.
VLM pipeline:       each page is rendered as an image and sent to Qwen2.5-VL
                    via Ollama, which converts it to markdown *including* figure
                    descriptions.  The resulting chunks contain real image content.

Prerequisites:
  ollama pull qwen2.5vl:7b          # ~5 GB
  python basics/rag/docling/download_demo_documents.py   # download PDFs

Run:
  python basics/rag/docling/vlm_picture_annotation.py [--pdf rag_paper.pdf] [--pages 3]
"""

import argparse
import json
import re
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "common" / "documents"
OUT_DIR = Path(__file__).parent / "output" / "vlm_annotation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_VLM_URL = "http://localhost:11434/v1/chat/completions"
VLM_MODEL = "qwen2.5vl:7b"


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

def make_standard_converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # Standard pipeline: layout + text extraction only.
    # No picture description (SmolVLM or any other inline VLM) — figures are
    # annotated only when VlmPipeline (Qwen2.5-VL) is explicitly enabled.
    # do_ocr=False → digital PDFs have a text layer.
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = False
    pdf_opts.do_picture_classification = False
    pdf_opts.do_picture_description = False
    pdf_opts.generate_picture_images = False
    pdf_opts.generate_page_images = False

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
        }
    )


def make_vlm_converter(model: str = VLM_MODEL, url: str = OLLAMA_VLM_URL):
    """Standard PDF pipeline with Qwen2.5-VL picture description via Ollama.

    Text is extracted via layout analysis (reliable).  Only cropped figure
    images are sent to Qwen2.5-VL — so a 10-page doc with 2 charts triggers
    exactly 2 VLM calls, never touching the surrounding text.

    Prerequisites:
        ollama pull qwen2.5vl:7b
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        PictureDescriptionApiOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = False                  # digital PDFs have a text layer
    pdf_opts.do_table_structure = True        # parse table rows/cols locally
    pdf_opts.do_picture_classification = False
    pdf_opts.do_picture_description = True   # route figures to Qwen2.5-VL
    pdf_opts.generate_picture_images = True  # crop figure images for the VLM
    pdf_opts.enable_remote_services = True   # required for API-based description

    pdf_opts.picture_description_options = PictureDescriptionApiOptions(
        url=url,  # type: ignore[arg-type]
        params={"model": model},
        prompt="Describe this chart or image in detail.",
        timeout=180.0,
        concurrency=1,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
        }
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_doc(doc) -> list[dict]:
    from docling.chunking import HybridChunker
    chunker = HybridChunker(merge_peers=True)
    chunks = list(chunker.chunk(dl_doc=doc))
    return [
        {
            "index": i,
            "text": chunk.text if hasattr(chunk, "text") else str(chunk),
            "char_count": len(chunk.text if hasattr(chunk, "text") else str(chunk)),
        }
        for i, chunk in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

# Standard pipeline: Docling emits <!-- image --> for figures it can't describe.
# VLM pipeline: figures are described in-line; we prompt for [Figure: ...] tags.
_FIGURE_PLACEHOLDER = re.compile(r"<!--\s*image\s*-->", re.I)
_FIGURE_CAPTION = re.compile(r"^Figure\s+\d+", re.M)   # "Figure 1: ..." captions
# Qwen2.5-VL appends descriptions inline after the figure caption.
# Detect by phrases that appear in natural image descriptions.
_FIGURE_DESCRIPTION = re.compile(
    r"The image (you'?ve provided|shows|depicts|displays|is a)|"
    r"(This|The) (image|figure|chart|graph|diagram|screenshot|flowchart) (shows|depicts|displays|illustrates|presents|appears|is)|"
    r"(bar|line|pie|scatter|stacked) (chart|graph)|"
    r"shows a series of|"
    r"flowchart|screenshot from|"
    r"\[Figure:",
    re.I,
)


def count_figures_placeholder(chunks: list[dict]) -> int:
    """Chunks that contain a bare <!-- image --> (standard pipeline, no description)."""
    return sum(1 for c in chunks if _FIGURE_PLACEHOLDER.search(c["text"]))


def count_figures_with_caption(chunks: list[dict]) -> int:
    """Chunks that contain a figure caption label."""
    return sum(1 for c in chunks if _FIGURE_CAPTION.search(c["text"]))


def count_figures_described(chunks: list[dict]) -> int:
    """Chunks where Qwen2.5-VL produced an inline description."""
    return sum(1 for c in chunks if _FIGURE_DESCRIPTION.search(c["text"]))


def find_figure_chunks(chunks: list[dict]) -> list[dict]:
    """Return chunks that contain any figure-related content."""
    return [
        c for c in chunks
        if (
            _FIGURE_PLACEHOLDER.search(c["text"])
            or _FIGURE_CAPTION.search(c["text"])
            or _FIGURE_DESCRIPTION.search(c["text"])
        )
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(pdf_name: str, max_pages: int) -> None:
    pdf_path = DOCS_DIR / pdf_name
    if not pdf_path.exists():
        print(f"[error] {pdf_path} not found — run download_demo_documents.py first")
        return

    stem = pdf_path.stem
    print(f"\n{'='*70}")
    print(f"Document : {pdf_name}")
    print(f"Max pages: {max_pages}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # Pass 1 — standard pipeline
    # ------------------------------------------------------------------
    print("\n[1/2] Standard Docling pipeline (no VLM) ...")
    std_conv = make_standard_converter()
    std_result = std_conv.convert(str(pdf_path))
    std_doc = std_result.document
    std_chunks = chunk_doc(std_doc)
    std_placeholders = count_figures_placeholder(std_chunks)
    std_captions = count_figures_with_caption(std_chunks)
    print(f"      {len(std_chunks)} chunks | {std_placeholders} <!-- image --> placeholders | {std_captions} figure captions")

    # ------------------------------------------------------------------
    # Pass 2 — VLM pipeline
    # ------------------------------------------------------------------
    print(f"\n[2/2] VLM pipeline ({VLM_MODEL} via Ollama) ...")
    print("      This takes ~10-30 s per page — be patient.")
    try:
        vlm_conv = make_vlm_converter()
        vlm_result = vlm_conv.convert(str(pdf_path))
        vlm_doc = vlm_result.document
        vlm_chunks = chunk_doc(vlm_doc)
        vlm_figures_described = count_figures_described(vlm_chunks)
        vlm_ok = True
        print(f"      {len(vlm_chunks)} chunks, {vlm_figures_described} contain VLM figure descriptions")
    except Exception as exc:
        print(f"      [warn] VLM pipeline failed: {exc}")
        print("      Is Ollama running?  ollama serve && ollama pull qwen2.5vl:7b")
        vlm_chunks = []
        vlm_figures_described = 0
        vlm_ok = False

    # ------------------------------------------------------------------
    # Side-by-side: show figure chunks from both passes
    # ------------------------------------------------------------------
    std_fig_chunks = find_figure_chunks(std_chunks)
    vlm_fig_chunks = find_figure_chunks(vlm_chunks) if vlm_ok else []

    print(f"\n{'─'*70}")
    print("FIGURE CHUNKS COMPARISON")
    print(f"{'─'*70}")

    max_show = 3
    for i in range(max_show):
        print(f"\n--- Figure chunk #{i + 1} ---")

        std_c = std_fig_chunks[i] if i < len(std_fig_chunks) else None
        vlm_c = vlm_fig_chunks[i] if i < len(vlm_fig_chunks) else None

        print("  Standard pipeline:")
        if std_c:
            preview = std_c["text"][:400].replace("\n", " ")
            print(f"    [{std_c['char_count']} chars] {preview}")
        else:
            print("    (no figure chunk)")

        print("  VLM pipeline:")
        if vlm_c:
            preview = vlm_c["text"][:400].replace("\n", " ")
            print(f"    [{vlm_c['char_count']} chars] {preview}")
        elif not vlm_ok:
            print("    (VLM unavailable)")
        else:
            print("    (no figure chunk)")

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    output = {
        "document": pdf_name,
        "standard": {
            "total_chunks": len(std_chunks),
            "figure_placeholder_chunks": std_placeholders,
            "figure_caption_chunks": std_captions,
            "chunks": std_chunks,
            "figure_chunks": std_fig_chunks,
        },
        "vlm": {
            "model": VLM_MODEL,
            "available": vlm_ok,
            "total_chunks": len(vlm_chunks),
            "figure_described_chunks": vlm_figures_described,
            "chunks": vlm_chunks,
            "figure_chunks": vlm_fig_chunks,
        },
    }

    out_path = OUT_DIR / f"{stem}_vlm_comparison.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"\n{'─'*70}")
    print("SUMMARY")
    print(f"{'─'*70}")
    print(f"  Standard: {len(std_chunks)} chunks | {std_placeholders} <!-- image --> (ignored) | {std_captions} captions")
    if vlm_ok:
        print(f"  VLM     : {len(vlm_chunks)} chunks | {vlm_figures_described} figures described")
    else:
        print("  VLM     : unavailable (start Ollama and pull qwen2.5vl:7b)")
    print(f"  Saved   : {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM vs standard Docling chunking demo")
    parser.add_argument(
        "--pdf",
        default="rag_paper.pdf",
        help="PDF filename inside basics/rag/common/documents/ (default: rag_paper.pdf)",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=5,
        help="Max pages to process (default: 5, use -1 for all)",
    )
    args = parser.parse_args()
    run(args.pdf, args.pages)


if __name__ == "__main__":
    main()
