"""
Download documents that demonstrate Docling chunking failures and VLM annotation.

Text-layout failure demos:
  1. attention_is_all_you_need.pdf  — two-column academic PDF (column mixing)
  2. bert_paper.pdf                 — two-column paper (column mixing)
  3. nist_sp800_53.pdf              — long government doc (header/footer contamination)

Image-heavy PDFs for VLM annotation (figure descriptions ignored without VLM):
  4. rag_paper.pdf                  — original RAG paper with architecture diagrams
  5. clip_paper.pdf                 — OpenAI CLIP paper with benchmark figures

Financial reports with charts/graphs for VLM annotation:
  6. tesla_q4_2023.pdf              — Tesla Q4 2023 slide-deck with production/revenue charts
  7. bis_annual_report_2024.pdf          — BIS Annual Report 2024 with macro charts

PDFs are written to two locations:
  - basics/rag/common/documents/   (for the chunking demo scripts)
  - rag/documents/                 (for the RAG ingestion pipeline)

Run:
  python basics/rag/docling/download_demo_documents.py
"""

import sys
import urllib.request
from pathlib import Path

# basics demo location
DEST = Path(__file__).parent.parent / "common" / "documents"
DEST.mkdir(parents=True, exist_ok=True)

# RAG pipeline documents folder (image-heavy + financial only — not the failure demos)
RAG_DOCS = Path(__file__).parent.parent.parent.parent / "rag" / "documents"
RAG_DOCS.mkdir(parents=True, exist_ok=True)

DOCUMENTS = [
    # --- text-layout failure demos (basics only) ---
    {
        "name": "attention_is_all_you_need.pdf",
        "url": "https://arxiv.org/pdf/1706.03762",
        "failure": "two-column column mixing",
        "notes": "NeurIPS 2017, classic two-column layout — column boundaries cause chunk mixing",
        "rag_docs": False,
    },
    {
        "name": "bert_paper.pdf",
        "url": "https://arxiv.org/pdf/1810.04805",
        "failure": "two-column column mixing",
        "notes": "NAACL 2019, two-column — second example of column interleaving",
        "rag_docs": False,
    },
    {
        "name": "nist_sp800_53.pdf",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf",
        "failure": "header/footer contamination",
        "notes": "NIST SP 800-53 Rev 5 — long government doc with running headers/footers and complex tables",
        "rag_docs": False,
    },
    # --- image-heavy PDFs for VLM annotation demo (both locations) ---
    {
        "name": "rag_paper.pdf",
        "url": "https://arxiv.org/pdf/2005.11401",
        "failure": "figure/image ignored without VLM",
        "notes": "Original RAG paper — architecture diagrams and performance charts; "
                 "standard Docling emits <!-- image --> placeholders; VLM describes them",
        "rag_docs": True,
    },
    {
        "name": "clip_paper.pdf",
        "url": "https://arxiv.org/pdf/2103.00020",
        "failure": "figure/image ignored without VLM",
        "notes": "OpenAI CLIP paper — many figures showing zero-shot transfer benchmark results",
        "rag_docs": True,
    },
    # --- financial reports — charts, tables, bar/line graphs (both locations) ---
    {
        "name": "tesla_q4_2023.pdf",
        "url": "https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q4-2023-Update.pdf",
        "failure": "figure/image ignored without VLM",
        "notes": "Tesla Q4 2023 earnings update — slide-deck with revenue charts, "
                 "production graphs, and vehicle delivery figures",
        "rag_docs": True,
    },
    {
        "name": "bis_annual_report_2024.pdf",
        "url": "https://www.bis.org/publ/arpdf/ar2024e.pdf",
        "failure": "figure/image ignored without VLM",
        "notes": "BIS Annual Economic Report 2024 — GDP, inflation, interest rate charts "
                 "and macro economic figures across multiple pages",
        "rag_docs": True,
    },
]


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return True
    print(f"  [download] {dest.name} from {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        dest.write_bytes(data)
        size_kb = len(data) // 1024
        print(f"  [ok] {dest.name} ({size_kb} KB)")
        return True
    except Exception as exc:
        print(f"  [fail] {dest.name}: {exc}", file=sys.stderr)
        return False


def main() -> None:
    print(f"Downloading demo documents to: {DEST}")
    print(f"Image-heavy PDFs also copied to: {RAG_DOCS}\n")
    ok = 0
    for doc in DOCUMENTS:
        print(f"• {doc['name']}  [{doc['failure']}]")
        print(f"  {doc['notes']}")
        dest_path = DEST / doc["name"]
        if download(doc["url"], dest_path):
            ok += 1
            if doc.get("rag_docs"):
                rag_path = RAG_DOCS / doc["name"]
                if not rag_path.exists():
                    import shutil
                    shutil.copy2(dest_path, rag_path)
                    print(f"  [copy] → {rag_path}")
                else:
                    print(f"  [skip copy] {rag_path.name} already in rag/documents/")
        print()

    print(f"Done: {ok}/{len(DOCUMENTS)} documents ready in {DEST}")

    # Also note the CUAD legal contract already in the project
    cuad_path = Path(__file__).parent.parent.parent / "rag" / "documents" / "legal"
    if cuad_path.exists():
        files = list(cuad_path.glob("*.md")) + list(cuad_path.glob("*.pdf"))
        if files:
            print(f"\nFound {len(files)} CUAD legal contracts at {cuad_path}")
            print("  These demonstrate: hierarchy loss in nested legal clauses")
            print(f"  Example: {files[0].name}")


if __name__ == "__main__":
    main()
