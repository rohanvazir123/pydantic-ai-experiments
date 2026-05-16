"""
Download documents that demonstrate Docling chunking failures.

Documents downloaded:
  1. attention_is_all_you_need.pdf  — two-column academic PDF (column mixing)
  2. bert_paper.pdf                 — another two-column paper (column mixing)
  3. apache_age_manual.pdf          — long technical manual (header/footer contamination)

Run:
  python basics/docling_lightrag_raganything/download_demo_documents.py
"""

import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).parent / "documents"
DEST.mkdir(exist_ok=True)

DOCUMENTS = [
    {
        "name": "attention_is_all_you_need.pdf",
        "url": "https://arxiv.org/pdf/1706.03762",
        "failure": "two-column column mixing",
        "notes": "NeurIPS 2017, classic two-column layout — column boundaries cause chunk mixing",
    },
    {
        "name": "bert_paper.pdf",
        "url": "https://arxiv.org/pdf/1810.04805",
        "failure": "two-column column mixing",
        "notes": "NAACL 2019, two-column — second example of column interleaving",
    },
    {
        "name": "nist_sp800_53.pdf",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf",
        "failure": "header/footer contamination",
        "notes": "NIST SP 800-53 Rev 5 — long government doc with running headers/footers and complex tables",
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
    print(f"Downloading demo documents to: {DEST}\n")
    ok = 0
    for doc in DOCUMENTS:
        print(f"• {doc['name']}  [{doc['failure']}]")
        print(f"  {doc['notes']}")
        if download(doc["url"], DEST / doc["name"]):
            ok += 1
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
