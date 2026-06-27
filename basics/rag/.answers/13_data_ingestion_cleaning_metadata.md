# Data Ingestion — Cleaning, Metadata Enrichment, and Image Conversion

The ingestion pipeline determines the ceiling quality of everything downstream. A RAG system built on dirty, poorly-structured, metadata-poor documents will never be accurate regardless of how sophisticated the retrieval and generation layers are. This file covers the three steps most teams rush past: cleaning, metadata enrichment, and image-to-text conversion.

## Table of Contents

- [Why This Is Skipped and Why That Hurts](#why-this-is-skipped-and-why-that-hurts)
- [Data Cleaning](#data-cleaning)
  - [Boilerplate Removal](#boilerplate-removal)
  - [Encoding and Character Issues](#encoding-and-character-issues)
  - [Deduplication](#deduplication)
  - [OCR Noise and Garbled Text](#ocr-noise-and-garbled-text)
  - [Structure Corruption](#structure-corruption)
- [Metadata Enrichment](#metadata-enrichment)
  - [Document-Level Metadata](#document-level-metadata)
  - [Section and Chapter Metadata](#section-and-chapter-metadata)
  - [Paragraph-Level Metadata](#paragraph-level-metadata)
  - [Table and Column Metadata](#table-and-column-metadata)
  - [Fine-Grained Citation Schema](#fine-grained-citation-schema)
  - [Automated Metadata Extraction Pipeline](#automated-metadata-extraction-pipeline)
- [Image-to-Text Conversion](#image-to-text-conversion)
  - [When Images Contain Critical Information](#when-images-contain-critical-information)
  - [Conversion Strategies by Image Type](#conversion-strategies-by-image-type)
  - [Full Image Processing Pipeline](#full-image-processing-pipeline)
  - [Storage and Indexing of Image-Derived Text](#storage-and-indexing-of-image-derived-text)
- [Full Ingestion Pipeline with All Three Stages](#full-ingestion-pipeline-with-all-three-stages)

---

## Why This Is Skipped and Why That Hurts

Most teams run: `load document → split into chunks → embed → index`. This is the bare minimum, and it produces:

- **Chunks that start with page numbers and headers:** "Page 34 | Employee Handbook | Section 4" is now part of the chunk's embedding, dragging it toward queries about page numbers rather than the actual content.
- **Chunks with no provenance:** A retrieved chunk says "The approval threshold is $5,000" — but from which document, which version, which section? The user cannot verify it.
- **Tables as garbled text:** Column headers and cell values are merged into unreadable strings.
- **Charts and figures indexed as captions only**, with the quantitative content lost entirely.
- **Near-duplicate chunks** from documents that were uploaded twice or from boilerplate sections that appear identically across many files, consuming retrieval slots with redundant content.

Every one of these failures is fixed in the ingestion pipeline, not in the retrieval or generation layers.

---

## Data Cleaning

### Boilerplate Removal

Boilerplate is repeated, non-informative text that appears across many documents or repeatedly within a single document. Including it in chunks wastes embedding space and causes retrieval confusion.

**Common boilerplate patterns:**

```
Page headers/footers:
  "CONFIDENTIAL — INTERNAL USE ONLY"
  "Page 3 of 47"
  "© 2024 Acme Corp. All rights reserved."
  "Employee Handbook | Version 2.4"

Navigation elements (HTML/web content):
  "Home | Products | About | Contact | Login"
  "Skip to main content"
  "Accept all cookies | Manage preferences"

Legal boilerplate:
  "This document does not constitute legal advice..."
  "Forward-looking statements involve risks and uncertainties..."
  (These often appear identically across all investor documents)

Watermarks and stamps (in PDFs):
  "DRAFT", "CONFIDENTIAL", "FOR REVIEW ONLY"
  "Superseded by version 3.1"
```

**Detection and removal:**

```python
class BoilerplateDetector:

    def __init__(self, corpus_sample: list[str]):
        # Build a frequency map: text that appears in > N% of documents is boilerplate
        self.boilerplate_phrases = self._extract_frequent_phrases(corpus_sample, threshold=0.4)
        self.pattern_rules = [
            r"^page\s+\d+\s+of\s+\d+$",            # "page 3 of 47"
            r"©\s*\d{4}",                            # copyright notices
            r"confidential\s*[-–]\s*internal",       # confidentiality headers
            r"^\s*\d+\s*$",                          # standalone page numbers
            r"forward.looking\s+statements",         # legal boilerplate
        ]

    def clean(self, text: str) -> str:
        lines = text.split("\n")
        clean_lines = []
        for line in lines:
            stripped = line.strip().lower()
            if self._is_boilerplate(stripped):
                continue
            clean_lines.append(line)
        return "\n".join(clean_lines)

    def _is_boilerplate(self, line: str) -> bool:
        # Pattern match
        for pattern in self.pattern_rules:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        # Corpus frequency match
        if line in self.boilerplate_phrases:
            return True
        return False
```

**For HTML content — use a content extractor first:**
```python
import trafilatura

def extract_main_content(html: str) -> str:
    """Extract article content, stripping navigation, ads, footers."""
    return trafilatura.extract(html, include_comments=False, include_tables=True)
```

---

### Encoding and Character Issues

PDF text extraction frequently produces encoding artifacts — characters that were rendered correctly in the PDF but come out as garbage in the extracted text.

**Common issues:**

```
Mojibake (wrong encoding applied):
  "café" rendered as "cafÃ©" (UTF-8 decoded as Latin-1)
  
Ligature explosion:
  "fi", "fl", "ffi" ligatures in fonts → extracted as "ﬁ", "ﬂ", "ﬃ" (Unicode ligature characters)
  These do not match "fi" in a keyword search or embedding

Smart quote / dash corruption:
  " " " (curly quotes) → â€œ and â€ (encoding failure)
  — (em dash) → â€" (encoding failure)
  
Hyphenation artifacts:
  "infor-\nmation" (word hyphenated at line break) → should be "information"
  But extracted as "infor-\nmation" literally

Zero-width and control characters:
  Invisible characters embedded in PDF text layers that break tokenization
```

**Cleaning pipeline:**

```python
import unicodedata
import ftfy  # fixes text encoding problems

def clean_encoding(text: str) -> str:
    # Fix encoding errors (mojibake, etc.)
    text = ftfy.fix_text(text)

    # Normalize Unicode (NFC: canonical decomposition, canonical composition)
    text = unicodedata.normalize("NFC", text)

    # Expand ligatures
    LIGATURES = {
        "ﬁ": "fi",  "ﬂ": "fl",  "ﬃ": "ffi",
        "ﬄ": "ffl", "ﬀ": "ff",  "ﬅ": "st",
    }
    for lig, expanded in LIGATURES.items():
        text = text.replace(lig, expanded)

    # Fix hyphenated line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Remove zero-width and control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​‌‍﻿]", "", text)

    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)       # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)    # max 2 consecutive newlines

    return text.strip()
```

---

### Deduplication

**Document-level deduplication:**
Before indexing, check whether the document content already exists in the index. Use a content hash (SHA-256 of the cleaned text). If the hash matches an existing document, skip re-ingestion.

```python
def should_ingest(document_id: str, content_hash: str, registry: DocumentRegistry) -> bool:
    existing = registry.get(document_id)
    if existing and existing.content_hash == content_hash:
        return False  # identical content already indexed
    return True
```

**Chunk-level deduplication:**
Some boilerplate sections appear identically in many documents (standard terms and conditions, legal disclaimers, company mission statements). These sections generate near-identical chunks that compete for retrieval slots.

```python
def deduplicate_chunks(chunks: list[Chunk], similarity_threshold: float = 0.98) -> list[Chunk]:
    """Remove near-duplicate chunks across the corpus using MinHash LSH."""
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=similarity_threshold, num_perm=128)
    unique_chunks = []

    for chunk in chunks:
        mh = MinHash(num_perm=128)
        for word in chunk.content.lower().split():
            mh.update(word.encode("utf-8"))

        if not lsh.query(mh):  # no near-duplicate found
            lsh.insert(chunk.id, mh)
            unique_chunks.append(chunk)

    return unique_chunks
```

MinHash LSH runs in O(n) time and handles millions of chunks efficiently. Deduplicate at ingestion time, not at query time.

---

### OCR Noise and Garbled Text

OCR on low-quality scans produces characteristic errors:

```
Character substitution:
  "1" vs "l" vs "I" — "the number ls" instead of "the number is"
  "0" vs "O" — "l00%" instead of "100%"
  "rn" recognised as "m" — "modem" instead of "modern"

Word splitting:
  "in f ormation" (space inserted mid-word)
  "infor mation" (hyphen + space from column layout)

Merged words:
  "theapproval" (space dropped between words)

Number/unit merging:
  "$ 5,000" → "$5,000" (OCR often splits currency symbols)
  "30 days" → "30days"
```

**OCR quality assessment before indexing:**

```python
def ocr_quality_score(text: str) -> float:
    """Returns 0.0 (garbage) to 1.0 (clean). Routes low-quality text for human review."""
    words = text.split()
    if not words:
        return 0.0

    # Check what fraction of words are in a dictionary
    from spellchecker import SpellChecker
    spell = SpellChecker()
    known = spell.known(words)
    word_quality = len(known) / len(words)

    # Check average word length (OCR noise produces very short "words")
    avg_len = sum(len(w) for w in words) / len(words)
    length_quality = min(avg_len / 5.0, 1.0)  # target avg 5+ chars per word

    return 0.6 * word_quality + 0.4 * length_quality

# Route:
score = ocr_quality_score(extracted_text)
if score < 0.5:
    # Re-run with a higher-quality OCR engine (Textract, Azure DI)
    extracted_text = high_quality_ocr(document)
elif score < 0.7:
    # Flag for human review before indexing
    queue_for_review(document, reason="low_ocr_quality")
```

---

### Structure Corruption

PDF extraction sometimes corrupts document structure:

- **Multi-column layouts:** Text from two columns is interleaved: "The policy states that emThird-party ployees must submit vendors must also..."
- **Table cells merged with surrounding text:** The cell value runs into the next paragraph.
- **Footnotes merged into body text:** Footnote number becomes part of a sentence.
- **Reading order errors:** In complex PDF layouts, the extracted text reads in the wrong order.

**Detection and handling:**
Use a document intelligence library (Docling, Azure Document Intelligence, AWS Textract) that understands PDF structure and preserves reading order. These are more expensive than pdfminer but produce dramatically cleaner output for complex layouts.

```python
def parse_pdf(path: str, quality: str = "standard") -> ParsedDocument:
    if quality == "standard":
        # Fast, cheap — good for simple PDFs
        return pdfminer_parse(path)
    elif quality == "high":
        # Slow, expensive — for complex layouts with tables and multi-column
        return docling_parse(path)  # or azure_document_intelligence_parse(path)
```

Route documents to the appropriate parser based on a quick structural complexity assessment (page count, presence of tables, column count).

---

## Metadata Enrichment

Metadata is what makes citations precise and retrieval context-aware. A chunk without metadata is an orphan — you know what it says but not where it came from, who wrote it, or where within the document it appears.

### Document-Level Metadata

Every document in the index must have:

```python
@dataclass
class DocumentMetadata:
    # Identity
    document_id:     str          # system-generated UUID
    source_url:      str          # original location (S3 path, SharePoint URL, etc.)
    filename:        str          # original filename

    # Provenance
    title:           str          # document title (from PDF metadata, H1, or filename)
    authors:         list[str]    # author names (from PDF metadata, bylines, email headers)
    organization:    str | None   # publishing organization or department
    created_date:    date | None  # document creation date
    modified_date:   date | None  # last modified date
    version:         str | None   # "v2.3", "2024 Q1 edition", etc.
    document_type:   str          # "policy", "report", "contract", "manual", "email", etc.

    # Classification
    department:      str | None   # HR, Finance, Legal, Engineering, etc.
    access_level:    str          # "public", "internal", "confidential", "restricted"
    tags:            list[str]    # topic tags for coarse filtering

    # Ingestion tracking
    ingested_at:     datetime
    content_hash:    str          # for change detection
    schema_version:  int          # index schema version at ingestion time
```

**Automated extraction:**
- **Title:** PDF title field → H1 tag → first heading → filename without extension (fallback)
- **Authors:** PDF author field → byline parsing ("By Jane Smith" / "Author: Jane Smith") → email header ("From: Jane Smith <jane@company.com>")
- **Dates:** PDF creation/modification metadata → document header parsing → file system dates (least reliable)
- **Document type:** Filename patterns ("policy_", "report_", "contract_") + content classification

---

### Section and Chapter Metadata

Every chunk must know where it sits within the document hierarchy.

```python
@dataclass
class SectionMetadata:
    chapter_number:   str | None    # "3", "IV", "Chapter Three"
    chapter_title:    str | None    # "Leave Policy"
    section_number:   str | None    # "3.4", "4.2.1"
    section_title:    str | None    # "Parental Leave Eligibility"
    subsection_title: str | None    # "Director and Above"
    depth:            int           # heading depth: 1=H1, 2=H2, 3=H3
    section_path:     list[str]     # ["HR Policies", "Leave Policy", "Parental Leave"]
```

**Extraction:**

For DOCX: use python-docx heading styles to build the hierarchy.
For PDF with bookmarks: extract the PDF bookmark tree.
For PDF without bookmarks: heuristic heading detection — larger font size, bold, shorter text, ends without period.
For HTML: parse `<h1>` through `<h6>` tags.

```python
def extract_section_hierarchy(document: ParsedDocument) -> list[SectionMetadata]:
    sections = []
    current_path = []

    for element in document.elements:
        if element.type == "HEADING":
            depth = element.heading_level  # 1, 2, 3, etc.
            title = element.text.strip()

            # Maintain the path stack
            current_path = current_path[:depth - 1] + [title]

            sections.append(SectionMetadata(
                section_title=title,
                depth=depth,
                section_path=current_path.copy(),
            ))

    return sections
```

---

### Paragraph-Level Metadata

For fine-grained citation, every chunk needs a position identifier within its section.

```python
@dataclass
class ParagraphMetadata:
    page_number:        int | None    # physical page number (from PDF)
    paragraph_index:    int           # paragraph number within the section (1-indexed)
    chunk_index:        int           # chunk number within the document (1-indexed)
    total_chunks:       int           # total chunks in the document
    char_offset_start:  int           # character offset from document start
    char_offset_end:    int           # character offset from document start
    is_list_item:       bool          # is this paragraph a list element?
    list_index:         int | None    # position within a list (1, 2, 3...)
    is_footnote:        bool          # is this a footnote?
    footnote_number:    int | None    # footnote number if applicable
```

The `char_offset_start` and `char_offset_end` allow the UI to highlight the exact passage in the original document when displaying a citation.

---

### Table and Column Metadata

Tables require the richest metadata because a table cell out of context is meaningless.

```python
@dataclass
class TableMetadata:
    table_id:           str           # unique ID within document: "doc_042_table_003"
    table_index:        int           # table number within the document
    caption:            str | None    # table caption ("Table 3: Q3 Revenue by Region")
    page_number:        int | None    # page where the table appears
    section_path:       list[str]     # section this table belongs to
    column_headers:     list[str]     # ["Quarter", "Revenue", "Expenses", "Margin"]
    row_count:          int
    column_count:       int
    representation:     str           # "markdown" | "prose" | "json"
    has_merged_cells:   bool
    notes:              str | None    # any notes below the table

@dataclass
class TableCellMetadata:
    table_id:     str
    row_index:    int           # 0-indexed row position
    col_index:    int           # 0-indexed column position
    row_header:   str | None    # value of the row header for this row
    col_header:   str | None    # column header for this column
    value:        str           # cell value
```

**Table indexing strategy:**

Index tables in two complementary formats:

1. **Prose format** (for semantic retrieval):
```
Table 3 (Q3 Revenue by Region, from Financial Report Q3 2024, page 12):
In Q3 2024, North America had revenue of $4.2M and expenses of $2.1M, giving a margin of 50%.
EMEA had revenue of $2.8M and expenses of $1.6M, giving a margin of 43%.
Asia Pacific had revenue of $1.9M and expenses of $1.2M, giving a margin of 37%.
```

2. **Markdown format** (for LLM consumption in context):
```markdown
| Region | Revenue | Expenses | Margin |
|--------|---------|----------|--------|
| North America | $4.2M | $2.1M | 50% |
| EMEA | $2.8M | $1.6M | 43% |
| Asia Pacific | $1.9M | $1.2M | 37% |
```

Store both representations. Retrieve using the prose embedding; serve the markdown format to the LLM. This gives embedding quality (prose) and LLM readability (markdown).

---

### Fine-Grained Citation Schema

The goal: every claim in a generated answer can be cited with enough precision for a human to find the exact source in under 30 seconds.

```python
@dataclass
class Citation:
    # Document identity
    document_id:      str
    document_title:   str
    document_version: str | None
    source_url:       str

    # Authors and date
    authors:          list[str]
    publication_date: date | None
    access_date:      date          # when the document was last indexed

    # Location within document
    chapter:          str | None    # "Chapter 3: Leave Policy"
    section:          str | None    # "Section 3.4: Parental Leave"
    subsection:       str | None    # "Subsection 3.4.2: Director Eligibility"
    page_number:      int | None    # 34
    paragraph_number: int | None    # 2 (second paragraph of the section)

    # For table citations
    table_id:         str | None
    table_caption:    str | None    # "Table 5: Approval Thresholds by Role"
    table_row:        str | None    # "Director"
    table_column:     str | None    # "Maximum Approval Amount"

    # For image citations
    figure_id:        str | None
    figure_caption:   str | None    # "Figure 3: Org Chart Q3 2024"

    # Exact text
    verbatim_excerpt: str           # the exact sentence(s) retrieved

# Rendered citation examples:
# "Employee Handbook 2024 (v2.3), Chapter 3: Leave Policy, Section 3.4, Page 34, Paragraph 2"
# "Q3 Financial Report, Table 5: Approval Thresholds by Role, Row: Director, Column: Maximum Amount"
# "Org Chart Q3 2024 (Figure 3): [image description]"
```

**Why verbatim excerpt matters:**
The LLM may paraphrase the retrieved text. Including the verbatim excerpt in the citation lets the user check whether the paraphrase accurately represents the source, and shows exactly what text the answer was based on.

---

### Automated Metadata Extraction Pipeline

```python
class MetadataExtractor:

    def extract(self, document: RawDocument) -> DocumentWithMetadata:

        # 1. Document-level metadata
        doc_meta = DocumentMetadata(
            document_id   = generate_uuid(),
            source_url    = document.source_url,
            filename      = document.filename,
            title         = self._extract_title(document),
            authors       = self._extract_authors(document),
            created_date  = self._extract_date(document, "created"),
            modified_date = self._extract_date(document, "modified"),
            version       = self._extract_version(document),
            document_type = self._classify_document_type(document),
            access_level  = self._extract_access_level(document),
            content_hash  = sha256(document.raw_content),
        )

        # 2. Clean the content
        clean_content = self.cleaner.clean(document.raw_content)

        # 3. Parse structure
        elements = self.parser.parse(clean_content, document.format)

        # 4. Build section hierarchy
        sections = self._build_section_hierarchy(elements)

        # 5. Extract tables with metadata
        tables = self._extract_tables(elements, sections)

        # 6. Extract images with conversion
        images = self._process_images(elements, sections)

        # 7. Create chunks with full metadata
        chunks = self._create_chunks(
            elements, sections, tables, images, doc_meta
        )

        return DocumentWithMetadata(
            metadata=doc_meta,
            chunks=chunks,
            tables=tables,
            images=images,
        )

    def _create_chunks(self, elements, sections, tables, images, doc_meta):
        chunks = []
        page_number = 1
        paragraph_index = 0

        for element in elements:
            if element.type == "PAGE_BREAK":
                page_number += 1
                continue

            if element.type == "PARAGRAPH":
                paragraph_index += 1
                current_section = self._find_section(element, sections)

                chunk = Chunk(
                    content  = element.text,
                    metadata = {
                        # Document level
                        "document_id":    doc_meta.document_id,
                        "title":          doc_meta.title,
                        "authors":        doc_meta.authors,
                        "document_type":  doc_meta.document_type,
                        "access_level":   doc_meta.access_level,
                        "version":        doc_meta.version,

                        # Section level
                        "section_path":   current_section.section_path,
                        "chapter":        current_section.chapter_title,
                        "section":        current_section.section_title,

                        # Paragraph level
                        "page_number":    page_number,
                        "paragraph_index": paragraph_index,
                        "char_offset_start": element.char_start,
                        "char_offset_end":   element.char_end,
                    }
                )
                chunks.append(chunk)

        return chunks
```

---

## Image-to-Text Conversion

### When Images Contain Critical Information

Images are not decoration. In business documents they often contain the most important information:

| Image type | Information density | Conversion priority |
|------------|--------------------|--------------------|
| Charts (bar, line, pie) | High — quantitative trends and comparisons | High |
| Tables as images | Very high — structured data | Critical |
| Org charts / process flows | Medium — relationships and hierarchy | Medium |
| Infographics | Variable | Medium |
| Scanned signatures | Low for content | Low |
| Logos / decorative images | None | Skip |
| Mathematical formulas | High | High |
| Scanned text pages | Very high | Critical (OCR) |
| Architecture diagrams | Medium-high | Medium |
| Screenshots of UIs | Medium | Low-medium |

---

### Conversion Strategies by Image Type

**Strategy 1 — OCR for scanned text / scanned pages:**

For pages that are images of text (scanned documents, photographs of whiteboards):

```python
def ocr_image(image: PIL.Image) -> str:
    # Option A: Tesseract (free, acceptable for clean scans)
    import pytesseract
    return pytesseract.image_to_string(image, lang="eng")

    # Option B: Cloud OCR (paid, better for complex layouts)
    # AWS Textract, Azure Computer Vision, Google Cloud Vision
    # Use for: handwriting, complex layouts, rotated text, poor scan quality
```

Quality threshold: if Tesseract OCR quality score < 0.65, re-run with cloud OCR.

**Strategy 2 — Multimodal LLM for charts, graphs, and diagrams:**

```python
import base64
from openai import OpenAI

def describe_chart(image: PIL.Image, context: str) -> str:
    """
    Use a vision-capable LLM to describe the chart content.
    context: surrounding text to help the LLM understand the chart's topic.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Document context: {context}\n\n"
                        "Describe this chart in detail. Include:\n"
                        "1. Chart type (bar, line, pie, scatter, etc.)\n"
                        "2. Title and axis labels if visible\n"
                        "3. All data points or values shown\n"
                        "4. The key trend or conclusion the chart communicates\n"
                        "5. Any annotations or callouts\n"
                        "Format the data as a structured description, not just prose."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                }
            ]
        }],
        max_tokens=500
    )
    return response.choices[0].message.content
```

**Example output for a bar chart:**
```
Chart type: Grouped bar chart
Title: "Q3 2024 Revenue by Region and Product Line"
X-axis: Regions (North America, EMEA, Asia Pacific, LATAM)
Y-axis: Revenue in millions USD (0 to 8)

Data:
  North America: Enterprise $6.2M, SMB $3.1M, Consumer $1.8M
  EMEA: Enterprise $4.1M, SMB $1.9M, Consumer $0.8M
  Asia Pacific: Enterprise $2.3M, SMB $1.1M, Consumer $1.4M
  LATAM: Enterprise $0.9M, SMB $0.4M, Consumer $0.3M

Key trend: North America dominates Enterprise revenue at $6.2M, 2.5× EMEA.
Asia Pacific shows unusually high Consumer revenue relative to Enterprise,
suggesting a different customer mix in that region.
```

This description is rich enough to answer questions like "Which region had the highest Enterprise revenue in Q3?" or "How did Asia Pacific's consumer mix compare to other regions?"

**Strategy 3 — Tables as images:**

Tables rendered as images (common in older PDFs and Word documents exported to PDF) are especially valuable to recover precisely.

```python
def extract_table_from_image(image: PIL.Image, context: str) -> TableData:
    """Extract structured table data from an image of a table."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Extract this table into a structured format.\n"
                        "Return JSON: {\"caption\": str, \"headers\": [str], "
                        "\"rows\": [[str]], \"notes\": str}\n"
                        "Preserve all values exactly as shown. "
                        "If a cell is merged, repeat the value for each logical cell."
                    )
                },
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]
        }],
        response_format={"type": "json_object"},
        max_tokens=1000
    )
    return TableData(**json.loads(response.choices[0].message.content))
```

**Strategy 4 — Mathematical formulas:**

LaTeX formulas embedded in PDFs are frequently lost in text extraction. Two approaches:

```python
# Option A: Mathpix OCR API (specialised for math)
def extract_formula(image: PIL.Image) -> dict:
    # Returns {"latex": "\\frac{P(A \\cap B)}{P(B)}", "text": "P(A|B) = P(A and B) / P(B)"}
    response = mathpix_client.latex(image)
    return {"latex": response["latex"], "text": response["text"]}

# Option B: Multimodal LLM with explicit formula instruction
def describe_formula(image: PIL.Image) -> str:
    prompt = "Describe this mathematical formula in plain English, suitable for search. Also write it in LaTeX notation."
    # Returns: "Bayes' theorem: the probability of A given B equals the probability of A and B divided by the probability of B. LaTeX: P(A|B) = \\frac{P(A \\cap B)}{P(B)}"
```

**Strategy 5 — Infographics and process diagrams:**

```python
def describe_infographic(image: PIL.Image, context: str) -> str:
    prompt = (
        "This is an infographic or process diagram from a business document.\n"
        f"Document context: {context}\n\n"
        "Describe all information shown: steps, relationships, quantities, labels, "
        "arrows and what they connect, callout boxes and their content. "
        "Be exhaustive — assume the reader cannot see the image."
    )
    # Returns detailed prose description of all visual elements
```

**Strategy 6 — Routing by image type:**

```python
def route_image(image: PIL.Image, context: str) -> ImageConversionResult:
    image_type = classify_image_type(image)  # ML classifier or rule-based

    if image_type == "SCANNED_TEXT":
        text = ocr_image(image)
        return ImageConversionResult(text=text, type="ocr", source="tesseract")

    elif image_type == "TABLE":
        table_data = extract_table_from_image(image, context)
        text = table_to_prose(table_data) + "\n\n" + table_to_markdown(table_data)
        return ImageConversionResult(text=text, type="table", structured_data=table_data)

    elif image_type in ("BAR_CHART", "LINE_CHART", "PIE_CHART", "SCATTER"):
        text = describe_chart(image, context)
        return ImageConversionResult(text=text, type="chart", source="gpt-4o-vision")

    elif image_type == "FORMULA":
        formula = extract_formula(image)
        text = f"{formula['text']} (LaTeX: {formula['latex']})"
        return ImageConversionResult(text=text, type="formula")

    elif image_type in ("LOGO", "DECORATIVE", "SIGNATURE"):
        return ImageConversionResult(text=None, type="skip")  # do not index

    else:  # diagram, org chart, infographic
        text = describe_infographic(image, context)
        return ImageConversionResult(text=text, type="description", source="gpt-4o-vision")
```

---

### Storage and Indexing of Image-Derived Text

Store three representations for every converted image:

```python
@dataclass
class IndexedImage:
    image_id:        str           # unique ID: "doc_042_img_007"
    document_id:     str
    page_number:     int | None
    figure_number:   int | None    # "Figure 3"
    caption:         str | None    # caption text from the document
    image_type:      str           # "chart", "table", "formula", "diagram"
    conversion_text: str           # full text representation for indexing
    raw_image_path:  str           # path to original image file (S3 key)
    section_path:    list[str]     # where in the document this image appears

    # For structured data (tables):
    structured_data: dict | None   # JSON representation of table data
```

**Indexing:**
- Embed `conversion_text` for semantic retrieval
- Store `structured_data` separately for structured queries
- Keep `raw_image_path` for the UI — show the original image alongside the text in citations

**Citation format for images:**
```
[Source: Q3 Financial Report | Figure 3: Revenue by Region Bar Chart | Page 12]
"North America had the highest Enterprise revenue at $6.2M, 2.5× higher than EMEA ($4.1M).
 Asia Pacific showed an unusually high consumer revenue ratio compared to other regions."
[View original chart ↗]
```

---

## Full Ingestion Pipeline with All Three Stages

```
Raw Document
    │
    ▼
┌──────────────────────────────────────────────────┐
│ STAGE 1: DATA CLEANING                           │
│  - Boilerplate removal (headers, footers, legal) │
│  - Encoding fixes (ftfy, Unicode normalisation)  │
│  - Deduplication (content hash + MinHash LSH)    │
│  - OCR quality assessment and re-OCR if needed   │
│  - Structure corruption detection                │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│ STAGE 2: METADATA EXTRACTION                     │
│  Document level:                                 │
│    - Title, authors, dates, version, type        │
│  Section level:                                  │
│    - Chapter, section, subsection hierarchy      │
│  Paragraph level:                                │
│    - Page number, paragraph index, char offsets  │
│  Table level:                                    │
│    - Caption, column headers, row count, format  │
│  Image level:                                    │
│    - Figure number, caption, type classification │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│ STAGE 3: IMAGE-TO-TEXT CONVERSION                │
│  Route by image type:                            │
│    Scanned text  → Tesseract / Cloud OCR         │
│    Tables        → GPT-4o vision → JSON + prose  │
│    Charts        → GPT-4o vision → structured desc│
│    Formulas      → Mathpix / GPT-4o → LaTeX+text │
│    Diagrams      → GPT-4o vision → prose desc    │
│    Logos/decor   → Skip (do not index)           │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│ STAGE 4: CHUNKING WITH FULL METADATA ATTACHED    │
│  Every chunk carries: document_id, title,        │
│  authors, section_path, page_number,             │
│  paragraph_index, char_offsets                   │
│  Table chunks carry: table_id, headers, caption  │
│  Image chunks carry: figure_id, image_type       │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
               Embedding + Indexing
```

Output: every chunk in the index is clean, attributed to a specific location in a specific document version, and carries enough metadata to generate a citation precise enough for a human to find the source in under 30 seconds.
