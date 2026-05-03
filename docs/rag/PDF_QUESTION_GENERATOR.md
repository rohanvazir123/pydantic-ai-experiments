# PDF Question Generator

Process PDFs using MinerU VLM for multimodal extraction and generate research questions with chunk-based context.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [MinerU Parser (Core Module)](#3-mineru-parser-core-module)
4. [Requirements](#4-requirements)
5. [CLI Usage](#5-cli-usage)
6. [Output Format](#6-output-format)
7. [Configuration](#7-configuration)
8. [API Reference](#8-api-reference)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Overview

The PDF Question Generator is a tool that:
- Parses PDFs using MinerU 2.5 VLM (GPU-accelerated)
- Extracts text, tables, equations, and figures with VLM descriptions
- Generates research questions using chunk-based context
- Outputs questions with supporting chunk references

### System Flow

```
PDF → MinerU VLM → Content Extraction → Chunk Context → LLM → Questions
         │
         ├── Text blocks
         ├── Tables (HTML)
         ├── Equations (LaTeX)
         └── Figures (VLM descriptions)
```

---

## 2. Architecture

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PDF QUESTION GENERATOR WORKFLOW                          │
└─────────────────────────────────────────────────────────────────────────────┘

python -m rag.ingestion.processors.pdf_question_generator --simple <pdf_path>
    │
    ▼
process_pdf_simple() [pdf_question_generator.py:432]
    │
    ├──► get_ollama_llm_funcs()           # Sync LLM for direct calls
    ├──► get_ollama_llm_funcs_async()     # Async LLM for modal processors
    ├──► get_ollama_embedding_func()      # Embeddings
    │
    ├──► LightRAG(llm_model_func=async_llm_func)
    │       └──► initialize_storages()
    │
    ▼
MinerUParser.parse_file() [mineru.py:409]
    │
    ├──► _initialize_sync()
    │       ├──► torch.cuda.is_available()
    │       ├──► Qwen2VLForConditionalGeneration.from_pretrained()
    │       ├──► AutoProcessor.from_pretrained()
    │       └──► MinerUClient(backend="transformers")
    │
    ├──► _pdf_to_images()                  # pypdfium2 conversion
    │
    └──► [FOR EACH PAGE]:
            └──► _extract_page()
                    ├──► client.two_step_extract(image)
                    │       └──► Returns blocks: {type, content, bbox}
                    │
                    └──► [FOR EACH figure/image BLOCK]:
                            └──► _describe_figure()
                                    └──► qwen_vl_utils.process_vision_info()
    │
    ▼
Content Processing [pdf_question_generator.py:560]
    │
    ├──► Count content types:
    │       ├── text_content[]
    │       ├── table_content[]
    │       ├── equation_content[]
    │       └── image_content[]
    │
    ├──► [Optional] Modal Processors:
    │       ├──► TableModalProcessor(modal_caption_func=async_llm_func)
    │       └──► EquationModalProcessor(modal_caption_func=async_llm_func)
    │
    ▼
extract_chunks_from_content_list() [pdf_question_generator.py:204]
    │
    └──► ChunkContext(chunk_id, content, page_idx, content_type)
    │
    ▼
format_chunks_as_context() [pdf_question_generator.py:125]
    │
    └──► "[chunk_id=c1][page=0]\n{content}\n"
    │
    ▼
Question Generation [pdf_question_generator.py:647]
    │
    ├──► QUESTION_GENERATION_PROMPT.format(context_chunks=...)
    │
    └──► await async_llm_func(prompt, system_prompt=...)
            │
            └──► JSON: {questions: [{question, supported_by, difficulty}]}
    │
    ▼
Output Files
    │
    ├──► {pdf_name}_questions.json         # Structured JSON output
    └──► {pdf_name}_complete_output.txt    # Human-readable with context
```

**Source files for this workflow:**

| Symbol | File | Line |
|--------|------|------|
| `process_pdf_simple()` | [`pdf_question_generator.py`](../rag/ingestion/processors/pdf_question_generator.py#L432) | L432 |
| `process_pdf_with_raganything()` | [`pdf_question_generator.py`](../rag/ingestion/processors/pdf_question_generator.py#L269) | L269 |
| `main_async()` | [`pdf_question_generator.py`](../rag/ingestion/processors/pdf_question_generator.py#L778) | L778 |
| `get_ollama_llm_funcs()` | [`lightrag_utils.py`](../rag/ingestion/processors/lightrag_utils.py#L42) | L42 |
| `get_ollama_llm_funcs_async()` | [`lightrag_utils.py`](../rag/ingestion/processors/lightrag_utils.py#L101) | L101 |
| `get_ollama_embedding_func()` | [`lightrag_utils.py`](../rag/ingestion/processors/lightrag_utils.py#L254) | L254 |
| `extract_chunks_from_content_list()` | [`pdf_question_generator.py`](../rag/ingestion/processors/pdf_question_generator.py#L205) | L205 |
| `extract_chunks_from_lightrag()` | [`pdf_question_generator.py`](../rag/ingestion/processors/pdf_question_generator.py#L167) | L167 |
| `format_chunks_as_context()` | [`pdf_question_generator.py`](../rag/ingestion/processors/pdf_question_generator.py#L126) | L126 |
| `MinerUParser.parse_file()` | [`mineru.py`](../rag/ingestion/chunkers/mineru.py#L409) | L409 |

### Key Data Structures

```python
@dataclass
class ChunkContext:
    """A chunk with context for LLM."""
    chunk_id: str           # e.g., "c1", "c2", "m1" (multimodal)
    content: str            # Text content
    entity_name: str = ""   # Optional entity name
    entity_type: str = ""   # Optional entity type
    page_idx: int = 0       # Page number (0-indexed)
    content_type: str = "text"  # text, table, equation, image

@dataclass
class ProcessingResult:
    """Result from processing a PDF."""
    pdf_path: str
    title: str = ""
    num_pages: int = 0
    num_text_chunks: int = 0
    num_images: int = 0
    num_tables: int = 0
    num_equations: int = 0
    questions: list[str] = field(default_factory=list)
    context_chunks: str = ""      # Formatted context sent to LLM
    raw_llm_response: str = ""    # Raw LLM response
    error: str | None = None
```

---

## 3. MinerU Parser (Core Module)

**File:** [`rag/ingestion/chunkers/mineru.py`](../rag/ingestion/chunkers/mineru.py)

The MinerU Parser is the central component for multimodal document extraction. It uses the MinerU 2.5 vision-language model to extract text, tables, figures, and layout information from PDFs and images.

### 3.1 Module Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MINERU PARSER ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   MinerUParser  │
                              │                 │
                              │  ctx: Context   │
                              │  dpi: int       │
                              │  _lock: Lock    │
                              └────────┬────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│    initialize()     │   │    parse_file()     │   │      close()        │
│                     │   │                     │   │                     │
│ Load Qwen2VL model  │   │ PDF → Images →      │   │ Release GPU memory  │
│ Load AutoProcessor  │   │ Extract blocks      │   │ torch.cuda.empty()  │
│ Init MinerUClient   │   │ Describe figures    │   │                     │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
                                       │
                                       ▼
                          ┌─────────────────────┐
                          │   ParsedDocument    │
                          │                     │
                          │ source_path: str    │
                          │ total_pages: int    │
                          │ blocks: list[Block] │
                          │ metadata: dict      │
                          └─────────────────────┘
```

**Source symbols in [`mineru.py`](../rag/ingestion/chunkers/mineru.py):**

| Symbol | Line |
|--------|------|
| `BlockType` | [L70](../rag/ingestion/chunkers/mineru.py#L70) |
| `ExtractedBlock` | [L86](../rag/ingestion/chunkers/mineru.py#L86) |
| `ParsedDocument` | [L121](../rag/ingestion/chunkers/mineru.py#L121) |
| `MinerUContext` | [L151](../rag/ingestion/chunkers/mineru.py#L151) |
| `MinerUParser` | [L161](../rag/ingestion/chunkers/mineru.py#L161) |
| `MinerUParser.initialize()` | [L185](../rag/ingestion/chunkers/mineru.py#L185) |
| `MinerUParser.parse_file()` | [L409](../rag/ingestion/chunkers/mineru.py#L409) |
| `MinerUParser.parse_image()` | [L471](../rag/ingestion/chunkers/mineru.py#L471) |
| `MinerUParser.close()` | [L252](../rag/ingestion/chunkers/mineru.py#L252) |
| `parse_document()` | [L493](../rag/ingestion/chunkers/mineru.py#L493) |

### 3.2 Classes

#### `BlockType` (Enum)

Content block types extracted by MinerU:

```python
class BlockType(str, Enum):
    HEADER = "header"           # Section headers
    TITLE = "title"             # Document/page titles
    TEXT = "text"               # Regular text paragraphs
    LIST = "list"               # Bulleted/numbered lists
    TABLE = "table"             # Tables (HTML format)
    TABLE_CAPTION = "table_caption"
    FIGURE = "figure"           # Charts, graphs, diagrams
    FIGURE_CAPTION = "figure_caption"
    IMAGE = "image"             # Photos, screenshots
    EQUATION = "equation"       # Mathematical equations
    UNKNOWN = "unknown"         # Unrecognized content
```

#### `ExtractedBlock` (Pydantic Model)

A single content block extracted from a document:

```python
class ExtractedBlock(BaseModel):
    block_type: BlockType       # Type of content (text, table, figure, etc.)
    content: str                # Text content or HTML for tables
    bbox: list[float] | None    # Normalized bounding box [x1, y1, x2, y2] (0-1)
    page_number: int            # Page number (1-indexed)
    confidence: float           # Extraction confidence (0.0-1.0)
    figure_description: str | None      # VLM-generated description for figures
    figure_image_base64: str | None     # Base64-encoded cropped figure image
    metadata: dict[str, Any]    # Additional metadata

    def to_text(self) -> str:
        """Convert block to plain text for chunking."""
```

**Example Block:**
```python
ExtractedBlock(
    block_type=BlockType.FIGURE,
    content="",
    bbox=[0.1, 0.2, 0.9, 0.6],
    page_number=3,
    confidence=1.0,
    figure_description="A flowchart showing the data pipeline with three stages...",
    figure_image_base64="iVBORw0KGgoAAAANSUhEUg...",
    metadata={}
)
```

#### `ParsedDocument` (Pydantic Model)

Complete result of parsing a document:

```python
class ParsedDocument(BaseModel):
    source_path: str            # Path to source file
    total_pages: int            # Total number of pages
    blocks: list[ExtractedBlock]  # All extracted content blocks
    metadata: dict[str, Any]    # Document metadata (parser, dpi, etc.)

    def to_text(self) -> str:
        """Convert entire document to plain text."""

    def get_figures(self) -> list[ExtractedBlock]:
        """Get all figure/image blocks."""

    def get_tables(self) -> list[ExtractedBlock]:
        """Get all table blocks."""
```

#### `MinerUContext` (Dataclass)

Internal context holding model references:

```python
@dataclass
class MinerUContext:
    client: MinerUClient | None = None      # MinerU extraction client
    model: Qwen2VLForConditionalGeneration | None = None  # VLM model
    processor: AutoProcessor | None = None   # Text/image processor
    describe_figures: bool = True            # Generate figure descriptions
    initialized: bool = False                # Initialization status
```

#### `MinerUParser` (Main Class)

The main parser class for document extraction:

```python
class MinerUParser:
    def __init__(self, describe_figures: bool = True, dpi: int = 200):
        """
        Args:
            describe_figures: Generate VLM descriptions for figures
            dpi: DPI for PDF rendering (higher = better quality, more memory)
        """

    async def initialize(self) -> bool:
        """Initialize MinerU models (lazy loading). Returns success status."""

    async def parse_file(self, file_path: str | Path) -> ParsedDocument:
        """Parse a PDF or image file. Auto-initializes if needed."""

    async def parse_image(self, image: PIL.Image) -> list[ExtractedBlock]:
        """Parse a PIL Image directly."""

    async def close(self) -> None:
        """Release GPU memory and resources."""
```

### 3.3 Internal Methods

#### PDF Processing Pipeline

```python
def _pdf_to_images(self, pdf_path: str) -> list[Image.Image]:
    """
    Convert PDF pages to PIL Images using pypdfium2.

    Process:
    1. Open PDF with pdfium.PdfDocument()
    2. For each page:
       - Calculate scale from DPI (scale = dpi / 72)
       - Render page to bitmap
       - Convert bitmap to PIL Image
    3. Return list of images
    """
```

#### Page Extraction

```python
def _extract_page(self, image: Image.Image, page_number: int) -> list[ExtractedBlock]:
    """
    Extract content blocks from a single page image.

    Process:
    1. Call client.two_step_extract(image)
       └── Returns: [{type, content, bbox}, ...]
    2. For each raw block:
       - Map type string to BlockType enum
       - If figure/image with bbox:
         - Crop figure region
         - Convert to base64
         - Generate VLM description (if enabled)
       - Create ExtractedBlock
    3. Return list of blocks
    """
```

#### Figure Description

```python
def _describe_figure(self, image: Image.Image) -> str:
    """
    Generate VLM description for a cropped figure.

    Process:
    1. Build chat message with image + prompt:
       "Describe this diagram or figure in detail.
        Include any text, labels, arrows, and relationships between elements."
    2. Apply chat template with processor
    3. Process vision info with qwen_vl_utils
    4. Generate description with model (max_new_tokens=512)
    5. Decode and return description text
    """
```

#### Figure Cropping

```python
def _crop_figure(self, image: Image.Image, bbox: list[float], padding: int = 10) -> Image.Image:
    """
    Crop a figure from page image using normalized bbox.

    Args:
        image: Full page image
        bbox: Normalized coordinates [x1, y1, x2, y2] in range 0-1
        padding: Extra pixels around crop (default: 10)

    Process:
    1. Convert normalized coords to pixel coords:
       left = x1 * width - padding
       top = y1 * height - padding
       right = x2 * width + padding
       bottom = y2 * height + padding
    2. Crop and return region
    """
```

### 3.4 Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MINERU PROCESSING FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

MinerUParser.parse_file("document.pdf")
    │
    ├──► [1] Check file exists
    │
    ├──► [2] Auto-initialize if needed
    │       └──► initialize()
    │               ├──► Check CUDA available
    │               ├──► Load Qwen2VLForConditionalGeneration
    │               │       └──► "opendatalab/MinerU2.5-2509-1.2B"
    │               ├──► Load AutoProcessor
    │               └──► Create MinerUClient(backend="transformers")
    │
    ├──► [3] Determine file type
    │       ├──► .pdf → _parse_pdf_sync()
    │       └──► .jpg/.png/etc → _parse_image_sync()
    │
    ▼
_parse_pdf_sync(pdf_path)
    │
    ├──► [4] Convert PDF to images
    │       └──► _pdf_to_images()
    │               └──► pypdfium2: page.render(scale=dpi/72)
    │
    ├──► [5] Extract each page
    │       └──► FOR i, image IN enumerate(images):
    │               └──► _extract_page(image, page_number=i+1)
    │                       │
    │                       ├──► client.two_step_extract(image)
    │                       │       └──► Returns raw blocks
    │                       │
    │                       └──► FOR raw_block IN raw_blocks:
    │                               ├──► Map type → BlockType
    │                               │
    │                               ├──► IF figure/image AND bbox:
    │                               │       ├──► _crop_figure()
    │                               │       ├──► _image_to_base64()
    │                               │       └──► _describe_figure()
    │                               │               └──► VLM generates description
    │                               │
    │                               └──► Create ExtractedBlock
    │
    └──► [6] Return ParsedDocument
            └──► ParsedDocument(
                    source_path=pdf_path,
                    total_pages=len(images),
                    blocks=all_blocks,
                    metadata={"parser": "mineru", "dpi": self.dpi}
                )
```

### 3.5 MinerU Client Two-Step Extract

The `client.two_step_extract(image)` method performs:

1. **Step 1 - Layout Detection:** Identify regions and their types
2. **Step 2 - Content Recognition:** Extract text/content from each region

Returns list of dictionaries:
```python
[
    {"type": "title", "content": "Chapter 1: Introduction", "bbox": [0.1, 0.05, 0.9, 0.1]},
    {"type": "text", "content": "This chapter covers...", "bbox": [0.1, 0.12, 0.9, 0.3]},
    {"type": "figure", "content": "", "bbox": [0.2, 0.35, 0.8, 0.6]},
    {"type": "table", "content": "<table>...</table>", "bbox": [0.1, 0.65, 0.9, 0.9]},
]
```

### 3.6 Usage Examples

#### Basic PDF Parsing

```python
import asyncio
from rag.ingestion.chunkers.mineru import MinerUParser

async def parse_pdf():
    parser = MinerUParser(describe_figures=True, dpi=200)
    doc = await parser.parse_file("document.pdf")

    print(f"Pages: {doc.total_pages}")
    print(f"Total blocks: {len(doc.blocks)}")

    # Count by type
    from collections import Counter
    types = Counter(b.block_type.value for b in doc.blocks)
    print(f"Block types: {dict(types)}")

    # Get full text
    text = doc.to_text()
    print(f"Text length: {len(text)} chars")

    await parser.close()

asyncio.run(parse_pdf())
```

#### Extract Figures with Descriptions

```python
async def extract_figures():
    parser = MinerUParser(describe_figures=True)
    doc = await parser.parse_file("document.pdf")

    figures = doc.get_figures()
    print(f"Found {len(figures)} figures")

    for i, fig in enumerate(figures, 1):
        print(f"\nFigure {i} (page {fig.page_number}):")
        print(f"  BBox: {fig.bbox}")
        print(f"  Description: {fig.figure_description[:100]}...")

        # Save figure image
        if fig.figure_image_base64:
            import base64
            img_data = base64.b64decode(fig.figure_image_base64)
            with open(f"figure_{i}.png", "wb") as f:
                f.write(img_data)

    await parser.close()
```

#### Extract Tables

```python
async def extract_tables():
    parser = MinerUParser()
    doc = await parser.parse_file("document.pdf")

    tables = doc.get_tables()
    print(f"Found {len(tables)} tables")

    for i, table in enumerate(tables, 1):
        print(f"\nTable {i} (page {table.page_number}):")
        print(f"  HTML: {table.content[:200]}...")
        print(f"  Plain text: {table.to_text()[:200]}...")

    await parser.close()
```

#### Parse Single Image

```python
from PIL import Image

async def parse_image():
    parser = MinerUParser()

    # From file
    doc = await parser.parse_file("diagram.png")

    # Or from PIL Image directly
    img = Image.open("diagram.png")
    blocks = await parser.parse_image(img)

    for block in blocks:
        print(f"[{block.block_type.value}] {block.content[:50]}...")

    await parser.close()
```

#### Convenience Function

```python
from rag.ingestion.chunkers.mineru import parse_document

async def quick_parse():
    # Auto-manages parser lifecycle
    doc = await parse_document("document.pdf")
    print(doc.to_text()[:500])
```

### 3.7 CLI Usage

```bash
# Parse PDF and show results
python -m rag.ingestion.chunkers.mineru document.pdf

# Output:
# Parsing: document.pdf
# Using GPU: NVIDIA GeForce RTX 4060
# Loading MinerU model...
# PDF has 12 pages
# Extracted 13 blocks from page 1
# ...
# ============================================================
# RESULTS: 12 pages, 127 blocks
# ============================================================
# Block types: {'text': 98, 'header': 12, 'title': 5, 'figure': 7, 'list': 5}
#
# First 5 blocks:
#   [title] CS168: The Modern Algorithmic Toolbox...
#   [text] Tim Roughgarden & Gregory Valiant...
#   [header] 1 Consistent Hashing...
#   ...
```

### 3.8 Memory and Performance

| Setting | VRAM Usage | Processing Time (12 pages) |
|---------|------------|---------------------------|
| `dpi=150` | ~5GB | ~3 min |
| `dpi=200` (default) | ~6GB | ~5 min |
| `dpi=300` | ~8GB | ~8 min |
| `describe_figures=False` | -1GB | -30% |

**Tips:**
- Use `dpi=150` for faster processing with acceptable quality
- Set `describe_figures=False` if figure descriptions not needed
- Call `parser.close()` after processing to free GPU memory
- Process documents sequentially to avoid OOM errors

---

## 4. Requirements

### Hardware
- **CUDA GPU** required for MinerU VLM (tested with RTX 4060)
- 8GB+ VRAM recommended

### Dependencies
```bash
# Core packages
pip install mineru_vl_utils transformers pypdfium2

# MinerU model (auto-downloaded on first run)
# opendatalab/MinerU2.5-2509-1.2B

# LightRAG and RAG-Anything
pip install lightrag raganything

# Ollama (for LLM/embeddings)
ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### Verify Setup
```bash
# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check MinerU
python -c "from mineru_vl_utils import MinerUClient; print('OK')"

# Check Ollama
curl http://localhost:11434/api/tags
```

---

## 5. CLI Usage

### Basic Usage
```bash
# Process a single PDF (simple mode with MinerU)
python -m rag.ingestion.processors.pdf_question_generator --simple <pdf_path>

# With clean (removes previous results)
python -m rag.ingestion.processors.pdf_question_generator --simple --clean <pdf_path>
```

### Examples
```bash
# Process lecture PDF
python -m rag.ingestion.processors.pdf_question_generator --simple \
    test_rag_anything_pdfs_for_question_generation/l1.pdf

# List PDFs in a directory
python -m rag.ingestion.processors.pdf_question_generator \
    --list-dir C:/Users/rohan/Desktop/csd168

# Use OpenAI instead of Ollama
python -m rag.ingestion.processors.pdf_question_generator --simple \
    --api-key YOUR_OPENAI_KEY <pdf_path>
```

### Command-Line Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `pdf_path` | Path to PDF file | Required |
| `--simple`, `-s` | Use simple mode (MinerU + modal processors) | False |
| `--clean`, `-c` | Clean previous results before running | False |
| `--use-ollama` | Use Ollama for LLM/embeddings | True |
| `--api-key` | OpenAI API key (disables Ollama) | None |
| `--base-url` | Optional API base URL | None |
| `--working-dir`, `-w` | Working directory for storage | `./pdf_processing_<name>` |
| `--list-dir` | List PDFs in directory (no processing) | None |

---

## 6. Output Format

### JSON Output (`{pdf_name}_questions.json`)
```json
{
  "pdf_path": "path/to/file.pdf",
  "title": "l1",
  "statistics": {
    "pages": 12,
    "text_chunks": 121,
    "tables": 0,
    "equations": 1,
    "images": 5
  },
  "questions": [
    {
      "question": "What are the key concepts in consistent hashing?",
      "supported_by": ["c1", "c7"],
      "difficulty": "medium"
    },
    {
      "question": "How does consistent hashing address issues in present-day systems?",
      "supported_by": ["c8"],
      "difficulty": "medium"
    }
  ],
  "entities": ["Chunks used: c1, c2, c3, c4, c5, c6, c7, c8, c9, c10"],
  "error": null
}
```

### Complete Output (`{pdf_name}_complete_output.txt`)
```
================================================================================
PDF QUESTION GENERATOR OUTPUT
PDF: l1
Path: path/to/file.pdf
================================================================================

STATISTICS
----------------------------------------
Pages: 12
Text chunks: 121
Tables: 0
Equations: 1
Images: 5

================================================================================
CONTEXT CHUNKS (sent to LLM)
================================================================================

[chunk_id=c1][page=0]
CS168: The Modern Algorithmic Toolbox
Lecture #1: Introduction and Consistent Hashing
Tim Roughgarden & Gregory Valiant
...

[chunk_id=c2][page=1]
...

================================================================================
RAW LLM RESPONSE
================================================================================

{
  "questions": [
    ...
  ]
}

================================================================================
GENERATED QUESTIONS
================================================================================

1. What are the key concepts in consistent hashing?
   Supported by: c1, c7
   Difficulty: medium

2. How does consistent hashing address issues in present-day systems?
   Supported by: c8
   Difficulty: medium
```

---

## 7. Configuration

### LLM Functions

The script uses two types of LLM functions:

```python
# Sync functions (for direct calls)
llm_func, vision_func = get_ollama_llm_funcs()

# Async functions (for modal processors & LightRAG)
async_llm_func, async_vision_func = await get_ollama_llm_funcs_async()
```

### Question Generation Prompt

```python
QUESTION_GENERATION_PROMPT = """TASK:
Generate 5-8 high-quality research questions based on the provided context chunks.
Questions should:
1. Test understanding of key concepts
2. Explore relationships between topics
3. Encourage critical thinking
4. Be answerable using the provided chunks

CONTEXT:
{context_chunks}

OUTPUT FORMAT (valid JSON only):
{{
    "questions": [
        {{
            "question": "What is the relationship between X and Y?",
            "supported_by": ["c1", "c2"],
            "difficulty": "medium"
        }}
    ]
}}
"""
```

### Chunk Context Format

```python
def format_chunks_as_context(chunks: list[ChunkContext], max_chars: int = 12000) -> str:
    """Format: [chunk_id=c1][page=0]\n{content}\n"""
```

---

## 8. API Reference

### Main Functions

#### [`process_pdf_simple()`](../rag/ingestion/processors/pdf_question_generator.py#L432)
```python
async def process_pdf_simple(
    pdf_path: str,
    working_dir: str,
    use_ollama: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ProcessingResult:
    """Process PDF using MinerU VLM + modal processors."""
```

#### [`extract_chunks_from_content_list()`](../rag/ingestion/processors/pdf_question_generator.py#L205)
```python
def extract_chunks_from_content_list(
    content_list: list[dict],
    entities: list[dict] | None = None,
    limit: int = 30,
) -> list[ChunkContext]:
    """Extract chunks from MinerU content list."""
```

#### [`format_chunks_as_context()`](../rag/ingestion/processors/pdf_question_generator.py#L126)
```python
def format_chunks_as_context(
    chunks: list[ChunkContext],
    max_chars: int = 12000,
) -> str:
    """Format chunks as structured context for LLM."""
```

### LLM Utilities ([`lightrag_utils.py`](../rag/ingestion/processors/lightrag_utils.py))

#### [`get_ollama_llm_funcs()`](../rag/ingestion/processors/lightrag_utils.py#L42)
```python
def get_ollama_llm_funcs(
    llm_model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    timeout: float = 120.0,
) -> tuple[Callable, Callable]:
    """Get sync LLM and vision functions using Ollama."""
```

#### [`get_ollama_llm_funcs_async()`](../rag/ingestion/processors/lightrag_utils.py#L101)
```python
async def get_ollama_llm_funcs_async(
    llm_model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    timeout: float = 120.0,
) -> tuple[Callable, Callable]:
    """Get async LLM and vision functions for modal processors."""
```

---

## 9. Troubleshooting

### 1. "CUDA GPU required for MinerU"
MinerU VLM requires a CUDA-capable GPU:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```
If no GPU, the script falls back to PyPDF2 (text-only extraction).

### 2. "object str can't be used in 'await' expression"
This occurs when sync functions are passed to modal processors. Ensure:
- LightRAG initialized with `async_llm_func`
- Modal processors use `async_llm_func` as `modal_caption_func`

### 3. "Failed to parse questions JSON"
The LLM sometimes returns malformed JSON. The script has fallback parsing:
```python
# Fallback: extract questions from raw response
for line in response.split("\n"):
    if "?" in line and len(line) > 10:
        questions.append({"question": line, "supported_by": []})
```

### 4. "MinerU VLM not available"
Install MinerU dependencies:
```bash
pip install mineru_vl_utils transformers pypdfium2 qwen_vl_utils
```

### 5. Ollama connection refused
Start Ollama server:
```bash
ollama serve
```

### 6. Out of GPU memory
MinerU uses ~6GB VRAM. Options:
- Close other GPU applications
- Reduce DPI: `MinerUParser(dpi=150)` instead of default 200
- Process fewer pages at a time

---

## Quick Reference

### Process PDF and Generate Questions
```python
import asyncio
from rag.ingestion.processors.pdf_question_generator import process_pdf_simple

async def main():
    result = await process_pdf_simple(
        pdf_path="document.pdf",
        working_dir="./output",
        use_ollama=True,
    )

    print(f"Pages: {result.num_pages}")
    print(f"Questions: {len(result.questions)}")
    for q in result.questions:
        print(f"- {q['question']}")
        print(f"  Supported by: {q.get('supported_by', [])}")

asyncio.run(main())
```

### Use MinerU Parser Directly
```python
import asyncio
from rag.ingestion.chunkers.mineru import MinerUParser

async def parse_pdf():
    parser = MinerUParser(describe_figures=True)
    doc = await parser.parse_file("document.pdf")

    print(f"Pages: {doc.total_pages}")
    print(f"Blocks: {len(doc.blocks)}")

    for block in doc.blocks:
        print(f"[{block.block_type.value}] {block.content[:100]}...")
        if block.figure_description:
            print(f"  Figure: {block.figure_description[:50]}...")

    await parser.close()

asyncio.run(parse_pdf())
```
