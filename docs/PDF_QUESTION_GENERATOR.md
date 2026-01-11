# PDF Question Generator

Process PDFs using MinerU VLM for multimodal extraction and generate research questions with chunk-based context.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Requirements](#3-requirements)
4. [CLI Usage](#4-cli-usage)
5. [Output Format](#5-output-format)
6. [Configuration](#6-configuration)
7. [API Reference](#7-api-reference)
8. [Troubleshooting](#8-troubleshooting)

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

## 3. Requirements

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

## 4. CLI Usage

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

## 5. Output Format

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

## 6. Configuration

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

## 7. API Reference

### Main Functions

#### `process_pdf_simple()`
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

#### `extract_chunks_from_content_list()`
```python
def extract_chunks_from_content_list(
    content_list: list[dict],
    entities: list[dict] | None = None,
    limit: int = 30,
) -> list[ChunkContext]:
    """Extract chunks from MinerU content list."""
```

#### `format_chunks_as_context()`
```python
def format_chunks_as_context(
    chunks: list[ChunkContext],
    max_chars: int = 12000,
) -> str:
    """Format chunks as structured context for LLM."""
```

### LLM Utilities (lightrag_utils.py)

#### `get_ollama_llm_funcs()`
```python
def get_ollama_llm_funcs(
    llm_model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    timeout: float = 120.0,
) -> tuple[Callable, Callable]:
    """Get sync LLM and vision functions using Ollama."""
```

#### `get_ollama_llm_funcs_async()`
```python
async def get_ollama_llm_funcs_async(
    llm_model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    timeout: float = 120.0,
) -> tuple[Callable, Callable]:
    """Get async LLM and vision functions for modal processors."""
```

---

## 8. Troubleshooting

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
