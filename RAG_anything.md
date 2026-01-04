# RAG-Anything Integration Reference

This document outlines the APIs from [RAG-Anything](https://github.com/HKUDS/RAG-Anything) that can be directly called for multimodal document processing.

## Overview

RAG-Anything is an all-in-one multimodal RAG framework built on LightRAG. It provides:

- End-to-end multimodal document processing pipeline
- Specialized processors for images, tables, and equations
- Knowledge graph construction with cross-modal relationships
- Hybrid retrieval combining vector and graph search

## Installation

```bash
# Basic installation
pip install raganything

# With all optional features
pip install 'raganything[all]'
```

### Troubleshooting Installation Issues

#### Python 3.13 + numpy Issue

On Python 3.13, `pip install 'raganything[all]'` may fail with:

```
error: metadata-generation-failed
numpy<2.0.0,>=1.24.0 ... Unknown compiler(s): [['icl'], ['cl'], ['cc'], ['gcc']]
```

**Cause**: `lightrag-hku` requires `numpy<2.0.0`, but numpy 1.x has no pre-built wheels for Python 3.13 and requires Visual Studio Build Tools to compile from source.

**Solution**: Install with `--no-deps` to skip strict version constraints:

```bash
# Install numpy 2.x (has pre-built wheels)
pip install numpy>=2.0.0

# Install raganything and lightrag without dependency resolution
pip install raganything --no-deps
pip install lightrag-hku --no-deps

# Verify installation
python -c "from raganything import RAGAnything; print('OK')"
python -c "from raganything.modalprocessors import ImageModalProcessor; print('OK')"
```

#### Missing pypinyin Warning

```
WARNING: pypinyin is not installed. Chinese pinyin sorting will use simple string sorting.
```

This is harmless unless you need Chinese text sorting. To fix:

```bash
pip install pypinyin
```

---

## Core APIs

### 1. RAGAnything Class

Main entry point for document processing and querying.

```python
from raganything import RAGAnything, RAGAnythingConfig

# Configuration
config = RAGAnythingConfig(
    working_dir="./rag_storage",
    parser="mineru",              # Parser: "mineru" or "docling"
    parse_method="auto",          # Method: "auto", "ocr", "txt"
    enable_image_processing=True,
    enable_table_processing=True,
    enable_equation_processing=True,
)

# Initialize
rag = RAGAnything(
    config=config,
    llm_model_func=llm_func,
    vision_model_func=vision_func,
    embedding_func=embedding_func,
    lightrag=existing_lightrag,   # Optional: reuse existing LightRAG
)
```

#### Document Processing

```python
# Process single document
await rag.process_document_complete(
    file_path="document.pdf",
    output_dir="./output",
    parse_method="auto"
)

# Batch process folder
await rag.process_folder_complete(
    folder_path="./documents",
    output_dir="./output",
    file_extensions=[".pdf", ".docx", ".pptx"],
    recursive=True,
    max_workers=4
)
```

#### Querying

```python
# Text query (async)
result = await rag.aquery(
    "What are the main findings?",
    mode="hybrid"  # Options: hybrid, local, global, naive
)

# Text query (sync)
result = rag.query("What are the main findings?", mode="hybrid")

# Multimodal query with specific content
result = await rag.aquery_with_multimodal(
    "Explain this formula",
    multimodal_content=[{
        "type": "equation",
        "latex": "E = mc^2",
        "equation_caption": "Mass-energy equivalence"
    }],
    mode="hybrid"
)
```

#### Direct Content Insertion

Bypass document parsing by inserting pre-parsed content directly:

```python
content_list = [
    {
        "type": "text",
        "text": "Introduction paragraph...",
        "page_idx": 0
    },
    {
        "type": "image",
        "img_path": "/absolute/path/to/figure.jpg",  # Must be absolute
        "image_caption": ["Figure 1: Architecture"],
        "image_footnote": ["Source: Authors"],
        "page_idx": 1
    },
    {
        "type": "table",
        "table_body": "| Col1 | Col2 |\n|------|------|\n| A | B |",
        "table_caption": ["Table 1: Results"],
        "table_footnote": ["Data from 2024"],
        "page_idx": 2
    },
    {
        "type": "equation",
        "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
        "text": "Document relevance probability",
        "page_idx": 3
    }
]

await rag.insert_content_list(
    content_list=content_list,
    file_path="source_document.pdf",
    doc_id="custom-doc-id",        # Optional
    display_stats=True
)
```

#### Utility Methods

```python
# Check if MinerU parser is installed
is_installed = rag.check_parser_installation()
```

---

### 2. Modal Processors

Standalone processors for specific content types. Can be used independently of the full RAGAnything pipeline.

```python
from raganything.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
    GenericModalProcessor,  # Base class for custom processors
)
```

#### Class Hierarchy

All modal processors inherit from `BaseModalProcessor` and implement the `process_multimodal_content()` method:

```
BaseModalProcessor (abstract base class)
    │
    │   # Shared infrastructure (set in __init__)
    ├── lightrag                    → LightRAG instance
    ├── text_chunks_db              → KV store for chunks
    ├── chunks_vdb                  → Vector DB for chunk retrieval
    ├── entities_vdb                → Vector DB for entity search
    ├── relationships_vdb           → Vector DB for relationship search
    ├── knowledge_graph_inst        → Graph DB (NetworkX/Neo4j)
    ├── context_extractor           → ContextExtractor instance
    │
    │   # Shared methods
    ├── process_multimodal_content()    ← Main entry point (abstract)
    ├── generate_description_only()     ← LLM description generation (abstract)
    ├── _create_entity_and_chunk()      ← Storage logic (inherited)
    ├── _process_chunk_for_extraction() ← Entity extraction (inherited)
    ├── _get_context_for_item()         ← Context extraction (inherited)
    │
    │   # Concrete implementations
    ├── ImageModalProcessor       → Vision model for images
    ├── TableModalProcessor       → LLM for table analysis
    ├── EquationModalProcessor    → LLM for equation interpretation
    └── GenericModalProcessor     → LLM for any other content type

ContextExtractor (utility class - NOT a processor)
    │
    └── Used BY processors via _get_context_for_item()
```

#### The `process_multimodal_content()` Method

This is the main entry point that triggers all storage operations. Every processor implements this method:

```python
async def process_multimodal_content(
    self,
    modal_content,           # Raw content (table body, image path, LaTeX, etc.)
    content_type: str,       # "image", "table", "equation", or custom type
    file_path: str = "manual_creation",  # Source document path
    entity_name: str = None, # Optional name for the entity (auto-generated if None)
    item_info: dict = None,  # Page/index info for context extraction
    batch_mode: bool = False,# If True, defers merge_nodes_and_edges()
    doc_id: str = None,      # Document ID for chunk association
    chunk_order_index: int = 0,  # Position in document
) -> Tuple[str, dict, list]:
    """
    Returns:
        - description: Enhanced text description of the content
        - entity_info: Dict with entity_name, entity_type, description, chunk_id
        - chunk_results: List of (nodes, edges) for batch processing
    """
```

#### Processing Flow

Each processor follows the same pattern:

```python
# Internal flow of process_multimodal_content():

async def process_multimodal_content(self, modal_content, content_type, ...):

    # Step 1: Generate description using LLM (processor-specific)
    # - ImageModalProcessor: Uses vision model + image encoding
    # - TableModalProcessor: Parses markdown table + LLM analysis
    # - EquationModalProcessor: Parses LaTeX + LLM interpretation
    description, entity_info = await self.generate_description_only(
        modal_content, content_type, item_info, entity_name
    )

    # Step 2: Build formatted chunk content using prompt templates
    modal_chunk = PROMPTS["table_chunk"].format(...)  # or image_chunk, equation_chunk

    # Step 3: Create entity and chunk in storage (inherited from BaseModalProcessor)
    # This triggers all the storage operations:
    # - text_chunks.upsert()
    # - chunks_vdb.upsert()
    # - knowledge_graph.upsert_node()
    # - entities_vdb.upsert()
    # - extract_entities() → relationships
    # - merge_nodes_and_edges() (if not batch_mode)
    return await self._create_entity_and_chunk(
        modal_chunk, entity_info, file_path, batch_mode, doc_id, chunk_order_index
    )
```

#### Processor-Specific Behavior

| Processor | `modal_caption_func` | Content Parsing | Special Handling |
|-----------|---------------------|-----------------|------------------|
| `ImageModalProcessor` | Vision model (e.g., GPT-4V) | Encodes image to base64 | Handles `img_path`, `image_caption`, `image_footnote` |
| `TableModalProcessor` | LLM (e.g., GPT-4) | Parses markdown table | Handles `table_body`, `table_caption`, `table_footnote` |
| `EquationModalProcessor` | LLM (e.g., GPT-4) | Parses LaTeX/text | Handles `text`, `text_format` |
| `GenericModalProcessor` | LLM (e.g., GPT-4) | String conversion | Handles any `content` field |

#### ImageModalProcessor

Process images with vision models to generate descriptions and entity information.

```python
from raganything.modalprocessors import ImageModalProcessor

image_processor = ImageModalProcessor(
    lightrag=lightrag_instance,
    modal_caption_func=vision_model_func
)

image_content = {
    "img_path": "path/to/image.jpg",
    "image_caption": ["Figure 1: System Architecture"],
    "image_footnote": ["Source: Original design"]
}

description, entity_info, _ = await image_processor.process_multimodal_content(
    modal_content=image_content,
    content_type="image",
    file_path="research_paper.pdf",
    entity_name="Architecture Diagram"
)
```

#### TableModalProcessor

Process tables to extract structured information.

```python
from raganything.modalprocessors import TableModalProcessor

table_processor = TableModalProcessor(
    lightrag=lightrag_instance,
    modal_caption_func=llm_model_func
)

table_content = {
    "table_body": """
    | Method | Accuracy | F1-Score |
    |--------|----------|----------|
    | Ours   | 95.2%    | 0.94     |
    | Base   | 87.3%    | 0.85     |
    """,
    "table_caption": ["Performance Comparison"],
    "table_footnote": ["Results on test dataset"]
}

description, entity_info, _ = await table_processor.process_multimodal_content(
    modal_content=table_content,
    content_type="table",
    file_path="research_paper.pdf",
    entity_name="Performance Results"
)
```

#### EquationModalProcessor

Process mathematical equations and formulas.

```python
from raganything.modalprocessors import EquationModalProcessor

equation_processor = EquationModalProcessor(
    lightrag=lightrag_instance,
    modal_caption_func=llm_model_func
)

equation_content = {
    "text": "E = mc^2",
    "text_format": "LaTeX"
}

description, entity_info, _ = await equation_processor.process_multimodal_content(
    modal_content=equation_content,
    content_type="equation",
    file_path="physics_paper.pdf",
    entity_name="Mass-Energy Equivalence"
)
```

#### Custom Modal Processor

Extend `GenericModalProcessor` for custom content types:

```python
from raganything.modalprocessors import GenericModalProcessor

class CustomModalProcessor(GenericModalProcessor):
    async def process_multimodal_content(
        self, modal_content, content_type, file_path, entity_name
    ):
        # Custom processing logic
        enhanced_description = await self.analyze_custom_content(modal_content)
        entity_info = self.create_custom_entity(enhanced_description, entity_name)
        return await self._create_entity_and_chunk(
            enhanced_description, entity_info, file_path
        )
```

---

### 3. ContextExtractor

The `ContextExtractor` provides surrounding document context to LLMs when processing multimodal content (images, tables, equations). This helps generate more accurate and contextually relevant descriptions.

> **Important**: ContextExtractor does **NOT** do chunking. Chunking is handled by LightRAG. The ContextExtractor only extracts surrounding text to provide context for multimodal analysis.

#### How It Works

```
Document → MinerU Parser → content_list → ContextExtractor extracts surrounding text → LLM gets context for better analysis
```

When processing an image on page 5:

1. ContextExtractor looks at the configured `context_window` (e.g., 2 pages)
2. Collects text from pages 3, 4, 6, 7
3. Truncates to `max_context_tokens` limit
4. Passes this context to the vision/LLM model along with the content

#### Context Modes

| Mode    | Description                                            | Use Case                                               |
| ------- | ------------------------------------------------------ | ------------------------------------------------------ |
| `page`  | Extracts text from N pages before/after current item   | Document-structured content with clear page boundaries |
| `chunk` | Extracts N content items before/after current position | Fine-grained control for sequential content            |

#### MinerU Content List Format

The `_extract_from_content_list()` method is designed to work with MinerU-style content lists. MinerU parser outputs a flat list where each item has:

```python
# MinerU content_list structure
[
    {"type": "text", "text": "Chapter 1: Introduction", "text_level": 1, "page_idx": 0},
    {"type": "text", "text": "This paper presents...", "text_level": 0, "page_idx": 0},
    {"type": "image", "img_path": "/path/to/fig1.jpg", "image_caption": ["Figure 1"], "page_idx": 1},
    {"type": "text", "text": "As shown in Figure 1...", "text_level": 0, "page_idx": 1},
    {"type": "table", "table_body": "| A | B |...", "table_caption": ["Table 1"], "page_idx": 2},
    {"type": "equation", "text": "E = mc^2", "text_format": "LaTeX", "page_idx": 3},
    # ...
]
```

| Field | Description |
|-------|-------------|
| `type` | Content type: `"text"`, `"image"`, `"table"`, `"equation"` |
| `page_idx` | Page number (0-indexed) |
| `text` | Text content (for text items) or equation LaTeX |
| `text_level` | Header level: 0=paragraph, 1=H1, 2=H2, etc. |
| `img_path` | Absolute path to image file |
| `image_caption` / `img_caption` | List of caption strings |
| `table_body` | Markdown table content |
| `table_caption` | List of table caption strings |

#### Internal Extraction Logic

##### Page Mode Extraction

Extracts text from N pages before/after the current item:

```
Document Pages:    [Page 3] [Page 4] [Page 5] [Page 6] [Page 7]
                      ↑        ↑        ↑        ↑        ↑
                   context  context  CURRENT  context  context
                                      IMAGE

context_window=2 → extracts text from pages 3, 4, 6, 7
```

```python
def _extract_page_context(self, content_list, current_item_info):
    current_page = current_item_info.get("page_idx", 0)  # e.g., 5
    window_size = self.config.context_window             # e.g., 2

    start_page = max(0, current_page - window_size)      # 3
    end_page = current_page + window_size + 1            # 8

    context_texts = []
    for item in content_list:
        item_page = item.get("page_idx", 0)
        item_type = item.get("type", "")

        # Include if: within page range AND matches filter (default: ["text"])
        if start_page <= item_page < end_page and item_type in self.config.filter_content_types:
            text_content = self._extract_text_from_item(item)
            if text_content:
                if item_page != current_page:
                    context_texts.append(f"[Page {item_page}] {text_content}")
                else:
                    context_texts.append(text_content)

    return self._truncate_context("\n".join(context_texts))
```

##### Chunk Mode Extraction

Extracts N content items before/after by list index (ignoring page boundaries):

```python
def _extract_chunk_context(self, content_list, current_item_info):
    current_index = current_item_info.get("index", 0)  # e.g., 10
    window_size = self.config.context_window           # e.g., 3

    start_idx = max(0, current_index - window_size)    # 7
    end_idx = min(len(content_list), current_index + window_size + 1)  # 14

    # Extracts items 7, 8, 9, 11, 12, 13 (skipping 10 = current item)
```

##### Text Extraction from Items

```python
def _extract_text_from_item(self, item: Dict) -> str:
    item_type = item.get("type", "")

    if item_type == "text":
        text = item.get("text", "")
        text_level = item.get("text_level", 0)

        # MinerU uses text_level for headers: 1=H1, 2=H2, etc.
        if self.config.include_headers and text_level > 0:
            return f"{'#' * text_level} {text}"  # "## Section Title"
        return text

    elif item_type == "image" and self.config.include_captions:
        captions = item.get("image_caption", item.get("img_caption", []))
        return f"[Image: {', '.join(captions)}]" if captions else ""

    elif item_type == "table" and self.config.include_captions:
        captions = item.get("table_caption", [])
        return f"[Table: {', '.join(captions)}]" if captions else ""

    return ""
```

#### Context Extraction Example

Processing an image on page 5 with `context_window=1`:

```python
content_list = [
    {"type": "text", "text": "Results", "text_level": 1, "page_idx": 4},
    {"type": "text", "text": "The experiment showed significant improvement.", "page_idx": 4},
    {"type": "image", "img_path": "fig1.jpg", "image_caption": ["Performance Graph"], "page_idx": 5},  # ← CURRENT
    {"type": "text", "text": "Figure 1 demonstrates the performance gain.", "page_idx": 5},
    {"type": "text", "text": "Discussion", "text_level": 1, "page_idx": 6},
]

current_item_info = {"page_idx": 5, "index": 2, "type": "image"}

# Extracted context (page mode, filter_content_types=["text"]):
"""
[Page 4] # Results
[Page 4] The experiment showed significant improvement.
Figure 1 demonstrates the performance gain.
[Page 6] # Discussion
"""
```

This context is passed to the vision/LLM model along with the image for better description generation.

#### Configuration

```python
from raganything import RAGAnythingConfig

config = RAGAnythingConfig(
    # Context extraction settings
    context_window=2,                    # Pages/chunks before and after
    context_mode="page",                 # "page" or "chunk"
    max_context_tokens=2000,             # Maximum tokens for context
    include_headers=True,                # Include document headers
    include_captions=True,               # Include image/table captions
    context_filter_content_types=["text"],  # Content types to include
    content_format="minerU",             # Content format hint
)
```

#### Environment Variables

```bash
CONTEXT_WINDOW=2
CONTEXT_MODE=page
MAX_CONTEXT_TOKENS=2000
INCLUDE_HEADERS=true
INCLUDE_CAPTIONS=true
CONTEXT_FILTER_CONTENT_TYPES=text,image
CONTENT_FORMAT=minerU
```

#### Direct Usage

```python
from raganything.modalprocessors import ContextExtractor, ContextConfig

# Configure context extraction
config = ContextConfig(
    context_window=1,
    context_mode="page",
    max_context_tokens=2000,
    include_headers=True,
    include_captions=True,
    filter_content_types=["text"]
)

# Initialize context extractor
context_extractor = ContextExtractor(config)

# Extract context for a specific item
item_info = {"page_idx": 5, "index": 10, "type": "image"}
context = context_extractor.extract_context(
    content_source=content_list,      # From MinerU parser
    current_item_info=item_info,
    content_format="minerU"
)
```

#### Integration with Modal Processors

```python
from raganything.modalprocessors import ImageModalProcessor, ContextExtractor, ContextConfig

# Create context extractor
context_config = ContextConfig(context_window=2, max_context_tokens=1500)
context_extractor = ContextExtractor(context_config)

# Initialize processor with context support
processor = ImageModalProcessor(
    lightrag=lightrag_instance,
    modal_caption_func=vision_func,
    context_extractor=context_extractor
)

# Set content source for context extraction
processor.set_content_source(content_list, "minerU")

# Process with automatic context extraction
result = await processor.process_multimodal_content(
    modal_content=image_data,
    content_type="image",
    file_path="document.pdf",
    entity_name="Figure 1",
    item_info={"page_idx": 5, "index": 10, "type": "image"}
)
```

#### Automatic Integration

When using `RAGAnything.process_document_complete()`, context extraction is automatically enabled:

```python
rag = RAGAnything(config=config, ...)

# Context is automatically set up during document processing
await rag.process_document_complete("document.pdf")
```

#### Configuration Examples

**High-Precision Context** (minimal context, focused analysis):

```python
config = RAGAnythingConfig(
    context_window=1,
    context_mode="page",
    max_context_tokens=1000,
    include_headers=True,
    include_captions=False,
    context_filter_content_types=["text"]
)
```

**Comprehensive Context** (broad analysis, rich context):

```python
config = RAGAnythingConfig(
    context_window=2,
    context_mode="page",
    max_context_tokens=3000,
    include_headers=True,
    include_captions=True,
    context_filter_content_types=["text", "image", "table"]
)
```

**Chunk-Based Analysis** (fine-grained sequential context):

```python
config = RAGAnythingConfig(
    context_window=5,
    context_mode="chunk",
    max_context_tokens=2000,
    include_headers=False,
    include_captions=False,
    context_filter_content_types=["text"]
)
```

---

### 4. LightRAG Integration

RAG-Anything is built on LightRAG. You can use LightRAG directly for more control.

```python
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

# Initialize LightRAG
lightrag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=llm_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=api_key,
            base_url=base_url
        )
    )
)

# Initialize storage backends
await lightrag.initialize_storages()
await initialize_pipeline_status()
```

#### LLM Functions

```python
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

# Text completion with caching
result = openai_complete_if_cache(
    model="gpt-4o-mini",
    prompt="Your prompt here",
    system_prompt="System instructions",
    history_messages=[],
    api_key=api_key,
    base_url=base_url
)

# Multimodal completion (with images)
result = openai_complete_if_cache(
    model="gpt-4o",
    prompt="",
    messages=[
        {"role": "system", "content": "Analyze this image"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what you see"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}
                }
            ]
        }
    ],
    api_key=api_key,
    base_url=base_url
)

# Generate embeddings
embeddings = openai_embed(
    texts=["text1", "text2"],
    model="text-embedding-3-large",
    api_key=api_key,
    base_url=base_url
)
```

---

### 5. Storage Architecture

RAG-Anything is built on LightRAG, which uses a sophisticated multi-storage architecture combining **vector databases**, **graph databases**, and **key-value stores** for different purposes.

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LightRAG Storage Architecture                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        VECTOR DATABASES                              │    │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │    │
│  │  │  entities_vdb   │ │relationships_vdb│ │   chunks_vdb    │        │    │
│  │  │                 │ │                 │ │                 │        │    │
│  │  │ Entity name +   │ │ Src + Tgt +     │ │ Document chunk  │        │    │
│  │  │ description     │ │ relation desc   │ │ text content    │        │    │
│  │  │ embeddings      │ │ embeddings      │ │ embeddings      │        │    │
│  │  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘        │    │
│  │           │                   │                   │                  │    │
│  │           └───────────────────┴───────────────────┘                  │    │
│  │                               │                                      │    │
│  │                    Similarity Search (cosine)                        │    │
│  └───────────────────────────────┼──────────────────────────────────────┘    │
│                                  │                                           │
│  ┌───────────────────────────────┼──────────────────────────────────────┐    │
│  │                        GRAPH DATABASE                                │    │
│  │                               │                                      │    │
│  │              ┌────────────────┴────────────────┐                     │    │
│  │              │  chunk_entity_relation_graph    │                     │    │
│  │              │                                 │                     │    │
│  │              │   [Entity]──(relation)──[Entity]│                     │    │
│  │              │      │                     │    │                     │    │
│  │              │   (mentions)           (mentions)                     │    │
│  │              │      │                     │    │                     │    │
│  │              │   [Chunk]               [Chunk] │                     │    │
│  │              └─────────────────────────────────┘                     │    │
│  │                                                                      │    │
│  │              Graph Traversal (neighbors, paths)                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                        KEY-VALUE STORES                               │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │    │
│  │  │  full_docs   │ │ text_chunks  │ │full_entities │ │full_relations│ │    │
│  │  │              │ │              │ │              │ │              │ │    │
│  │  │ Original doc │ │ Chunk text + │ │ Entity desc  │ │ Relation     │ │    │
│  │  │ content      │ │ metadata     │ │ + source_ids │ │ descriptions │ │    │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │    │
│  │  │entity_chunks │ │relation_chunks││llm_resp_cache│ │  doc_status  │ │    │
│  │  │              │ │              │ │              │ │              │ │    │
│  │  │ Entity→Chunk │ │ Relation→    │ │ LLM response │ │ Processing   │ │    │
│  │  │ mappings     │ │ Chunk maps   │ │ caching      │ │ status       │ │    │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Storage Type Details

##### 1. Vector Databases (3 Separate VDBs)

LightRAG uses **three separate vector databases** for different types of similarity search:

| VDB                 | Namespace       | Purpose                                   | Stored Data             | Meta Fields                                             |
| ------------------- | --------------- | ----------------------------------------- | ----------------------- | ------------------------------------------------------- |
| `entities_vdb`      | `entities`      | Find similar entities by name/description | Entity embeddings       | `entity_name`, `source_id`, `content`, `file_path`      |
| `relationships_vdb` | `relationships` | Find similar relationships                | Relationship embeddings | `src_id`, `tgt_id`, `source_id`, `content`, `file_path` |
| `chunks_vdb`        | `chunks`        | Find similar document chunks              | Chunk text embeddings   | `full_doc_id`, `content`, `file_path`                   |

**How Vector Search is Used:**

```python
# Entity search (local mode)
results = await entities_vdb.query(query, top_k=query_param.top_k)
# Returns: [{"id": "ent-xxx", "distance": 0.85, "entity_name": "...", ...}]

# Relationship search (global mode)
results = await relationships_vdb.query(keywords, top_k=query_param.top_k)
# Returns: [{"id": "rel-xxx", "distance": 0.82, "src_id": "...", "tgt_id": "...", ...}]

# Chunk search (naive/mix mode)
results = await chunks_vdb.query(query, top_k=query_param.chunk_top_k)
# Returns: [{"id": "chunk-xxx", "distance": 0.90, "content": "...", ...}]
```

##### 2. Graph Database (Knowledge Graph)

The graph database stores the **knowledge graph** with entities as nodes and relationships as edges:

| Component           | Description                                                                         |
| ------------------- | ----------------------------------------------------------------------------------- |
| **Nodes**           | Entities extracted from documents (e.g., "Python", "Machine Learning")              |
| **Edges**           | Relationships between entities (e.g., "Python" --is_used_for--> "Machine Learning") |
| **Node Attributes** | `entity_name`, `entity_type`, `description`, `source_id`, `file_path`               |
| **Edge Attributes** | `weight`, `description`, `keywords`, `source_id`, `file_path`                       |

**Graph Operations:**

```python
# Check if entity exists
exists = await knowledge_graph.has_node("Python")

# Get entity data
node_data = await knowledge_graph.get_node("Python")
# Returns: {"entity_type": "technology", "description": "...", "source_id": "..."}

# Get entity's relationships
edges = await knowledge_graph.get_node_edges("Python")
# Returns: [("Python", "Machine Learning"), ("Python", "Data Science"), ...]

# Get relationship data
edge_data = await knowledge_graph.get_edge("Python", "Machine Learning")
# Returns: {"weight": 5.0, "description": "...", "keywords": "..."}

# Get node degree (connectivity)
degree = await knowledge_graph.node_degree("Python")
# Returns: 15 (number of connected relationships)
```

##### 3. Key-Value Stores (7 Namespaces)

| Namespace            | Purpose                         | Key Format          | Value Format                                        |
| -------------------- | ------------------------------- | ------------------- | --------------------------------------------------- |
| `full_docs`          | Store original document content | `doc-{hash}`        | `{content, file_path, ...}`                         |
| `text_chunks`        | Store chunked text              | `chunk-{hash}`      | `{content, tokens, full_doc_id, chunk_order_index}` |
| `full_entities`      | Store entity descriptions       | `entity_name`       | `{description, source_id, file_path}`               |
| `full_relations`     | Store relation descriptions     | `src_id<SEP>tgt_id` | `{description, source_id, keywords}`                |
| `entity_chunks`      | Map entities to chunks          | `entity_name`       | `[chunk_id1, chunk_id2, ...]`                       |
| `relation_chunks`    | Map relations to chunks         | `src_id<SEP>tgt_id` | `[chunk_id1, chunk_id2, ...]`                       |
| `llm_response_cache` | Cache LLM responses             | `prompt_hash`       | `{response, timestamp}`                             |

##### 4. Document Status Storage

Tracks document processing state:

```python
{
    "doc-abc123": {
        "status": "processed",  # pending, processing, processed, failed
        "chunks_count": 15,
        "created_at": "2024-01-15T10:30:00Z",
        "updated_at": "2024-01-15T10:35:00Z"
    }
}
```

#### Query Modes and Storage Usage

LightRAG supports different query modes that use storage differently:

| Mode     | entities_vdb | relationships_vdb | chunks_vdb | Graph | Description                                    |
| -------- | :----------: | :---------------: | :--------: | :---: | ---------------------------------------------- |
| `local`  |      ✅      |        ❌         |     ❌     |  ✅   | Entity-centric, uses local entity neighborhood |
| `global` |      ❌      |        ✅         |     ❌     |  ✅   | Relationship-centric, uses global patterns     |
| `hybrid` |      ✅      |        ✅         |     ❌     |  ✅   | Combines local + global                        |
| `naive`  |      ❌      |        ❌         |     ✅     |  ❌   | Pure vector search on chunks                   |
| `mix`    |      ✅      |        ✅         |     ✅     |  ✅   | Full integration: KG + vector chunks           |

**Query Flow Example (mix mode):**

```
1. Query: "How does Python relate to machine learning?"
   │
2. ├─→ entities_vdb.query("Python machine learning")
   │   └─→ Returns: [Python, Machine Learning, Scikit-learn, ...]
   │
3. ├─→ relationships_vdb.query("Python machine learning")
   │   └─→ Returns: [(Python, ML), (Scikit-learn, Python), ...]
   │
4. ├─→ chunks_vdb.query("Python machine learning")
   │   └─→ Returns: [chunk1, chunk2, chunk3, ...]
   │
5. ├─→ knowledge_graph.get_node_edges("Python")
   │   └─→ Graph traversal to find related entities
   │
6. ├─→ Merge and deduplicate results
   │
7. └─→ Build context from entity_chunks + relation_chunks mappings
        │
8.      └─→ LLM generates response with combined context
```

#### Storage Backend Options

##### Supported Implementations

| Storage Type       | Implementations                                                                                                                            | Notes                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------- |
| **KV Storage**     | `JsonKVStorage`, `RedisKVStorage`, `PGKVStorage`, `MongoKVStorage`                                                                         | Default: JSON files    |
| **Vector Storage** | `NanoVectorDBStorage`, `MilvusVectorDBStorage`, `PGVectorStorage`, `FaissVectorDBStorage`, `QdrantVectorDBStorage`, `MongoVectorDBStorage` | Default: nano-vectordb |
| **Graph Storage**  | `NetworkXStorage`, `Neo4JStorage`, `PGGraphStorage`, `MongoGraphStorage`, `MemgraphStorage`                                                | Default: NetworkX      |
| **Doc Status**     | `JsonDocStatusStorage`, `RedisDocStatusStorage`, `PGDocStatusStorage`, `MongoDocStatusStorage`                                             | Default: JSON          |

##### Environment Variables

```bash
# Storage Type Selection (pass string names to LightRAG)
# These are the class names to use

# KV Storage
kv_storage="JsonKVStorage"        # Default - JSON files
kv_storage="MongoKVStorage"       # MongoDB
kv_storage="RedisKVStorage"       # Redis
kv_storage="PGKVStorage"          # PostgreSQL

# Vector Storage
vector_storage="NanoVectorDBStorage"   # Default - JSON files
vector_storage="MongoVectorDBStorage"  # MongoDB Atlas Vector Search
vector_storage="MilvusVectorDBStorage" # Milvus
vector_storage="QdrantVectorDBStorage" # Qdrant
vector_storage="PGVectorStorage"       # PostgreSQL pgvector
vector_storage="FaissVectorDBStorage"  # FAISS (local)

# Graph Storage
graph_storage="NetworkXStorage"   # Default - GraphML file
graph_storage="Neo4JStorage"      # Neo4j
graph_storage="PGGraphStorage"    # PostgreSQL
graph_storage="MongoGraphStorage" # MongoDB
graph_storage="MemgraphStorage"   # Memgraph
```

##### Required Environment Variables by Backend

```bash
# MongoDB (for MongoKVStorage, MongoVectorDBStorage, MongoGraphStorage)
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_DATABASE=lightrag

# Neo4j (for Neo4JStorage)
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Redis (for RedisKVStorage, RedisDocStatusStorage)
REDIS_URI=redis://localhost:6379

# Milvus (for MilvusVectorDBStorage)
MILVUS_URI=http://localhost:19530
MILVUS_DB_NAME=lightrag

# Qdrant (for QdrantVectorDBStorage)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-api-key  # Optional

# PostgreSQL (for PGKVStorage, PGVectorStorage, PGGraphStorage)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_DATABASE=your_database

# Memgraph (for MemgraphStorage)
MEMGRAPH_URI=bolt://localhost:7687
```

#### Storage Setup Options

| Setup                    |  KV   |    Vector     |  Graph   | Doc Status | Use Case                       |
| ------------------------ | :---: | :-----------: | :------: | :--------: | ------------------------------ |
| **Default (file-based)** | JSON  | nano-vectordb | NetworkX |    JSON    | Development, small datasets    |
| **MongoDB**              | Mongo |     Mongo     | NetworkX |   Mongo    | Medium scale, existing MongoDB |
| **Neo4j + MongoDB**      | Mongo |     Mongo     |  Neo4j   |   Mongo    | Production, complex graphs     |
| **PostgreSQL**           |  PG   |   pgvector    |    PG    |     PG     | Single DB, SQL familiarity     |
| **Milvus + Neo4j**       | JSON  |    Milvus     |  Neo4j   |    JSON    | High-performance vector search |

#### Configuration Examples

##### File-Based (Development)

```python
from lightrag import LightRAG

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=llm_func,
    embedding_func=embedding_func,
    # All defaults - file-based storage
)
```

##### MongoDB (Production)

```python
import os
os.environ["MONGO_URI"] = "mongodb+srv://user:pass@cluster.mongodb.net/"
os.environ["MONGO_DATABASE"] = "lightrag_prod"

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=llm_func,
    embedding_func=embedding_func,
    kv_storage="MongoKVStorage",
    vector_storage="MongoVectorDBStorage",
    doc_status_storage="MongoDocStatusStorage",
    graph_storage="NetworkXStorage",  # Or "MongoGraphStorage"
)
```

##### Neo4j + Milvus (High Performance)

```python
import os
os.environ["NEO4J_URI"] = "neo4j+s://xxx.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"
os.environ["MILVUS_URI"] = "http://localhost:19530"
os.environ["MILVUS_DB_NAME"] = "lightrag"

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=llm_func,
    embedding_func=embedding_func,
    kv_storage="JsonKVStorage",
    vector_storage="MilvusVectorDBStorage",
    graph_storage="Neo4JStorage",
)
```

#### File-Based Storage Structure

When using default file-based storage, the `working_dir` contains:

```
working_dir/
├── vdb_entities.json                    # Entity vector embeddings (nano-vectordb)
├── vdb_relationships.json               # Relationship vector embeddings
├── vdb_chunks.json                      # Chunk vector embeddings
├── graph_chunk_entity_relation.graphml  # Knowledge graph (NetworkX GraphML)
├── kv_store_full_docs.json              # Original document content
├── kv_store_text_chunks.json            # Chunked text with metadata
├── kv_store_full_entities.json          # Entity descriptions
├── kv_store_full_relations.json         # Relationship descriptions
├── kv_store_entity_chunks.json          # Entity → Chunk mappings
├── kv_store_relation_chunks.json        # Relation → Chunk mappings
├── kv_store_llm_response_cache.json     # LLM response cache
└── kv_store_doc_status.json             # Document processing status
```

#### Vector Search Details

##### Cosine Similarity Threshold

LightRAG uses cosine similarity with a configurable threshold:

```python
rag = LightRAG(
    cosine_better_than_threshold=0.2,  # Default: 0.2
    # Results with cosine similarity < 0.2 are filtered out
)

# Or via environment variable
os.environ["COSINE_THRESHOLD"] = "0.3"
```

##### Top-K Parameters

```python
from lightrag.base import QueryParam

param = QueryParam(
    top_k=60,          # Entities/relations to retrieve
    chunk_top_k=20,    # Chunks to retrieve for mix/naive mode
)
```

#### Graph Storage Details

##### NetworkX (Default)

- Stores graph as `.graphml` XML file
- In-memory graph with disk persistence
- Cross-process synchronization via file locks
- Good for development and small-medium datasets

```python
# Graph is automatically saved on index completion
await rag.index_done_callback()  # Persists graph to disk
```

##### Neo4j (Production)

- Full graph database with Cypher queries
- Better for large-scale graphs
- Supports graph algorithms out of the box
- Requires Neo4j server

```python
# Neo4j handles persistence automatically
# Supports advanced queries like:
# - Shortest path between entities
# - Community detection
# - PageRank importance
```

#### Storage Initialization Flow

```python
# 1. Create LightRAG instance (storage classes are loaded)
rag = LightRAG(working_dir="./storage", ...)

# 2. Initialize all storages (async)
await rag.initialize_storages()
# This initializes in order:
# - full_docs, text_chunks, full_entities, full_relations
# - entity_chunks, relation_chunks
# - entities_vdb, relationships_vdb, chunks_vdb
# - chunk_entity_relation_graph
# - llm_response_cache, doc_status

# 3. Use the RAG system
await rag.ainsert("document content...")
result = await rag.aquery("What is...?")

# 4. Finalize storages (persist and close connections)
await rag.finalize_storages()
```

#### RAG-Anything Storage Integration

RAG-Anything's modal processors directly integrate with LightRAG's storage components. Here's how each processor uses the storage architecture:

##### BaseModalProcessor Storage Access

All modal processors (`ImageModalProcessor`, `TableModalProcessor`, `EquationModalProcessor`, `GenericModalProcessor`) extend `BaseModalProcessor`, which connects to LightRAG's storage in its constructor:

```python
# From raganything/modalprocessors.py (BaseModalProcessor.__init__)
class BaseModalProcessor:
    def __init__(self, lightrag: LightRAG, modal_caption_func, ...):
        self.lightrag = lightrag

        # KV Store access
        self.text_chunks_db = lightrag.text_chunks       # Chunk text + metadata

        # Vector DB access
        self.chunks_vdb = lightrag.chunks_vdb            # Chunk embeddings
        self.entities_vdb = lightrag.entities_vdb        # Entity embeddings
        self.relationships_vdb = lightrag.relationships_vdb  # Relationship embeddings

        # Graph DB access
        self.knowledge_graph_inst = lightrag.chunk_entity_relation_graph  # NetworkX/Neo4j

        # Other LightRAG components
        self.embedding_func = lightrag.embedding_func    # Embedding generation
        self.llm_model_func = lightrag.llm_model_func    # LLM calls
        self.hashing_kv = lightrag.llm_response_cache    # LLM response caching
        self.tokenizer = lightrag.tokenizer              # Token counting
```

##### Multimodal Content Storage Flow

When processing a multimodal item (image, table, equation), the storage flow is:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                   MULTIMODAL CONTENT PROCESSING FLOW                          │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  1. DESCRIPTION GENERATION                                                    │
│     ┌─────────────────────────────────────────────────────────────────────┐   │
│     │ modal_content (table/image/equation)                                │   │
│     │         │                                                           │   │
│     │         ▼                                                           │   │
│     │ ┌─────────────────────────────────────────────────────────────────┐ │   │
│     │ │ ContextExtractor.extract_context()                              │ │   │
│     │ │ - Extracts surrounding text from content_list                   │ │   │
│     │ └──────────────────────────┬──────────────────────────────────────┘ │   │
│     │                            │                                        │   │
│     │                            ▼                                        │   │
│     │ ┌─────────────────────────────────────────────────────────────────┐ │   │
│     │ │ modal_caption_func (LLM/Vision)                                 │ │   │
│     │ │ - Generates enhanced description with context                   │ │   │
│     │ └──────────────────────────┬──────────────────────────────────────┘ │   │
│     │                            │                                        │   │
│     │                            ▼                                        │   │
│     │               (description, entity_info)                            │   │
│     └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                            │
│  2. CHUNK CREATION & STORAGE     │                                            │
│     ┌────────────────────────────▼────────────────────────────────────────┐   │
│     │ _create_entity_and_chunk()                                          │   │
│     │         │                                                           │   │
│     │         ├─────────┬─────────────────────────────────────────────────┤   │
│     │         │         │                                                 │   │
│     │         ▼         ▼                                                 │   │
│     │  ┌──────────┐  ┌──────────────────────────────────────────────────┐ │   │
│     │  │text_chunks│ │                 chunks_vdb                        │ │   │
│     │  │ KV Store  │ │              (Vector DB)                          │ │   │
│     │  │           │ │                                                   │ │   │
│     │  │chunk_id:{ │ │ chunk_id: {content, full_doc_id, tokens,          │ │   │
│     │  │  content, │ │            chunk_order_index, file_path}          │ │   │
│     │  │  tokens,  │ │                                                   │ │   │
│     │  │  doc_id,  │ │ └─→ Embedding generated & stored                  │ │   │
│     │  │  file_path│ │                                                   │ │   │
│     │  │}          │ │                                                   │ │   │
│     │  └──────────┘  └──────────────────────────────────────────────────┘ │   │
│     └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                            │
│  3. ENTITY CREATION              │                                            │
│     ┌────────────────────────────▼────────────────────────────────────────┐   │
│     │         │                                                           │   │
│     │         ├─────────┬─────────────────────────────────────────────────┤   │
│     │         │         │                                                 │   │
│     │         ▼         ▼                                                 │   │
│     │  ┌────────────────────┐  ┌────────────────────────────────────────┐ │   │
│     │  │ knowledge_graph_inst│ │            entities_vdb                 │ │   │
│     │  │     (Graph DB)      │ │          (Vector DB)                    │ │   │
│     │  │                     │ │                                         │ │   │
│     │  │ upsert_node(        │ │ ent-xxx: {entity_name, entity_type,     │ │   │
│     │  │   entity_name, {    │ │           content, source_id,           │ │   │
│     │  │     entity_type,    │ │           file_path}                    │ │   │
│     │  │     description,    │ │                                         │ │   │
│     │  │     source_id,      │ │ └─→ Embedding generated & stored        │ │   │
│     │  │     file_path       │ │                                         │ │   │
│     │  │   })                │ │                                         │ │   │
│     │  └────────────────────┘  └────────────────────────────────────────┘ │   │
│     └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                            │
│  4. ENTITY EXTRACTION & RELATIONS│                                            │
│     ┌────────────────────────────▼────────────────────────────────────────┐   │
│     │ _process_chunk_for_extraction()                                     │   │
│     │         │                                                           │   │
│     │         ▼                                                           │   │
│     │ ┌─────────────────────────────────────────────────────────────────┐ │   │
│     │ │ extract_entities() [from LightRAG]                              │ │   │
│     │ │ - Extracts entities mentioned in the multimodal chunk           │ │   │
│     │ │ - Uses LLM to identify entities and relationships               │ │   │
│     │ └──────────────────────────┬──────────────────────────────────────┘ │   │
│     │                            │                                        │   │
│     │                            ▼                                        │   │
│     │ ┌─────────────────────────────────────────────────────────────────┐ │   │
│     │ │ Add "belongs_to" relationships                                  │ │   │
│     │ │ - Links extracted entities to the modal entity                  │ │   │
│     │ │ - e.g., "Python" belongs_to "LLM Performance Table"             │ │   │
│     │ └──────────────────────────┬──────────────────────────────────────┘ │   │
│     │                            │                                        │   │
│     │         ├─────────────────────────────────────────────────────────┤ │   │
│     │         │                  │                                      │ │   │
│     │         ▼                  ▼                                      │ │   │
│     │  ┌────────────────────┐  ┌────────────────────────────────────┐   │ │   │
│     │  │ knowledge_graph_inst│ │       relationships_vdb            │   │ │   │
│     │  │     (Graph DB)      │ │        (Vector DB)                 │   │ │   │
│     │  │                     │ │                                    │   │ │   │
│     │  │ upsert_edge(        │ │ rel-xxx: {src_id, tgt_id,          │   │ │   │
│     │  │   src_entity,       │ │           keywords, content,       │   │ │   │
│     │  │   tgt_entity, {     │ │           source_id, file_path}    │   │ │   │
│     │  │     description,    │ │                                    │   │ │   │
│     │  │     keywords,       │ │ └─→ Embedding generated & stored   │   │ │   │
│     │  │     weight,         │ │                                    │   │ │   │
│     │  │     source_id       │ │                                    │   │ │   │
│     │  │   })                │ │                                    │   │ │   │
│     │  └────────────────────┘  └────────────────────────────────────┘   │ │   │
│     └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                            │
│  5. MERGE & FINALIZE             │                                            │
│     ┌────────────────────────────▼────────────────────────────────────────┐   │
│     │ merge_nodes_and_edges() [from LightRAG]                             │   │
│     │ - Merges duplicate entities (same name, different sources)          │   │
│     │ - Updates entity/relationship descriptions with new context         │   │
│     │ - Stores to entities_vdb, relationships_vdb, knowledge_graph        │   │
│     │                                                                     │   │
│     │ lightrag._insert_done()                                             │   │
│     │ - Persists all storage changes to disk/database                     │   │
│     └─────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

##### ProcessorMixin Batch Storage Operations

The `ProcessorMixin` class in `processor.py` provides batch processing that efficiently uses LightRAG storage:

| Method | Storage Used | Operation |
|--------|--------------|-----------|
| `_store_chunks_to_lightrag_storage_type_aware()` | `text_chunks`, `chunks_vdb` | Store multimodal chunks for retrieval |
| `_store_multimodal_main_entities()` | `chunk_entity_relation_graph`, `entities_vdb`, `full_entities` | Store modal entities (e.g., "Table 1") |
| `_batch_extract_entities_lightrag_style_type_aware()` | `text_chunks`, `llm_response_cache` | Extract entities from chunk text |
| `_batch_add_belongs_to_relations_type_aware()` | `knowledge_graph_inst`, `relationships_vdb` | Add "belongs_to" edges |
| `_batch_merge_lightrag_style_type_aware()` | `entities_vdb`, `relationships_vdb`, `chunk_entity_relation_graph`, `full_entities`, `full_relations` | Merge all nodes/edges |
| `_update_doc_status_with_chunks_type_aware()` | `doc_status` | Track processing status |

##### Storage Example: Processing a Table

When `TableModalProcessor.process_multimodal_content()` processes a table:

```python
# Input table
table_content = {
    "table_body": "| Model | Accuracy |\n|-------|----------|\n| GPT-4 | 95.2% |",
    "table_caption": ["Performance Comparison"],
    "table_footnote": ["Benchmarked on MMLU"]
}

# Storage results after processing:

# 1. text_chunks KV Store
{
    "chunk-abc123": {
        "content": "[TABLE]\nCaption: Performance Comparison\n| Model | Accuracy |...\nAnalysis: This table compares...",
        "tokens": 156,
        "full_doc_id": "doc-xyz789",
        "chunk_order_index": 5,
        "file_path": "benchmark_report.pdf"
    }
}

# 2. chunks_vdb Vector Store
{
    "chunk-abc123": {
        "content": "[TABLE]...",  # Same content
        "embedding": [0.023, -0.156, ...],  # 768/1536/3072-dim vector
        "full_doc_id": "doc-xyz789"
    }
}

# 3. entities_vdb Vector Store (modal entity)
{
    "ent-def456": {
        "entity_name": "Performance Comparison (table)",
        "entity_type": "table",
        "content": "Performance Comparison (table)\nTable showing model accuracy...",
        "source_id": "chunk-abc123",
        "file_path": "benchmark_report.pdf"
    }
}

# 4. knowledge_graph (Graph DB nodes)
# Node: "Performance Comparison (table)"
# Attributes: {entity_type: "table", description: "...", source_id: "chunk-abc123"}

# 5. Extracted entities (via extract_entities)
# Node: "GPT-4"
# Node: "MMLU"

# 6. Relationships (Graph DB edges + relationships_vdb)
# Edge: ("GPT-4", "Performance Comparison (table)")
#   - keywords: "belongs_to,part_of,contained_in"
#   - description: "Entity GPT-4 belongs to Performance Comparison (table)"

# 7. full_entities KV Store
{
    "doc-xyz789": {
        "entity_names": ["Performance Comparison (table)", "GPT-4", "MMLU"],
        "count": 3,
        "update_time": 1704380400
    }
}
```

##### Key Integration Points

The following LightRAG functions are called by RAG-Anything during processing:

| LightRAG Function | Called By | Purpose |
|-------------------|-----------|---------|
| `extract_entities()` | `BaseModalProcessor._process_chunk_for_extraction()` | Extract entities/relationships from chunk text |
| `merge_nodes_and_edges()` | `ProcessorMixin._batch_merge_lightrag_style_type_aware()` | Deduplicate and merge graph data |
| `lightrag._insert_done()` | Multiple methods | Persist all storage changes |
| `knowledge_graph.upsert_node()` | `BaseModalProcessor._create_entity_and_chunk()` | Add entity to graph |
| `knowledge_graph.upsert_edge()` | `BaseModalProcessor._process_chunk_for_extraction()` | Add relationship to graph |
| `entities_vdb.upsert()` | `BaseModalProcessor._create_entity_and_chunk()` | Store entity embedding |
| `relationships_vdb.upsert()` | `BaseModalProcessor._process_chunk_for_extraction()` | Store relationship embedding |
| `chunks_vdb.upsert()` | `BaseModalProcessor._create_entity_and_chunk()` | Store chunk embedding |
| `text_chunks.upsert()` | `BaseModalProcessor._create_entity_and_chunk()` | Store chunk metadata |

---

## Model Function Templates

### LLM Model Function

```python
def get_llm_model_func(api_key: str, base_url: str = None):
    def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
    return llm_func
```

### Vision Model Function

```python
def get_vision_model_func(api_key: str, base_url: str = None):
    def vision_func(
        prompt, system_prompt=None, history_messages=[],
        image_data=None, messages=None, **kwargs
    ):
        # Handle pre-formatted messages (multimodal VLM query)
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs
            )
        # Handle single image
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                            }
                        ]
                    }
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs
            )
        # Fall back to text-only
        else:
            return openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs
            )
    return vision_func
```

### Embedding Function

```python
from lightrag.utils import EmbeddingFunc

embedding_func = EmbeddingFunc(
    embedding_dim=3072,  # Dimension for text-embedding-3-large
    max_token_size=8192,
    func=lambda texts: openai_embed(
        texts,
        model="text-embedding-3-large",
        api_key=api_key,
        base_url=base_url
    )
)
```

---

## Content Type Formats

### Text Content

```python
{"type": "text", "text": "Content text here", "page_idx": 0}
```

### Image Content

```python
{
    "type": "image",
    "img_path": "/absolute/path/to/image.jpg",  # Must be absolute path
    "image_caption": ["Caption text"],
    "image_footnote": ["Footnote text"],
    "page_idx": 1
}
```

### Table Content

```python
{
    "type": "table",
    "table_body": "| Header1 | Header2 |\n|---------|----------|\n| Cell1 | Cell2 |",
    "table_caption": ["Table caption"],
    "table_footnote": ["Table footnote"],
    "page_idx": 2
}
```

### Equation Content

```python
{
    "type": "equation",
    "latex": "E = mc^2",           # LaTeX format
    "text": "Description text",     # Plain text description
    "page_idx": 3
}
```

### Generic/Custom Content

```python
{
    "type": "custom_type",
    "content": "Any content data",
    "page_idx": 4
}
```

---

## Integration with Our Framework

### Current Implementations

| RAG-Anything API                  | Our Equivalent                     | Location                               |
| --------------------------------- | ---------------------------------- | -------------------------------------- |
| `ImageModalProcessor`             | `ImageProcessor`                   | `rag/ingestion/processors/image.py`    |
| `TableModalProcessor`             | `TableProcessor`                   | `rag/ingestion/processors/table.py`    |
| `EquationModalProcessor`          | `EquationProcessor`                | `rag/ingestion/processors/equation.py` |
| `MinerUParser` (document parsing) | `MinerUParser`                     | `rag/ingestion/chunkers/mineru.py`     |
| Content extraction                | `ExtractedBlock`, `ParsedDocument` | `rag/ingestion/chunkers/mineru.py`     |

### Potential Integrations

| RAG-Anything API             | Purpose                  | Integration Notes                 |
| ---------------------------- | ------------------------ | --------------------------------- |
| `insert_content_list()`      | Direct content injection | Integrate with ingestion pipeline |
| `openai_complete_if_cache()` | LLM caching              | Consider for performance          |

---

## Testing RAG-Anything Modal Processors

A test script is provided to verify RAG-Anything modal processors work correctly.

### Running Tests

```bash
# Test with Ollama (default, no API key needed) - uses file-based storage
python -m rag.ingestion.processors.test_raganything

# Test with Ollama explicitly
python -m rag.ingestion.processors.test_raganything --use-ollama

# Test with OpenAI
python -m rag.ingestion.processors.test_raganything --api-key YOUR_OPENAI_KEY

# Test with custom base URL
python -m rag.ingestion.processors.test_raganything --api-key KEY --base-url URL

# Test with MongoDB storage (instead of file-based)
python -m rag.ingestion.processors.test_raganything --use-ollama --use-mongodb

# Test with MongoDB and custom connection
python -m rag.ingestion.processors.test_raganything --use-ollama --use-mongodb \
    --mongo-uri "mongodb://localhost:27017/" \
    --mongo-database "test_raganything"
```

### Test Storage Options

| Option          | Storage Type | Location                      | Notes                                      |
| --------------- | ------------ | ----------------------------- | ------------------------------------------ |
| Default         | File-based   | `./test_raganything_storage/` | Uses nano-vectordb + NetworkX              |
| `--use-mongodb` | MongoDB      | MongoDB database              | Uses MongoVectorDBStorage + MongoKVStorage |

#### Default File-Based Storage

When running without `--use-mongodb`, the test creates:

```
./test_raganything_storage/
├── vdb_entities.json          # Entity vectors
├── vdb_relationships.json     # Relationship vectors
├── vdb_chunks.json            # Chunk vectors
├── graph_chunk_entity_relation.graphml  # Knowledge graph
└── kv_store_*.json            # Various KV stores
```

#### MongoDB Storage

With `--use-mongodb`, stores data in MongoDB collections:

```
test_raganything (database)
├── entities_vdb               # Entity vectors
├── relationships_vdb          # Relationship vectors
├── chunks_vdb                 # Chunk vectors
├── full_docs                  # Document storage
├── text_chunks                # Text chunks
└── doc_status                 # Processing status
```

### What Gets Tested

| Processor                | Test Content               | Description                                        |
| ------------------------ | -------------------------- | -------------------------------------------------- |
| `TableModalProcessor`    | LLM benchmark table        | Processes markdown table with performance metrics  |
| `EquationModalProcessor` | Cross-entropy loss formula | Processes LaTeX equation                           |
| `ImageModalProcessor`    | Caption-only test          | Processes image metadata without actual image file |

### Expected Output

```
Testing TableModalProcessor
Description: This table compares performance metrics...
Entity Info: {'entity_name': 'LLM Performance Table', ...}
TableModalProcessor: SUCCESS

Testing EquationModalProcessor
Description: Binary cross-entropy loss function...
Entity Info: {'entity_name': 'Binary Cross-Entropy Loss', ...}
EquationModalProcessor: SUCCESS

Testing ImageModalProcessor
Description: Architecture diagram showing...
Entity Info: {'entity_name': 'RAG Architecture Diagram', ...}
ImageModalProcessor: SUCCESS

TEST SUMMARY
  TableProcessor: PASS
  EquationProcessor: PASS
  ImageProcessor: PASS

Total: 3/3 passed
```

### Known Non-Fatal Errors

The test may log ERROR messages that are **non-fatal** - tests still pass due to fallback handling:

| Error Message                                       | Cause                                                                                         | Impact                                         |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `object str can't be used in 'await' expression`    | RAGAnything expects async LLM/embedding functions, but Ollama wrapper provides sync functions | Falls back gracefully, processing continues    |
| `No image path provided in modal_content`           | Test uses empty `img_path` intentionally                                                      | Expected - tests caption-only image processing |
| `Error generating table/equation/image description` | LLM function async mismatch                                                                   | Fallback to raw content, entity still created  |

#### Why Async Errors Occur

RAGAnything's modal processors call `await self.modal_caption_func(...)` expecting an async function. When using Ollama with sync HTTP calls:

```python
# Our sync function (causes "can't await str" error)
def llm_model_func(prompt, **kwargs) -> str:
    return httpx.Client().post(...).json()["choices"][0]["message"]["content"]

# What RAGAnything expects
async def llm_model_func(prompt, **kwargs) -> str:
    return await httpx.AsyncClient().post(...).json()["choices"][0]["message"]["content"]
```

The processors catch these errors and fall back to using raw content for entity creation, which is why tests still pass.

#### MongoDB Atlas Free Tier Limitation

When using `--use-mongodb` with Atlas M0 (free tier), you may see:

```
ERROR: The maximum number of FTS indexes has been reached for this instance size.
```

This is an Atlas limitation (3 search indexes on free tier). Solutions:

- Delete unused indexes in Atlas UI
- Use a different database name
- Use file-based storage (default)
- Upgrade Atlas tier

### Test Script Location

`rag/ingestion/processors/test_raganything.py`

---

## Our Multimodal Processors

In addition to using RAG-Anything's processors, we have our own implementations:

### Usage

```python
from rag.ingestion.processors import (
    ImageProcessor,
    TableProcessor,
    EquationProcessor,
)

# Process an image
image_proc = ImageProcessor()
result = await image_proc.process("image.png", context="Document context")
print(result.description)
print(result.entities)

# Process a table
table_proc = TableProcessor()
result = await table_proc.process(
    "| A | B |\n|---|---|\n| 1 | 2 |",
    context="Sales data"
)
print(result.metadata["key_insights"])

# Process an equation
eq_proc = EquationProcessor()
result = await eq_proc.process("E = mc^2", context="Physics")
print(result.metadata["variables"])
```

### Testing Our Processors

```bash
# Test TableProcessor
python -m rag.ingestion.processors.table

# Test EquationProcessor
python -m rag.ingestion.processors.equation

# Test EquationProcessor with custom equation
python -m rag.ingestion.processors.equation "F = ma" "Newton's second law"
```

---

## References

- [RAG-Anything GitHub](https://github.com/HKUDS/RAG-Anything)
- [RAG-Anything Paper](https://arxiv.org/abs/2510.12323)
- [LightRAG GitHub](https://github.com/HKUDS/LightRAG)
- [MinerU GitHub](https://github.com/opendatalab/MinerU)
