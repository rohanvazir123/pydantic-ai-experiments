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

### 3. LightRAG Integration

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

| RAG-Anything API | Our Equivalent | Location |
|------------------|----------------|----------|
| `ImageModalProcessor` | `ImageProcessor` | `rag/ingestion/processors/image.py` |
| `TableModalProcessor` | `TableProcessor` | `rag/ingestion/processors/table.py` |
| `EquationModalProcessor` | `EquationProcessor` | `rag/ingestion/processors/equation.py` |
| `MinerUParser` (document parsing) | `MinerUParser` | `rag/ingestion/chunkers/mineru.py` |
| Content extraction | `ExtractedBlock`, `ParsedDocument` | `rag/ingestion/chunkers/mineru.py` |

### Potential Integrations

| RAG-Anything API | Purpose | Integration Notes |
|------------------|---------|-------------------|
| `insert_content_list()` | Direct content injection | Integrate with ingestion pipeline |
| `openai_complete_if_cache()` | LLM caching | Consider for performance |

---

## References

- [RAG-Anything GitHub](https://github.com/HKUDS/RAG-Anything)
- [RAG-Anything Paper](https://arxiv.org/abs/2510.12323)
- [LightRAG GitHub](https://github.com/HKUDS/LightRAG)
- [MinerU GitHub](https://github.com/opendatalab/MinerU)
