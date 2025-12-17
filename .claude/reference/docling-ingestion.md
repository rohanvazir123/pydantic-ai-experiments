# Docling Ingestion Patterns

Reference guide for document processing and chunking with Docling.

## Document Conversion

### Basic Conversion

```python
from docling.document_converter import DocumentConverter

# Initialize converter (reuse for multiple documents)
converter = DocumentConverter()

# Convert document (supports PDF, DOCX, PPTX, XLSX, HTML, etc.)
result = converter.convert(file_path)

# Get markdown representation
markdown_content = result.document.export_to_markdown()

# Keep DoclingDocument for HybridChunker
docling_doc = result.document
```

**Supported Formats:**
- **Documents**: PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS
- **Web**: HTML, HTM
- **Text**: MD, TXT
- **Audio**: MP3, WAV, M4A, FLAC (via Whisper ASR)

### Error Handling

```python
import os

def convert_document_safe(file_path: str):
    """Convert document with error handling."""
    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        markdown = result.document.export_to_markdown()
        return markdown, result.document
    except Exception as e:
        logger.exception(
            "document_conversion_failed",
            file=file_path,
            format=os.path.splitext(file_path)[1],
            error=str(e)
        )
        # Return None to allow pipeline to continue
        return None, None
```

## HybridChunker

### Basic Setup

```python
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

# Initialize tokenizer (reuse across documents)
model_id = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create chunker
chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=512,      # Match embedding model limits
    merge_peers=True     # Merge small adjacent chunks
)
```

**Why HybridChunker?**
- **Token-aware**: Uses actual tokenizer, not character estimates
- **Structure-preserving**: Respects paragraphs, sections, tables
- **Contextualized**: Includes heading hierarchy in each chunk
- **Battle-tested**: Maintained by Docling team

### Chunking Documents

```python
from typing import List

async def chunk_document(
    content: str,
    title: str,
    source: str,
    metadata: dict,
    docling_doc,  # DoclingDocument from converter
    chunker: HybridChunker
) -> List[DocumentChunk]:
    """
    Chunk a document using Docling's HybridChunker.

    Args:
        content: Document content (markdown format)
        title: Document title
        source: Document source path
        metadata: Additional metadata
        docling_doc: DoclingDocument from converter (REQUIRED)
        chunker: HybridChunker instance

    Returns:
        List of document chunks with contextualized content
    """
    if not docling_doc:
        logger.warning("no_docling_doc", title=title)
        return []

    base_metadata = {
        "title": title,
        "source": source,
        "chunk_method": "hybrid",
        **metadata
    }

    try:
        # Chunk the DoclingDocument
        chunk_iter = chunker.chunk(dl_doc=docling_doc)
        chunks = list(chunk_iter)

        # Convert to DocumentChunk objects
        document_chunks = []
        current_pos = 0

        for i, chunk in enumerate(chunks):
            # Get contextualized text (includes heading hierarchy)
            contextualized_text = chunker.contextualize(chunk=chunk)

            # Count actual tokens
            token_count = len(tokenizer.encode(contextualized_text))

            # Create chunk metadata
            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "token_count": token_count,
                "has_context": True
            }

            # Estimate character positions
            start_char = current_pos
            end_char = start_char + len(contextualized_text)

            document_chunks.append(DocumentChunk(
                content=contextualized_text.strip(),
                index=i,
                start_char=start_char,
                end_char=end_char,
                metadata=chunk_metadata,
                token_count=token_count
            ))

            current_pos = end_char

        logger.info(
            "chunks_created",
            title=title,
            count=len(document_chunks),
            avg_tokens=sum(c.token_count for c in document_chunks) // len(document_chunks)
        )
        return document_chunks

    except Exception as e:
        logger.exception("chunking_failed", title=title)
        return []
```

### Contextualization Example

**Without context:**
```
The system uses a distributed architecture for scalability.
```

**With context (from HybridChunker):**
```
# Technical Architecture

## System Design

### Distributed Architecture

The system uses a distributed architecture for scalability.
```

The heading hierarchy provides crucial context for retrieval!

### Configuration Options

```python
from docling.chunking import HybridChunker

# Conservative chunking (smaller, more chunks)
chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=256,      # Smaller chunks
    merge_peers=False    # Don't merge
)

# Aggressive chunking (larger, fewer chunks)
chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=1024,     # Larger chunks
    merge_peers=True     # Merge small chunks
)

# Recommended for RAG
chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=512,      # Fits most embedding models
    merge_peers=True     # Balance chunk count
)
```

## Audio Transcription

### Whisper ASR Integration

```python
from pathlib import Path
from docling.document_converter import DocumentConverter, AudioFormatOption
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import InputFormat
from docling.pipeline.asr_pipeline import AsrPipeline

def setup_audio_converter():
    """Create DocumentConverter configured for audio transcription."""

    # Configure Whisper Turbo model
    pipeline_options = AsrPipelineOptions()
    pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

    # Create converter with audio support
    converter = DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    return converter

def transcribe_audio(file_path: str) -> tuple[str, object]:
    """
    Transcribe audio file using Whisper ASR.

    Args:
        file_path: Path to audio file (MP3, WAV, M4A, FLAC)

    Returns:
        Tuple of (markdown_transcript, docling_document)
    """
    try:
        converter = setup_audio_converter()

        # IMPORTANT: Pass Path object, not string
        audio_path = Path(file_path).resolve()

        logger.info("audio_transcription_started", file=audio_path.name)

        # Transcribe
        result = converter.convert(audio_path)

        # Export to markdown (includes timestamps)
        transcript = result.document.export_to_markdown()

        logger.info(
            "audio_transcription_completed",
            file=audio_path.name,
            length=len(transcript)
        )

        return transcript, result.document

    except Exception as e:
        logger.exception(
            "audio_transcription_failed",
            file=file_path,
            error=str(e)
        )
        return None, None
```

### Supported Audio Formats

- **MP3**: Most common, lossy compression
- **WAV**: Uncompressed, high quality
- **M4A**: Apple's AAC format
- **FLAC**: Lossless compression

### Whisper Model Options

```python
# Fast, less accurate
pipeline_options.asr_options = asr_model_specs.WHISPER_TINY

# Balanced (recommended)
pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

# Slower, more accurate
pipeline_options.asr_options = asr_model_specs.WHISPER_LARGE
```

## Multi-Format Ingestion Pipeline

### Complete Pipeline

```python
import os
import glob
from typing import List
from pathlib import Path

class DocumentIngestionPipeline:
    """Pipeline for ingesting multi-format documents."""

    def __init__(self, chunker, embedder):
        self.chunker = chunker
        self.embedder = embedder
        self.converter = DocumentConverter()
        self.audio_converter = setup_audio_converter()

    def find_documents(self, folder: str) -> List[str]:
        """Find all supported documents in folder."""
        patterns = [
            "*.md", "*.markdown", "*.txt",  # Text
            "*.pdf",                         # PDF
            "*.docx", "*.doc",              # Word
            "*.pptx", "*.ppt",              # PowerPoint
            "*.xlsx", "*.xls",              # Excel
            "*.html", "*.htm",              # HTML
            "*.mp3", "*.wav", "*.m4a", "*.flac",  # Audio
        ]

        files = []
        for pattern in patterns:
            files.extend(
                glob.glob(os.path.join(folder, "**", pattern), recursive=True)
            )

        return sorted(files)

    async def process_document(self, file_path: str):
        """Process a single document."""
        file_ext = os.path.splitext(file_path)[1].lower()

        # Audio formats
        if file_ext in ['.mp3', '.wav', '.m4a', '.flac']:
            content, docling_doc = transcribe_audio(file_path)
        # Other formats
        else:
            content, docling_doc = convert_document_safe(file_path)

        if not content or not docling_doc:
            logger.warning("document_processing_skipped", file=file_path)
            return None

        # Extract title
        title = self._extract_title(content, file_path)

        # Chunk document
        chunks = await self.chunker.chunk_document(
            content=content,
            title=title,
            source=os.path.relpath(file_path),
            metadata={"file_size": len(content)},
            docling_doc=docling_doc
        )

        if not chunks:
            logger.warning("no_chunks_created", file=file_path)
            return None

        # Generate embeddings
        embedded_chunks = await self.embedder.embed_chunks(chunks)

        return {
            "title": title,
            "source": os.path.relpath(file_path),
            "content": content,
            "chunks": embedded_chunks
        }

    def _extract_title(self, content: str, file_path: str) -> str:
        """Extract title from content or filename."""
        # Try to find markdown title
        lines = content.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()

        # Fallback to filename
        return os.path.splitext(os.path.basename(file_path))[0]
```

## Metadata Extraction

### From Frontmatter

```python
def extract_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from markdown."""
    if not content.startswith('---'):
        return {}

    try:
        import yaml

        end_marker = content.find('\n---\n', 4)
        if end_marker == -1:
            return {}

        frontmatter = content[4:end_marker]
        metadata = yaml.safe_load(frontmatter)

        return metadata if isinstance(metadata, dict) else {}

    except Exception as e:
        logger.warning("frontmatter_extraction_failed", error=str(e))
        return {}
```

### From Document Properties

```python
def extract_document_metadata(content: str, file_path: str) -> dict:
    """Extract metadata from document."""
    metadata = {
        "file_path": file_path,
        "file_size": len(content),
        "line_count": len(content.split('\n')),
        "word_count": len(content.split()),
        "ingestion_date": datetime.now().isoformat()
    }

    # Try frontmatter
    frontmatter = extract_frontmatter(content)
    metadata.update(frontmatter)

    return metadata
```

## Performance Considerations

### Batch Processing

```python
async def process_documents_batch(
    file_paths: List[str],
    batch_size: int = 10
):
    """Process documents in batches for better performance."""

    results = []

    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]

        # Process batch concurrently
        tasks = [process_document(path) for path in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        for result in batch_results:
            if not isinstance(result, Exception) and result:
                results.append(result)

        logger.info(
            "batch_processed",
            batch_num=i // batch_size + 1,
            total_batches=(len(file_paths) + batch_size - 1) // batch_size
        )

    return results
```

### Memory Management

```python
# ❌ BAD - Loads all documents into memory
all_docs = [process_document(f) for f in files]

# ✅ GOOD - Process and save incrementally
for file_path in files:
    doc_data = await process_document(file_path)
    if doc_data:
        await save_to_mongodb(doc_data)
        # doc_data goes out of scope, memory freed
```

## Common Issues

### Issue 1: "DoclingDocument not found"

```python
# ❌ WRONG - Lost DoclingDocument
markdown = result.document.export_to_markdown()
chunks = chunker.chunk(dl_doc=markdown)  # Wrong! markdown is string

# ✅ CORRECT - Keep DoclingDocument
markdown = result.document.export_to_markdown()
docling_doc = result.document  # Keep this!
chunks = chunker.chunk(dl_doc=docling_doc)
```

### Issue 2: "Chunks too large"

```python
# Reduce max_tokens in chunker
chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=256,  # Smaller chunks
    merge_peers=False
)
```

### Issue 3: "Audio transcription fails"

```python
# MUST use Path object, not string
audio_path = Path(file_path).resolve()  # ✅ Correct
result = converter.convert(audio_path)

# NOT this:
result = converter.convert(file_path)  # ❌ Wrong
```
