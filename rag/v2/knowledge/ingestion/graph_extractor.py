"""docling-graph knowledge graph extractor.

Runs docling-graph run_pipeline() inside asyncio.to_thread (sync LLM I/O).
Returns PipelineContext (with .knowledge_graph NetworkX DiGraph) or None on failure.

IMPORTANT: Do NOT use CypherExporter.
CypherExporter generates Neo4j syntax incompatible with AGE's SQL wrapper.
The caller (pipeline.py) calls age_store.import_docling_graph(context, ...)
which iterates the NetworkX DiGraph directly.

Soft failure: on timeout or any exception, returns None and sets
chunk_metadata["graph_extraction_failed"] = True. The vector path continues.
Job is NOT moved to DLQ on graph extraction failure.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from knowledge.config.settings import Settings, load_settings
from knowledge.corpus.ontologies.loader import load_ontology

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


async def extract_graph(
    doc_path: Path,
    corpus_config: Any,     # CorpusConfig
    settings: Settings | None = None,
) -> Any | None:            # docling_graph.pipeline.context.PipelineContext | None
    """Run docling-graph extraction on a document.

    Returns PipelineContext with .knowledge_graph (NetworkX DiGraph), or None.

    Caller must check corpus_config.enable_graph_extraction before calling;
    this function does NOT guard against that — it is an assertion error to
    call extract_graph when extraction is disabled.
    """
    _settings = settings or load_settings()

    def _run_sync() -> Any:
        from docling_graph import PipelineConfig, run_pipeline

        ontology_class = load_ontology(corpus_config.graph_ontology_path)

        config = PipelineConfig(
            source=str(doc_path),
            template=ontology_class,
            backend=corpus_config.graph_extraction_backend,           # "llm" | "vlm"
            inference="local",
            provider_override=corpus_config.graph_extraction_provider, # "ollama"
            model_override=corpus_config.graph_extraction_model,
            processing_mode=corpus_config.graph_processing_mode,       # "many-to-one"
            extraction_contract=corpus_config.graph_extraction_contract, # "staged"
            use_chunking=True,
            chunk_max_tokens=_settings.chunk_max_tokens,
            structured_output=True,
            dump_to_disk=False,    # API mode — no files on disk
        )
        return run_pipeline(config)   # returns PipelineContext

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_run_sync),
            timeout=_settings.graph_extraction_timeout_s,
        )
    except TimeoutError:
        logger.warning(
            "Graph extraction timed out for '%s' (%.0fs limit)",
            doc_path.name, _settings.graph_extraction_timeout_s,
        )
        return None
    except ImportError:
        logger.warning(
            "docling-graph not installed — skipping graph extraction for '%s'",
            doc_path.name,
        )
        return None
    except Exception as exc:
        logger.error("Graph extraction failed for '%s': %s", doc_path.name, exc)
        return None
