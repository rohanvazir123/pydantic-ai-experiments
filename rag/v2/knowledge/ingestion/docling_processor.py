"""Docling document converter — two converters, one per format family.

Architecture anchor: mirrors rag/ingestion/pipeline.py._get_pdf_converter()
and ._get_standard_converter() exactly.

Two converters (lazy-initialised, cached per worker):
  PDF        → _get_pdf_converter()      VLM-optional (Qwen2.5-VL via Ollama)
  DOCX/MD/… → _get_standard_converter() text-layer only, no VLM, no OCR

Format routing:
  .pdf                            → PDF converter
  .docx .doc .pptx .ppt .xlsx
  .xls .html .htm .md .markdown   → standard converter
  .mp3 .wav .m4a .flac            → audio ASR pipeline
  everything else                 → plain text read

All conversion runs in asyncio.to_thread (CPU-bound, sync Docling API).
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from knowledge.config.settings import Settings, load_settings
from knowledge.ingestion.models import ConversionResult

logger = logging.getLogger(__name__)

_PDF_FORMATS       = frozenset({".pdf"})
_STRUCTURED_FORMATS = frozenset({
    ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".html", ".htm", ".md", ".markdown",
})
_AUDIO_FORMATS     = frozenset({".mp3", ".wav", ".m4a", ".flac"})


class DoclingProcessor:
    """Wraps Docling DocumentConverter with lazy init and format routing.

    One instance per worker process — converters are expensive to initialise
    and are cached as instance attributes after first use.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_settings()
        self._pdf_converter: Any = None
        self._standard_converter: Any = None

    def _get_pdf_converter(self) -> Any:
        if self._pdf_converter is not None:
            return self._pdf_converter

        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption

        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_ocr = False
        pdf_opts.do_table_structure = True
        pdf_opts.do_picture_classification = False

        if self._settings.vlm_enabled:
            from docling.datamodel.pipeline_options import PictureDescriptionApiOptions
            pdf_opts.do_picture_description = True
            pdf_opts.generate_picture_images = True
            pdf_opts.enable_remote_services = True
            pdf_opts.picture_description_options = PictureDescriptionApiOptions(
                url=self._settings.vlm_base_url,
                params={
                    "model": self._settings.vlm_model,
                    "temperature": 0.1,
                },
                prompt=(
                    "Describe this image in detail, explicitly capturing any "
                    "charts, tables, diagrams, or structural data shown."
                ),
                timeout=self._settings.vlm_timeout,
                concurrency=self._settings.vlm_concurrency,
            )
            logger.info("PDF converter: standard pipeline + VLM (model=%s)", self._settings.vlm_model)
        else:
            pdf_opts.do_picture_description = False
            pdf_opts.generate_picture_images = False
            pdf_opts.generate_page_images = False
            logger.info("PDF converter: standard pipeline (text-only, no VLM)")

        self._pdf_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
        )
        return self._pdf_converter

    def _get_standard_converter(self) -> Any:
        if self._standard_converter is not None:
            return self._standard_converter

        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption

        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_ocr = False
        pdf_opts.do_picture_classification = False
        pdf_opts.do_picture_description = False
        pdf_opts.generate_picture_images = False
        pdf_opts.generate_page_images = False

        self._standard_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
        )
        logger.info("Standard converter initialised (DOCX/PPTX/XLSX/HTML/MD)")
        return self._standard_converter

    def _convert_with_docling(self, path: Path, converter: Any) -> ConversionResult:
        """Synchronous Docling conversion — must be called inside asyncio.to_thread."""
        try:
            result = converter.convert(str(path))
            markdown = result.document.export_to_markdown()
            logger.info("Converted '%s' → %d chars", path.name, len(markdown))
            return ConversionResult(
                markdown=markdown,
                docling_doc=result.document,
                format=path.suffix.lstrip(".").lower(),
            )
        except Exception as exc:
            logger.error("Docling conversion failed for '%s': %s", path.name, exc)
            # Try plain text fallback
            try:
                text = path.read_text(encoding="utf-8")
                return ConversionResult(markdown=text, docling_doc=None, format="text")
            except Exception:
                return ConversionResult(
                    markdown=f"[Error: Could not read file {path.name}]",
                    docling_doc=None,
                    format="error",
                )

    def _convert_audio(self, path: Path) -> ConversionResult:
        """Synchronous Whisper ASR via Docling — must be called inside asyncio.to_thread."""
        try:
            from docling.datamodel import asr_model_specs
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import AsrPipelineOptions
            from docling.document_converter import AudioFormatOption, DocumentConverter
            from docling.pipeline.asr_pipeline import AsrPipeline

            pipeline_opts = AsrPipelineOptions()
            pipeline_opts.asr_options = asr_model_specs.WHISPER_TURBO

            converter = DocumentConverter(
                format_options={
                    InputFormat.AUDIO: AudioFormatOption(
                        pipeline_cls=AsrPipeline,
                        pipeline_options=pipeline_opts,
                    )
                }
            )
            result = converter.convert(str(path))
            markdown = result.document.export_to_markdown()
            logger.info("Transcribed '%s' → %d chars", path.name, len(markdown))
            return ConversionResult(markdown=markdown, docling_doc=result.document, format="audio")

        except Exception as exc:
            logger.error("Audio transcription failed for '%s': %s", path.name, exc)
            return ConversionResult(
                markdown=f"[Error: Could not transcribe audio file {path.name}]",
                docling_doc=None,
                format="audio_error",
            )

    def _read_plain_text(self, path: Path) -> ConversionResult:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        return ConversionResult(markdown=text, docling_doc=None, format="text")

    def _process_sync(self, path: Path) -> ConversionResult:
        """Synchronous dispatch — route by extension, call converter."""
        ext = path.suffix.lower()
        if ext in _AUDIO_FORMATS:
            return self._convert_audio(path)
        if ext in _PDF_FORMATS:
            return self._convert_with_docling(path, self._get_pdf_converter())
        if ext in _STRUCTURED_FORMATS:
            return self._convert_with_docling(path, self._get_standard_converter())
        return self._read_plain_text(path)

    async def process(self, path: Path) -> ConversionResult:
        """Convert a document. Offloads CPU-bound Docling work to threadpool."""
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        return await asyncio.to_thread(self._process_sync, path)
