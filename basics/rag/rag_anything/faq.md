# FAQ — RAG-Anything

## Will RAG-Anything do better than Docling on research papers?

Partially — it depends on the failure type.

**Where RAG-Anything does better:**

- **Figures** — dedicated modal processors per content type. Figures go through a VLM pipeline by design, not as an optional add-on wired manually.
- **Tables** — dedicated table processor that exports to structured formats before chunking, rather than relying on TableFormer's coordinate-based cell matching. Better at complex multi-column table layouts.
- **Multi-modal coherence** — text, tables, figures, and audio are processed in separate pipelines then recombined. A bad table parse does not corrupt the surrounding text chunks.

**Where it won't help:**

- **Column mixing** — RAG-Anything still depends on an underlying PDF text extractor (Docling, PyMuPDF, or PDFMiner). If the extractor mis-orders columns, RAG-Anything inherits that problem. It does not fix layout analysis.
- **Equations** — same situation as Docling + VLM unless a specialist math processor (Nougat, GOT-OCR) is explicitly configured.

**Summary:**

| Problem | Docling alone | Docling + VLM | RAG-Anything |
|---|---|---|---|
| Column mixing | Partial (ACCURATE mode) | No improvement | No improvement |
| Figures / charts | Lost | Fixed | Fixed (built-in) |
| Complex tables | Partial | No improvement | Better |
| Equations | Garbled | Description only | Description only |
| Multi-modal coherence | Poor | Moderate | Good |

**The honest caveat:** Task C is specifically about evaluating RAG-Anything empirically. The real question is which PDF backend it uses and whether its table processor actually outperforms TableFormer on the same documents we already tested. We will find out rather than rely on claims.

---
