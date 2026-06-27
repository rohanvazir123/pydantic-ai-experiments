# RAG System Design Basics

Key design areas, trade-offs, and deep-dive topics for building production Retrieval-Augmented Generation systems.

## Table of Contents

- [Pipeline Architecture](#pipeline-architecture)
- [Chunking Strategy](#chunking-strategy)
- [Embedding and Indexing](#embedding-and-indexing)
- [Retrieval](#retrieval)
- [Context Assembly](#context-assembly)
- [Generation and Hallucination Prevention](#generation-and-hallucination-prevention)
- [Evaluation](#evaluation)
- [Knowledge Base and Ingestion](#knowledge-base-and-ingestion)
- [Latency and Performance](#latency-and-performance)
- [Advanced Patterns](#advanced-patterns)
- [Security and Access Control](#security-and-access-control)
- [Domain-Specific RAG Benchmarking](#domain-specific-rag-benchmarking)

---

## Pipeline Architecture

**1.** Walk me through the end-to-end architecture of a production RAG system. Where are the failure modes at each stage and how do you detect them without a human in the loop?

**2.** Where do you draw the boundary between what retrieval handles and what the LLM handles? What problems does retrieval fundamentally not solve, and what are the consequences of asking retrieval to do too much?

**3.** How does your system handle queries that require synthesising information across multiple documents versus queries answerable from a single passage? Does the same pipeline handle both?

**4.** Advanced RAG (HyDE, reranking, query expansion, multi-query) versus naive RAG (embed query, retrieve top-k, generate) — when is the complexity of advanced RAG justified and what does it actually cost?

**5.** When do you use RAG versus fine-tuning versus stuffing the full corpus into the context window? Lay out the trade-offs across accuracy, cost, latency, and knowledge currency.

---

## Chunking Strategy

**6.** How do you decide chunk size? Walk through the failure modes of chunks that are too small versus too large, and how your choice affects both retrieval precision and generation quality.

**7.** Semantic chunking versus fixed-size chunking versus recursive character splitting — when does each win? What does semantic chunking cost that fixed-size does not?

**8.** A retrieved chunk contains the answer to the user's question but lacks the context to make that answer meaningful — the relevant heading is in the previous chunk, the table it references is in the next chunk. How do you handle this at chunking time and at retrieval time?

**9.** Your corpus is heterogeneous — PDFs with embedded tables and figures, DOCX files, HTML pages, audio transcripts, code files. How does chunking strategy differ per document type, and what breaks when you apply a uniform strategy?

**10.** How do you handle document hierarchy — section headers, subsections, numbered lists, footnotes — in your chunking and indexing strategy? How does losing this hierarchy affect retrieval quality?

---

## Embedding and Indexing

**11.** How do you choose an embedding model for a new domain? What are the trade-offs between general-purpose models (OpenAI text-embedding-3-large, nomic-embed-text) and domain-fine-tuned models?

**12.** Dense retrieval versus sparse retrieval (BM25) versus hybrid — lay out the specific failure modes of each and the conditions under which each outperforms the others.

**13.** How does your index handle document updates? A document is edited, a section is removed, a new version is uploaded. What breaks if you do not handle this correctly, and how do you design the update pipeline?

**14.** At what corpus size does your vector indexing strategy change? Walk through the trade-offs between flat index, IVF, HNSW, and product quantization as the corpus scales from 10,000 to 100 million chunks.

**15.** How do you handle a multilingual corpus? Does a single embedding model work across languages, or do you need language-specific indexes? What breaks at retrieval time when query and document language differ?

---

## Retrieval

**16.** Walk me through your full reranking architecture. What does a cross-encoder reranker add over embedding cosine similarity, and what does it cost in latency and accuracy?

**17.** How do you tune the top-k retrieval parameter? What are the specific failure modes of k being too low versus too large, and does k change based on query type?

**18.** How does query transformation improve retrieval quality? Compare HyDE (hypothetical document embeddings), query expansion, and multi-query retrieval — when does each help and what are the failure modes?

**19.** How do you detect when retrieval has failed — the relevant chunk is not in the top-k? How does that failure propagate through the pipeline, and can you catch it before the LLM generates a hallucinated answer?

**20.** A query requires connecting information from three different documents — a policy document, a case study, and a technical specification. None of the individual chunks fully answers the question. How does your retrieval strategy handle multi-hop reasoning?

---

## Context Assembly

**21.** How do you construct the context window from retrieved chunks — ordering, deduplication, formatting? What does the wrong ordering do to generation quality?

**22.** You retrieved 20 chunks averaging 600 tokens each. Your context window is 128k tokens but LLM cost and latency scale with input length. How do you decide what goes in and what stays out?

**23.** Two retrieved chunks contain contradictory information — the policy was updated and both the old and new versions are indexed. How does your system detect the contradiction and handle it in generation?

**24.** How do you preserve document structure — tables, code blocks, numbered lists, mathematical notation — through chunking, indexing, and context assembly? What breaks when structure is lost?

**25.** What is the lost-in-the-middle problem in long context LLMs? How does it affect RAG system design, and what are the mitigation strategies?

---

## Generation and Hallucination Prevention

**26.** How do you prevent the LLM from generating facts not grounded in the retrieved context? What is the role of the system prompt, and what does it still fail to prevent?

**27.** How do you implement citation and attribution — linking specific claims in the generated answer back to their source documents and passages? What are the failure modes of citation systems?

**28.** The retrieved context is insufficient to answer the question — either the relevant document was not indexed, the question is out of scope, or retrieval failed. How does your system detect this and what does it return?

**29.** How do you calibrate when the system should say "I don't know" versus attempt an answer with low-confidence context? What signals drive this decision, and what are the product consequences of miscalibration in each direction?

**30.** How does the LLM's parametric knowledge interact with retrieved context? What happens when the LLM "knows" something that contradicts the retrieved document — does the context win, and how do you enforce it?

---

## Evaluation

**31.** What metrics do you use to evaluate a RAG system — faithfulness, answer relevance, context recall, context precision? What does each measure and what does each fail to capture?

**32.** How do you build a ground-truth evaluation set for RAG when the correct answer is often a synthesised summary across multiple sources, not a single extractable fact?

**33.** How do you detect hallucinations automatically at scale without a human reviewing every response? What is the reliability of automated hallucination detection?

**34.** How do you evaluate retrieval quality separately from generation quality? Why is it important to measure them independently, and what does conflating them hide?

**35.** What is RAGAS and what are its specific limitations? When does RAGAS give misleading scores, and what would you replace or supplement it with?

**36.** How do you evaluate a RAG system when there is no single correct answer — when multiple valid responses exist and the question of quality is inherently subjective?

---

## Knowledge Base and Ingestion

**37.** How do you design the document ingestion pipeline for a corpus of 500,000 documents with continuous updates? Walk through the architecture including parsing, chunking, embedding, and indexing.

**38.** How do you handle document updates and deletions without re-indexing the entire corpus? What consistency issues arise during partial re-indexing?

**39.** How do you detect and handle document staleness — an indexed document that has diverged from its source? What is the user-facing impact if stale documents are retrieved and used for generation?

**40.** Your corpus includes PDFs with scanned images, tables, charts, and embedded figures. How do you extract and index the information in non-text elements, and what do you lose if you skip them?

**41.** How do you handle private, access-controlled documents in a multi-tenant corpus? A user should only retrieve documents they are authorised to read — how is this enforced at the retrieval layer?

---

## Latency and Performance

**42.** What are the latency components of a RAG pipeline and which are on the critical path? Where can you parallelise, where can you cache, and what are the correctness risks of each optimisation?

**43.** What can be cached in a RAG system — embeddings, retrieved chunks, generated answers — and what are the invalidation conditions for each? What silently breaks if you cache without proper invalidation?

**44.** How does streaming change the RAG pipeline architecture and user experience? Are there cases where streaming is harmful?

**45.** At high concurrency — thousands of users querying the same corpus simultaneously — what breaks first, and how do you design the system to handle it?

**46.** When does pre-generating answers for expected queries make sense, and what are the risks? How do you keep pre-generated answers from going stale?

---

## Advanced Patterns

**47.** What is HyDE (Hypothetical Document Embeddings)? When does it improve retrieval quality and when does it hurt? What is its cost?

**48.** What is RAG fusion and multi-query retrieval? When is generating multiple query variants and merging results worth the additional LLM calls?

**49.** How do you implement agentic RAG — where the system decides whether to retrieve, what to retrieve, and whether to retrieve again based on intermediate results? What are the failure modes of agentic loops?

**50.** Your documents are longer than the context window. You cannot fit a single document in context. How do you handle book-length documents, lengthy legal contracts, or multi-hundred-page technical manuals?

**51.** What is self-RAG? How does a model learn to decide when retrieval is needed versus when its parametric knowledge is sufficient? What does this require in terms of training and infrastructure?

**52.** How do you combine RAG over unstructured documents with structured data sources — SQL databases, knowledge graphs? When does a hybrid retrieval architecture make sense and how do you merge the results?

---

## Security and Access Control

**53.** How do you enforce document-level access controls in a RAG system? A user should only see content from documents they are authorised to read — where in the pipeline is this enforced, and what breaks if any layer is bypassed?

**54.** How do you protect against prompt injection through document content? A malicious document contains instructions to the LLM — how do you detect and neutralise this?

**55.** Your corpus contains documents with PII — names, email addresses, financial data, medical records. How do you prevent PII from being surfaced in generated responses to users who are not authorised to see it?

---

## Domain-Specific RAG Benchmarking

Unlike NL2SQL (where Spider2-lite gives a deterministic execution-based score), RAG evaluation is harder because answers are open-ended. The ecosystem is fragmented across retrieval-only benchmarks, end-to-end QA benchmarks, and synthetic test generators. No single benchmark covers everything.

### The Benchmark Landscape

| Benchmark | Type | Domains | What it measures | Deterministic? |
|-----------|------|---------|-----------------|---------------|
| **BEIR** | Retrieval-only | 19 datasets: scientific, medical, financial, legal, news | NDCG@k, Recall@k, MRR@k on retrieval | Yes (qrels) |
| **FinanceBench** | End-to-end RAG | Finance (SEC filings, earnings) | Answer correctness over real PDFs | Partial (numerical answers) |
| **RAGAS testset** | Synthetic E2E | Your own domain | Faithfulness, relevance, recall | No (LLM judge) |
| **BioASQ** | Retrieval + QA | Biomedical | Retrieval + yes/no + factoid answers | Yes (exact match for factoid) |
| **QASPER** | E2E RAG | Scientific papers | Evidence retrieval + answer generation | Partial |
| **HotpotQA** | Multi-hop QA | General | Two-hop reasoning across documents | Yes (exact match) |
| **LegalBench** | Task-specific | Legal | Legal reasoning and classification | Yes |
| **CUAD** | Contract QA | Legal contracts | Contract clause extraction | Yes (span match) |

### BEIR — Best for retrieval component evaluation

19 open datasets. Install and run in under an hour.

```bash
pip install beir

python -c "
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models

# Load a domain-specific dataset
# Available: scifact, fiqa, nfcorpus, arguana, trec-covid, webis-touche2020,
#            dbpedia-entity, fever, climate-fever, nq, hotpotqa, quora, msmarco
corpus, queries, qrels = GenericDataLoader(data_folder='scifact').load(split='test')

# Swap in your embedding model
model = DRES(models.SentenceBERT('your-embedding-model'), batch_size=32)
retriever = EvaluateRetrieval(model, score_function='cos_sim', k_values=[1,5,10,100])
results = retriever.retrieve(corpus, queries)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
print(ndcg)  # {'NDCG@1': 0.72, 'NDCG@5': 0.81, 'NDCG@10': 0.84, ...}
"
```

Each dataset is downloadable via `beir.util.download_and_unzip`. No credentials needed for most datasets.

### FinanceBench — Closest to Spider2-lite for financial RAG

150 open-source annotated Q&A pairs over real SEC filings. Each example has a gold answer, the source evidence text, and the page number. Requires your RAG system to retrieve the right passage and produce the correct numerical or factual answer.

```json
{
  "financebench_id": "financebench_id_00001",
  "question": "What is Apple's revenue for fiscal year 2022?",
  "answer": "$394.33 billion",
  "evidence": [
    {
      "evidence_text": "Net sales: $394,328 million...",
      "page_number": 31,
      "doc_name": "AAPL_2022_10K"
    }
  ]
}
```

Repo: https://github.com/patronus-ai/financebench

### RAGAS synthetic test generation — build your own domain benchmark

When no off-the-shelf benchmark covers your corpus, generate a domain-specific eval set from your own documents:

```python
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

generator = TestsetGenerator(llm=LangchainLLMWrapper(ChatOpenAI(model="gpt-4o")))
testset = generator.generate_with_langchain_docs(
    your_documents,
    testset_size=200,
    # generates: simple factual, multi-hop, abstract/thematic questions
)
# testset.to_pandas() → question, ground_truth, reference_contexts, evolution_type
```

The limitation: questions are generated by the same class of model you're evaluating, introducing optimism bias.

### Design questions

**56.** How do you select which benchmark to use when evaluating a domain-specific RAG system — a retrieval-only benchmark like BEIR, an end-to-end QA benchmark like FinanceBench, or a synthetic generator like RAGAS? What does each measure that the others miss?

**57.** BEIR measures retrieval quality with NDCG@k against a fixed qrel set. Your production system uses a cross-encoder reranker after initial retrieval. How do you benchmark the reranker's contribution separately from the embedding model, and what does BEIR's standard evaluation miss about your pipeline?

**58.** FinanceBench found that GPT-4 with retrieval got 81% of financial questions wrong or refused to answer. What are the most common failure modes in financial document RAG, and how do you specifically address each in your pipeline design?

**59.** How do you build a domain-specific eval set for a corpus where no public benchmark exists — for example, a company's internal policy documents, proprietary technical manuals, or industry-specific contracts? What are the biases in synthetic test generation and how do you mitigate them?

**60.** You run BEIR NDCG@10 on your embedding model and score 0.78 on SciFact. Your end-to-end answer quality on your own domain corpus is 62%. What explains the gap, and which metric should you prioritise improving?

Full setup, integration walkthrough, and failure analysis: [.answers/14_domain_rag_eval.md](.answers/14_domain_rag_eval.md)
