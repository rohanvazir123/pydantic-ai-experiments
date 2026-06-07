# RAG v2 — Implementation TODO

> Methodical, bottom-up build plan. Start from storage, move up through bus → ingestion → retrieval → agent → API → scheduler → frontend. Each phase has a clear deliverable and test gate before the next phase begins.
>
> **Do not skip phases.** Each phase's output is the input contract for the next.

---

## Table of Contents

- [Project Layout](#project-layout)
  - [Backend File Tree](#backend-file-tree)
  - [Frontend File Tree](#frontend-file-tree)
- [UI Design](#ui-design)
- [Phase 0 — Housekeeping & Pre-Flight](#phase-0--housekeeping--pre-flight)
- [Phase 1 — Foundation: Config, Migrations, DB Schema](#phase-1--foundation-config-migrations-db-schema)
- [Phase 2 — Storage Layer](#phase-2--storage-layer)
- [Phase 3 — Message Bus (Redis Streams)](#phase-3--message-bus-redis-streams)
- [Phase 4 — Ingestion Pipeline](#phase-4--ingestion-pipeline)
- [Phase 5 — Retrieval Pipeline](#phase-5--retrieval-pipeline)
- [Phase 6 — Agent & Confidence-Aware Pipeline](#phase-6--agent--confidence-aware-pipeline)
- [Phase 7 — Validation, Hooks, Model Router](#phase-7--validation-hooks-model-router)
- [Phase 8 — API Layer](#phase-8--api-layer)
- [Phase 9 — Security Layer](#phase-9--security-layer)
- [Phase 10 — Ingestion Scheduler](#phase-10--ingestion-scheduler)
- [Phase 11 — Observability](#phase-11--observability)
- [Phase 12 — Evaluation System](#phase-12--evaluation-system)
- [Phase 13 — Docker Compose & Infra](#phase-13--docker-compose--infra)
  - [13.1 Dockerfile](#131-dockerfile-backenddockerfile)
  - [13.2 Docker Compose](#132-docker-compose-backenddocker-composeyml)
  - [13.3 Docker Compose Observability](#133-docker-compose-observability-backenddocker-composeobservabilityyml)
  - [13.4 Nginx Config](#134-nginx-config-backendinfranginxnginxconf)
  - [13.5 Makefile targets](#135-makefile-targets)
- [Phase 14 — Frontend](#phase-14--frontend)
- [Phase 15 — CI/CD & Cloud IaC](#phase-15--cicd--cloud-iac)
  - [15.0 Frontend Deployment — Docker / Node.js](#150-frontend-deployment--docker--nodejs-primary)
- [Phase 16 — Load & Chaos Testing](#phase-16--load--chaos-testing)

---

## Project Layout

The repository root contains two top-level product folders plus shared infra. The existing `rag/` and `kg/` modules stay in place until `knowledge/` reaches feature parity (tracked in Phase 1).

```
rovaz/                              # repo root
├── backend/                        # Python backend (knowledge/ module + infra)
├── frontend/                       # Next.js + Tailwind CSS chatbot UI
├── misc/                           # experimental / archived code (existing)
├── rag/                            # EXISTING — do not delete until backend/ is complete
├── kg/                             # EXISTING — absorbed into backend/knowledge/store/
└── docker-compose.yml              # top-level compose for full-stack local dev
```

---

### Backend File Tree

```
backend/
├── knowledge/
│   ├── __init__.py
│   ├── config/
│   │   └── settings.py                   # Pydantic-settings; all knobs in one place
│   ├── api/
│   │   ├── app.py                        # FastAPI factory (lifespan, middleware stack)
│   │   ├── auth.py                       # JWT decode + RBAC dependency; JWE helpers
│   │   ├── middleware.py                 # CorrelationID, structured-log, audit emission
│   │   ├── quota.py                      # per-tenant rate limiting + budget enforcement
│   │   ├── timeout.py                    # TimeoutBudget dataclass + helpers
│   │   ├── schemas.py                    # Pydantic request/response models (versioned)
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── ingest.py                 # POST /v1/ingest, GET /v1/ingest/{id}/status, SSE stream
│   │       ├── search.py                 # POST /v1/search (sync fast-path)
│   │       ├── chat.py                   # POST /v1/chat, GET /v1/chat/stream (SSE)
│   │       ├── corpus.py                 # GET /v1/corpus, POST /v1/corpus/{id}/cache/invalidate
│   │       ├── evaluate.py               # POST /v1/evaluate/run, GET /v1/evaluate/run/{id}
│   │       ├── feedback.py               # POST /v1/feedback, POST /v1/signals
│   │       ├── scheduler.py              # GET/POST/DELETE /v1/scheduler/jobs (periodic ingest)
│   │       ├── admin.py                  # tenant management, quota override (admin role)
│   │       └── health.py                 # GET /health, GET /metrics
│   ├── bus/
│   │   ├── __init__.py
│   │   ├── publisher.py                  # async Redis Streams XADD helper
│   │   ├── consumer.py                   # base consumer loop: XREADGROUP, ack, retry, DLQ
│   │   ├── circuit_breaker.py            # CircuitBreaker: CLOSED/OPEN/HALF-OPEN in Redis
│   │   ├── backoff.py                    # exponential_backoff() with jitter
│   │   └── schemas.py                    # IngestJob, SearchRequest, WorkerEvent, EvalJob
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── worker.py                     # Redis consumer → pipeline orchestrator (entrypoint)
│   │   ├── pipeline.py                   # per-document orchestrator: asyncio.gather(chunk+graph)
│   │   ├── docling_processor.py          # Docling DocumentConverter wrapper (cached instance)
│   │   ├── chunker.py                    # HybridChunker wrapper → list[ChunkData]
│   │   ├── graph_extractor.py            # docling-graph run_pipeline() wrapper (asyncio.to_thread)
│   │   ├── embedder.py                   # async OpenAI-compatible embedder + L1 lru_cache
│   │   └── models.py                     # ChunkData, SearchResult (with raw_score + confidence)
│   ├── store/
│   │   ├── __init__.py
│   │   ├── vector.py                     # PostgresHybridStore: HNSW + tsvector GIN + RRF
│   │   ├── graph.py                      # AgeGraphStore: Apache AGE Cypher ops over asyncpg
│   │   ├── entity_index.py               # EntityIndex: tsvector shadow table for entity search
│   │   └── cache.py                      # RedisCache: L2 embedding/search/fingerprint cache
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── worker.py                     # Redis consumer → retrieval pipeline (async search)
│   │   ├── retriever.py                  # hybrid retriever: vector + text + optional graph
│   │   ├── graph_retriever.py            # NL→Cypher against AgeGraphStore
│   │   ├── fusion.py                     # RRF (k=60) + CrossEncoder reranker
│   │   └── semantic_cache.py             # L3 semantic cache: pgvector cosine-sim lookup
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── pipeline.py                   # ConfidenceAwarePipeline: 3-layer gate orchestrator
│   │   ├── agent.py                      # Pydantic AI agent: GenerationResult structured output
│   │   ├── judge.py                      # LLMJudge: JudgeResult(verdict, confidence, reasoning)
│   │   ├── model_router.py               # QueryRouter → RoutingDecision (nano model)
│   │   ├── cost_guard.py                 # check_cost_circuit_breaker(): tenant + system limits
│   │   └── prompts.py                    # system prompt templates
│   ├── corpus/
│   │   ├── __init__.py
│   │   ├── registry.py                   # CorpusRegistry: load configs, enforce RBAC at query time
│   │   └── ontologies/                   # Pydantic ontology templates for docling-graph extraction
│   │       ├── __init__.py
│   │       ├── loader.py                 # load_ontology(path) → type[BaseModel]; LRU-cached per worker
│   │       ├── generic.py                # default: GenericDocument (no domain; extracts named entities)
│   │       └── <corpus_id>.py            # user-uploaded domain ontologies (see DESIGN §Knowledge Graph)
│   ├── scheduler/
│   │   ├── __init__.py
│   │   ├── job_store.py                  # ScheduledJob CRUD in PostgreSQL (scheduled_jobs table)
│   │   ├── runner.py                     # APScheduler integration: cron + interval triggers
│   │   └── schemas.py                    # ScheduledJob, JobTrigger, JobStatus Pydantic models
│   ├── hooks/
│   │   ├── __init__.py
│   │   ├── registry.py                   # HookRegistry, HookPoint enum, Hook type alias
│   │   ├── context.py                    # HookContext dataclass
│   │   └── builtins.py                   # placeholder hooks registered at app startup
│   ├── validation/
│   │   ├── __init__.py
│   │   └── pipeline.py                   # V1–V6 validation chain; ContentPolicyResult schema
│   ├── memory/
│   │   ├── __init__.py
│   │   └── mem0_store.py                 # Mem0Store (pgvector-backed per-user memory)
│   ├── billing/
│   │   ├── __init__.py
│   │   ├── metering.py                   # BillingEvent emit + nightly Stripe flush cron
│   │   └── provisioner.py                # TenantProvisioner: onboard, offboard, GDPR erase
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── harness.py                    # EvaluationHarness: orchestrates full eval runs
│   │   ├── datasets.py                   # GoldDataset: load/save/validate (JSONL + PostgreSQL)
│   │   ├── runner.py                     # async runner; publishes EvalJob to knowledge:eval stream
│   │   ├── reporter.py                   # metric aggregation, regression detection, CI report
│   │   ├── schemas.py                    # EvalRun, EvalResult, UserFeedback, ImplicitSignal
│   │   └── metrics/
│   │       ├── __init__.py
│   │       ├── retrieval.py              # HitRate@k, MRR@k, NDCG@k, Precision@k, Recall@k
│   │       ├── faithfulness.py           # claim decomposition + NLI check (nano model)
│   │       ├── answer_relevance.py       # reverse-question embedding similarity
│   │       ├── correctness.py            # BLEU-4, ROUGE, METEOR, BERTScore, semantic-sim
│   │       ├── performance.py            # latency spans, token counts, cost estimation
│   │       └── online.py                 # user feedback aggregation + implicit signal processing
│   └── observability/
│       ├── __init__.py
│       ├── metrics.py                    # Prometheus counters/histograms via prometheus-client
│       ├── langfuse.py                   # Langfuse trace + span helpers
│       └── alerts.py                     # email alert sender (SMTP async, DLQ/circuit/budget events)
├── migrations/
│   ├── 001_initial_schema.sql            # documents, chunks, audit_events
│   ├── 002_corpus_tenant.sql             # corpus_id, tenant_id columns + RLS policies
│   ├── 003_semantic_cache.sql            # semantic_cache table + HNSW index
│   ├── 004_evaluation.sql                # gold_samples, eval_runs, eval_results
│   ├── 005_feedback.sql                  # user_feedback, implicit_signals, token_usage
│   ├── 006_billing.sql                   # tenants, tenant_quotas, billing_events
│   └── 007_scheduler.sql                 # scheduled_jobs table
├── tests/
│   ├── unit/
│   │   ├── test_backoff.py
│   │   ├── test_circuit_breaker.py
│   │   ├── test_fusion.py
│   │   ├── test_validation.py
│   │   ├── test_quota.py
│   │   └── test_scheduler.py
│   ├── integration/
│   │   ├── test_vector_store.py
│   │   ├── test_ingestion_pipeline.py
│   │   ├── test_retrieval_pipeline.py
│   │   ├── test_agent.py
│   │   ├── test_semantic_cache.py
│   │   └── test_api.py
│   ├── load/
│   │   ├── locustfile.py
│   │   └── results/                      # git-ignored large CSVs; summaries committed
│   └── conftest.py
├── infra/
│   ├── nginx/
│   │   └── nginx.conf                    # TLS termination + proxy to API
│   ├── certs/                            # mkcert self-signed for local dev (git-ignored)
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboards/
│           └── rag_v2.json               # pre-built 7-row Grafana dashboard
├── docker-compose.yml                    # local dev: api, workers, pg, redis, ollama, nginx
├── docker-compose.observability.yml      # langfuse, prometheus, grafana (--profile observability)
├── Dockerfile                            # multi-stage: api image + worker image
├── pyproject.toml                        # uv + hatchling; optional extras
├── .env.example
├── install.sh
├── install.ps1
└── Makefile
```

---

### Frontend File Tree

```
frontend/
├── src/
│   ├── app/                              # Next.js 15 App Router
│   │   ├── layout.tsx                    # root layout: font, theme provider, auth wrapper
│   │   ├── page.tsx                      # redirect → /chat
│   │   ├── globals.css                   # Tailwind base + CSS variables (dark/light theme)
│   │   ├── (auth)/
│   │   │   └── login/
│   │   │       └── page.tsx              # JWT login page
│   │   ├── chat/
│   │   │   └── page.tsx                  # main chatbot view
│   │   ├── ingest/
│   │   │   └── page.tsx                  # ingestion management + scheduler
│   │   ├── corpus/
│   │   │   └── page.tsx                  # corpus browser + admin
│   │   ├── eval/
│   │   │   └── page.tsx                  # evaluation runs + metrics dashboard
│   │   ├── logs/
│   │   │   └── page.tsx                  # on-demand log viewer (admin role only); links to Langfuse traces
│   │   └── admin/
│   │       └── page.tsx                  # tenant + quota management (admin role only)
│   ├── components/
│   │   ├── chat/
│   │   │   ├── ChatShell.tsx             # full-page layout: sidebar + chat area
│   │   │   ├── ConversationSidebar.tsx   # conversation history list
│   │   │   ├── MessageList.tsx           # scrollable message list with auto-scroll
│   │   │   ├── MessageBubble.tsx         # user / assistant bubble with markdown rendering
│   │   │   ├── StreamingMessage.tsx      # SSE token streaming with typing cursor
│   │   │   ├── CitationPanel.tsx         # collapsible right panel: source citations
│   │   │   ├── CitationCard.tsx          # single citation: title, excerpt, confidence badge
│   │   │   ├── ConfidenceBadge.tsx       # color-coded confidence bar (green/yellow/red)
│   │   │   ├── PipelineStatusBadge.tsx   # answered / abstained_retrieval / abstained_judge
│   │   │   ├── LowConfidenceWarning.tsx  # yellow banner when low_confidence_context=true
│   │   │   ├── FeedbackBar.tsx           # thumbs up/down + tag picker per message
│   │   │   ├── CorpusSelector.tsx        # multi-select corpus dropdown in input bar
│   │   │   ├── ModelTierPicker.tsx       # small / large override (power user)
│   │   │   └── InputBar.tsx              # textarea + send button + corpus + tier controls
│   │   ├── ingest/
│   │   │   ├── IngestPage.tsx            # tab layout: Upload / Scheduled / History
│   │   │   ├── UploadDropzone.tsx        # drag-and-drop file upload + URL ingest input
│   │   │   ├── CorpusTargetPicker.tsx    # select target corpus for ingest job
│   │   │   ├── JobQueue.tsx              # live list of queued + running jobs
│   │   │   ├── JobStatusCard.tsx         # job: status pill, progress bar, SSE-driven updates
│   │   │   ├── SchedulerPanel.tsx        # create/edit/delete scheduled ingest jobs
│   │   │   ├── ScheduleForm.tsx          # cron expression builder + folder/URL source config
│   │   │   ├── ScheduleList.tsx          # table of scheduled jobs with next-run countdown
│   │   │   └── IngestHistory.tsx         # paginated table of completed ingest jobs
│   │   ├── corpus/
│   │   │   ├── CorpusList.tsx            # card grid of all accessible corpora
│   │   │   ├── CorpusCard.tsx            # corpus: name, doc count, last ingest, graph toggle
│   │   │   ├── CorpusCreateModal.tsx     # create new corpus: name, folders, graph toggle
│   │   │   └── CacheInvalidateButton.tsx # admin: flush L2+L3 for corpus
│   │   ├── eval/
│   │   │   ├── EvalDashboard.tsx         # overview: latest run stats, trend charts
│   │   │   ├── RunTriggerForm.tsx        # trigger new eval run: corpus, model tier, k
│   │   │   ├── RunStatusCard.tsx         # run: status, progress, sample count
│   │   │   ├── MetricsTable.tsx          # HitRate / MRR / NDCG / faithfulness / relevance
│   │   │   ├── RegressionDiff.tsx        # baseline vs current: green/red delta indicators
│   │   │   └── LatencyHeatmap.tsx        # stage breakdown heatmap chart
│   │   ├── admin/
│   │   │   ├── TenantTable.tsx           # list tenants: tier, quota, billing status
│   │   │   ├── QuotaEditor.tsx           # override quota limits for enterprise tenants
│   │   │   └── BudgetGauge.tsx           # cost gauge: spent / limit with warning thresholds
│   │   └── ui/
│   │       ├── Button.tsx
│   │       ├── Badge.tsx
│   │       ├── Spinner.tsx
│   │       ├── Tabs.tsx
│   │       ├── Modal.tsx
│   │       ├── Tooltip.tsx
│   │       ├── ProgressBar.tsx
│   │       ├── EmptyState.tsx
│   │       ├── ErrorBanner.tsx
│   │       └── ThemeToggle.tsx           # dark / light mode toggle
│   ├── lib/
│   │   ├── api.ts                        # typed API client (fetch wrapper + error handling)
│   │   ├── sse.ts                        # SSE stream reader helper (ReadableStream)
│   │   ├── auth.ts                       # JWT storage (httpOnly cookie), refresh logic
│   │   └── format.ts                     # formatDate, formatCost, formatMs utilities
│   ├── hooks/
│   │   ├── useChat.ts                    # chat state, send message, SSE streaming
│   │   ├── useIngest.ts                  # submit job, poll status, SSE job progress
│   │   ├── useCorpus.ts                  # fetch corpus list, create, invalidate cache
│   │   ├── useScheduler.ts               # CRUD for scheduled ingestion jobs
│   │   ├── useEval.ts                    # trigger run, poll results, regression compare
│   │   └── useTheme.ts                   # dark/light mode state with localStorage
│   ├── store/
│   │   └── chatStore.ts                  # Zustand store: conversations, active corpus, settings
│   └── types/
│       ├── api.ts                        # API request/response TypeScript types
│       ├── chat.ts                       # Message, Citation, PipelineStatus types
│       ├── ingest.ts                     # IngestJob, ScheduledJob types
│       └── eval.ts                       # EvalRun, EvalResult, GoldSample types
├── public/
│   └── favicon.ico
├── Dockerfile                            # multi-stage: deps → builder → runner (standalone output)
├── tailwind.config.ts
├── next.config.ts                        # output: "standalone" enabled
├── package.json
├── package-lock.json                     # committed; lockfile ensures reproducible builds
├── tsconfig.json
└── .env.local.example
```

---

## UI Design

> The frontend is a professional, dark-first chatbot application. Every design decision below is a constraint — treat them as requirements, not suggestions.

### Design System

| Token | Value |
|-------|-------|
| Primary font | Inter (variable weight) |
| Mono font | JetBrains Mono (code blocks, citations) |
| Dark background | `#0f1117` (near-black, not pure black) |
| Dark surface | `#1a1d27` (cards, panels) |
| Dark border | `#2d3048` |
| Accent blue | `#4f6ef7` (primary action, links) |
| Success green | `#22c55e` |
| Warning amber | `#f59e0b` |
| Error red | `#ef4444` |
| Light mode | System default with Tailwind `slate` palette |

### Page Layouts

#### `/chat` — Main Chat View

```
┌────────────────────────────────────────────────────────────────────────┐
│  HEADER: Logo | CorpusSelector (multi) | ModelTierPicker | ThemeToggle │
├──────────────┬─────────────────────────────────────┬───────────────────┤
│ CONVERSATION │           MESSAGE LIST               │  CITATION PANEL  │
│  SIDEBAR     │  ┌─────────────────────────┐         │  (collapsible)   │
│  (240px)     │  │ UserBubble              │         │                  │
│              │  │  "What is the PTO..."   │         │ [Source 1]       │
│ > Chat 1     │  └─────────────────────────┘         │  Employee Hand.. │
│   Chat 2     │  ┌─────────────────────────┐         │  ████████  0.87  │
│              │  │ AssistantBubble         │         │  "Employees acc.."│
│              │  │  [StreamingMessage]     │         │                  │
│              │  │  + PipelineStatusBadge  │         │ [Source 2]       │
│              │  │  + FeedbackBar 👍 👎    │         │  HR Policy 2024  │
│              │  │  + LowConfidenceWarning │         │  ████░░░░  0.52  │
│              │  └─────────────────────────┘         │                  │
├──────────────┴────────────────────────────────────────────────────────┤
│ INPUT BAR: [textarea placeholder: "Ask about your knowledge base..."]  │
│             [Corpus: hr-policies ▾] [Tier: auto ▾]       [Send →]     │
└────────────────────────────────────────────────────────────────────────┘
```

**Behaviour rules:**
- Citation panel opens on first assistant response; collapses on mobile (< 768px).
- Each `CitationCard` has a color-coded `ConfidenceBadge`: green ≥ 0.7, amber 0.4–0.69, red < 0.4.
- `PipelineStatusBadge` is a small pill below the assistant message: "Answered" (green), "Low confidence" (amber), "Abstained — retrieval gap" (red with tooltip).
- `LowConfidenceWarning` is a yellow inline banner above the answer text when `low_confidence_context: true`.
- SSE streaming: tokens appear one by one. A blinking cursor shows during streaming. On stream end, `CitationPanel` populates with sources.
- `FeedbackBar` appears below each completed assistant message. Thumbs up/down submit instantly; tag picker (hallucinated / irrelevant / incomplete / correct) appears on thumbs-down.
- The conversation sidebar lists past conversations with relative timestamps. Clicking one loads history; "New chat" clears state.

#### `/ingest` — Ingestion Management

```
┌────────────────────────────────────────────────────────────────────────┐
│  HEADER: Ingestion Management                                          │
├────────────────────────────────────────────────────────────────────────┤
│  TABS: [Upload & Ingest] [Scheduled Jobs] [History]                    │
├────────────────────────────────────────────────────────────────────────┤
│  TAB: Upload & Ingest                                                  │
│                                                                        │
│  ┌── UploadDropzone ──────────────────────────────────────────────┐    │
│  │  Drag & drop files here, or click to browse                   │    │
│  │  Supported: PDF, DOCX, MD, TXT, MP3, WAV (Whisper ASR)        │    │
│  │  Or enter a URL:  [_________________________________] [Add]    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                        │
│  Target corpus: [hr-policies ▾]   Graph extraction: [● ON]            │
│                                                                        │
│  [Submit Ingest Job →]                                                 │
│                                                                        │
│  ── Active Jobs ────────────────────────────────────────────────────  │
│  ┌── JobStatusCard ──────────────────────────────────────────────┐    │
│  │  📄 employee_handbook_v3.pdf  [● Processing]                  │    │
│  │  ████████████░░░░░░░░░░░░  65%  Embedding chunks...           │    │
│  │  Corpus: hr-policies  •  Started 42s ago                      │    │
│  └────────────────────────────────────────────────────────────────┘    │
├────────────────────────────────────────────────────────────────────────┤
│  TAB: Scheduled Jobs                                                   │
│                                                                        │
│  [+ New Scheduled Job]                                                 │
│                                                                        │
│  ┌── ScheduleList ───────────────────────────────────────────────┐    │
│  │  Name          Source              Corpus       Next Run  Freq │    │
│  │  ─────────────────────────────────────────────────────────── │    │
│  │  HR Docs sync  /mnt/hr-docs/       hr-policies  in 6h   Daily │    │
│  │  Legal weekly  gs://bucket/legal/  legal-corp   in 3d   Weekly│    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                        │
│  ── ScheduleForm (modal) ──────────────────────────────────────────── │
│  │  Job name:  [___________]                                     │    │
│  │  Source:    ● Local path  ○ URL  ○ S3/GCS bucket              │    │
│  │  Path:      [/mnt/documents/hr/  ]                            │    │
│  │  Corpus:    [hr-policies ▾]                                   │    │
│  │  Schedule:  ● Every  [1] [days ▾]                            │    │
│  │             ○ Cron:  [0 2 * * *]  (UTC)                      │    │
│  │  Mode:      ● Incremental (skip unchanged)  ○ Full rescan     │    │
│  │  Graph extraction: [● ON]                                     │    │
│  │  [Cancel]                        [Create Job]                 │    │
├────────────────────────────────────────────────────────────────────────┤
│  TAB: History                                                          │
│  Paginated table: filename | corpus | status | chunks | duration | ago │
└────────────────────────────────────────────────────────────────────────┘
```

#### `/corpus` — Corpus Browser

```
┌────────────────────────────────────────────────────────────────────────┐
│  Knowledge Corpora                                  [+ New Corpus]     │
├────────────────────────────────────────────────────────────────────────┤
│  ┌── CorpusCard ───────────┐  ┌── CorpusCard ───────────┐            │
│  │  📚 hr-policies         │  │  ⚖️  legal-contracts     │            │
│  │  1,243 chunks           │  │  8,901 chunks            │            │
│  │  Last ingested: 2h ago  │  │  Last ingested: 3d ago   │            │
│  │  Graph: ● ON            │  │  Graph: ○ OFF            │            │
│  │  [Invalidate Cache]     │  │  [Invalidate Cache]      │            │
│  │  [View Documents]       │  │  [View Documents]        │            │
│  └─────────────────────────┘  └─────────────────────────┘            │
└────────────────────────────────────────────────────────────────────────┘
```

#### `/eval` — Evaluation Dashboard

```
┌────────────────────────────────────────────────────────────────────────┐
│  Evaluation  •  corpus: [hr-policies ▾]          [Run Evaluation →]   │
├────────────────────────────────────────────────────────────────────────┤
│  Latest Run: #42  •  June 5 2026  •  Baseline: #38  •  ✅ No regression│
│                                                                        │
│  ── Retrieval Quality ─────────────────────────────────────────────── │
│  Hit@5   MRR@5   NDCG@5   Precision@5   Recall@5                       │
│  0.92    0.78    0.84     0.71          0.68   (vs baseline ▲+0.03)   │
│                                                                        │
│  ── Generation Quality ──────────────────────────────────────────────  │
│  Faithfulness: 0.88   Answer Relevance: 0.82   Abstention Rate: 8%    │
│                                                                        │
│  ── Latency ─────────────────────────────────────────────────────────  │
│  P50: 620ms   P95: 1,840ms   P99: 3,100ms   SLA compliance: 97.2%    │
│                                                                        │
│  ── Per-Sample Results (paginated table) ───────────────────────────── │
│  Query | Hit | MRR | Faith | Status | Latency                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Responsive Behaviour
- **Desktop (≥ 1280px)**: full 3-column chat layout (sidebar + messages + citations).
- **Tablet (768–1279px)**: citations panel collapses to an icon; tap to slide in.
- **Mobile (< 768px)**: sidebar hidden (hamburger); citations appear as a bottom sheet.

### Accessibility
- All interactive elements have `aria-label`.
- Streaming messages use `aria-live="polite"` region.
- Colour contrast ≥ 4.5:1 for all text on both dark and light themes.
- Keyboard navigation: Tab through input bar controls; Enter to send; Escape closes modals.

---

## Phase 0 — Housekeeping & Pre-Flight

> Gate: all existing tests pass; clean slate for new modules.

- [ ] Run `python -m pytest rag/tests/ -m "not integration" -v` — confirm 0 failures after the housekeeping moves from RAGV2_DESIGN.md Phase A: `kg/legal/` → `misc/kg_legal_cuad/`, `rag/legal/` → `misc/kg_legal_cuad/rag_data/`, `rag/ingestion/cuad_ingestion.py` → `misc/kg_legal_cuad/`, `rag/tests/knowledge_graph/` → `misc/kg_legal_cuad/tests/kg/`
- [ ] Verify `ruff check rag/ && ruff format rag/` clean
- [ ] Create `backend/` and `frontend/` directories at repo root
- [ ] Create `backend/knowledge/__init__.py` and stub all sub-packages (empty `__init__.py` only)
- [ ] Create `frontend/` with `package.json` scaffold (`next`, `tailwindcss`, `typescript` deps)
- [ ] Add `backend/.python-version` → `3.13`
- [ ] Add `backend/pyproject.toml` with all optional extras as designed (copy structure from `rag/pyproject.toml` and extend)
- [ ] Commit empty scaffold; tag `v2-scaffold`

**Test gate:** `cd backend && uv sync --extra all` completes without error.

---

## Phase 1 — Foundation: Config, Migrations, DB Schema

> Gate: settings load cleanly; all migrations run on a blank DB; RLS enforced.

### 1.1 Settings (`knowledge/config/settings.py`)

- [ ] Port all fields from `rag/config/settings.py`
- [ ] Add: `redis_url`, `redis_max_connections`
- [ ] Add: `corpus_configs: list[CorpusConfig]` (parsed from `CORPUS_CONFIGS_JSON` env)
- [ ] Add: JWT fields — `jwt_algorithm`, `jwt_public_key_path`, `jwks_cache_ttl_s`
- [ ] Add: JWE fields — `jwe_algorithm`, `jwe_content_encryption`
- [ ] Add: cache fields — `semantic_cache_enabled`, `semantic_cache_threshold`, `semantic_cache_ttl_minutes`, `semantic_cache_max_rows`
- [ ] Add: worker fields — `ingest_worker_concurrency`, `retrieval_worker_concurrency`, `max_retries`, `job_timeout_s`
- [ ] Add: model tier fields — `model_tier_nano`, `model_tier_small`, `model_tier_large`, `model_routing_enabled`, `model_routing_timeout_s`
- [ ] Add: confidence fields — `min_confidence_score`, `confidence_warn_threshold`, `retrieval_confidence_threshold`, `judge_confidence_threshold`, `judge_k`
- [ ] Add: alert fields — `alert_email`, `smtp_host`, `smtp_port`, `smtp_user`, `smtp_password`, `smtp_from`
- [ ] Add: cost circuit breaker — `system_daily_cost_limit_usd`
- [ ] Add: scheduler fields — `scheduler_enabled`, `scheduler_max_concurrent_jobs`
- [ ] Write unit tests: `tests/unit/test_settings.py` — load from `.env.example`, validate credential masking

### 1.2 Database Migrations (`migrations/`)

- [ ] `001_initial_schema.sql` — `documents`, `chunks` (without `corpus_id`/`tenant_id` yet — those are added additively in 002), `audit_events`; pgvector + AGE extensions; HNSW on `chunks.embedding`; GIN on `chunks.content_tsv`
- [ ] `002_corpus_tenant.sql` — additive migration: `ALTER TABLE documents ADD COLUMN corpus_id TEXT NOT NULL DEFAULT 'default'`, same for `chunks`; then `ALTER TABLE … ALTER COLUMN … DROP DEFAULT`; `ALTER TABLE … ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'`; B-tree indexes on both; RLS policies (`CREATE POLICY tenant_isolation ON … USING (tenant_id = current_setting('app.tenant_id'))`)
- [ ] `003_semantic_cache.sql` — `semantic_cache` table with HNSW index on `query_emb`
- [ ] `004_evaluation.sql` — `gold_samples`, `eval_runs`, `eval_results`
- [ ] `005_feedback.sql` — `user_feedback`, `implicit_signals`, `token_usage`
- [ ] `006_billing.sql` — `tenants`, `tenant_quotas`, `billing_events`
- [ ] `007_scheduler.sql` — `scheduled_jobs` table (id, name, source_config, corpus_id, cron_expr, mode, next_run_at, last_run_at, status)
- [ ] Add `Makefile` target: `make migrate` runs all migration files in order against `DATABASE_URL`
- [ ] Verify RLS: connect as application user, `SET LOCAL app.tenant_id = 'x'` — confirm rows from other tenants invisible

**Test gate:** `make migrate` on blank DB → all tables + indexes created; `SELECT 1 FROM chunks` returns 0 rows.

---

## Phase 2 — Storage Layer

> Gate: each store can connect, write, read, and close without leaking connections. No ingestion yet.

### 2.1 Vector Store (`knowledge/store/vector.py`)

- [ ] Port `PostgresHybridStore` from `rag/storage/vector_store/postgres.py`
- [ ] Add `corpus_id` + `tenant_id` filter to every query (`WHERE corpus_id = $1 AND tenant_id = $2`)
- [ ] Add `SET LOCAL app.tenant_id` before every connection checkout (RLS enforcement)
- [ ] Keep `SearchResult` fields but add `raw_score_type: Literal["cosine_similarity", "ts_rank", "rrf"]` and `confidence: float | None = None`
- [ ] Expose `semantic_search`, `text_search`, `hybrid_search` (RRF), `upsert_chunks`, `delete_by_corpus`
- [ ] Write integration tests: `tests/integration/test_vector_store.py`

### 2.2 Graph Store (`knowledge/store/graph.py`)

- [ ] Port `AgeGraphStore` from `kg/age_graph_store.py`
- [ ] Add tenant-namespaced graph names (`{tenant_id}_{corpus_id}`)
- [ ] Expose `upsert_entities`, `upsert_edges`, `run_cypher`, `delete_tenant_graph`
- [ ] Write integration tests (mock AGE socket if AGE container not running)

### 2.3 Entity Index (`knowledge/store/entity_index.py`)

- [ ] Port from `kg/entity_index.py`
- [ ] Add `corpus_id` scoping to all queries

### 2.4 Redis Cache (`knowledge/store/cache.py`)

- [ ] Implement `RedisCache` using `redis.asyncio`
- [ ] Methods: `get_embedding`, `set_embedding`, `get_search`, `set_search`, `get_fingerprint`, `set_fingerprint`, `delete_corpus_search_cache`
- [ ] All keys namespace by `corpus_id` and `tenant_id`
- [ ] Serialize with `msgpack` (faster than JSON for binary vectors)
- [ ] Write unit tests with `fakeredis` (no live Redis needed)

**Test gate:** `tests/unit/test_cache.py` — all cache operations pass; `tests/integration/test_vector_store.py` — upsert + search + delete round-trip passes.

---

## Phase 3 — Message Bus (Redis Streams)

> Gate: a job can be published, consumed, retried, and dead-lettered. Circuit breaker state shared across multiple consumer instances.

### 3.1 Backoff (`knowledge/bus/backoff.py`)

- [ ] Implement `exponential_backoff(attempt, base_s, multiplier, max_s, jitter_factor)` exactly as designed
- [ ] Write unit tests: `tests/unit/test_backoff.py` — test schedule for 3 attempts; verify jitter bounds

### 3.2 Circuit Breaker (`knowledge/bus/circuit_breaker.py`)

- [ ] Implement `CircuitBreaker` with Redis state storage (`cb:{name}:state`, `cb:{name}:failures`, `cb:{name}:opened_at`)
- [ ] States: `CLOSED → OPEN → HALF-OPEN → CLOSED`
- [ ] Thresholds configurable via `CircuitBreakerSettings` dataclass
- [ ] `async def call(coro)` — transparent wrapper; raises `CircuitOpenError` when open
- [ ] Write unit tests: `tests/unit/test_circuit_breaker.py` — use `fakeredis`; test all state transitions

### 3.3 Publisher (`knowledge/bus/publisher.py`)

- [ ] `async def publish_ingest_job(job: IngestJob) -> str` — `XADD knowledge:ingest * ...`; returns message ID
- [ ] `async def publish_search_request(req: SearchRequest) -> str`
- [ ] `async def publish_eval_job(job: EvalJob) -> str`
- [ ] Job hash stored alongside: `HSET job:{job_id} status "queued" corpus_id ... submitted_at ...`

### 3.4 Consumer Base (`knowledge/bus/consumer.py`)

- [ ] `async def consume_loop(stream, group, worker_id, handler)` — `XREADGROUP` loop
- [ ] `_execute_with_retry(msg_id, job, handler)` — try/except/ack/DLQ/backoff logic exactly as designed in RAGV2_DESIGN.md
- [ ] `move_to_dlq(job, exc)` — `XADD knowledge:ingest:dlq`; fire `ON_ERROR` hook; send alert email
- [ ] Heartbeat: `SET worker:{id}:heartbeat {ts} EX 30` every 10s
- [ ] Consumer group creation on startup: `XGROUP CREATE ... MKSTREAM`
- [ ] Write unit tests: `tests/unit/test_consumer.py` — mock handler; verify ack on success, retry on transient, DLQ on permanent, DLQ after MAX_RETRIES

### 3.5 Job Schemas (`knowledge/bus/schemas.py`)

- [ ] `IngestJob`: `job_id`, `tenant_id`, `corpus_id`, `source_path`, `enable_graph_extraction`, `mode` (full/incremental), `attempt`
- [ ] `SearchRequest`: `request_id`, `tenant_id`, `corpus_id`, `query`, `k`, `callback_stream`
- [ ] `WorkerEvent`: `event_type` (heartbeat/job_complete/job_failed), `worker_id`, `job_id`, `ts`
- [ ] `EvalJob`: `run_id`, `corpus_id`, `sample_ids`, `model_tier`, `k`

**Test gate:** all unit tests in `tests/unit/test_backoff.py`, `test_circuit_breaker.py`, `test_consumer.py` pass without live Redis.

---

## Phase 4 — Ingestion Pipeline

> Gate: a single document can be ingested end-to-end (Docling → chunks → embeddings → vector store) via the worker. No API or retrieval needed yet.
>
> **Architecture anchor:** v2 carries forward the same Docling integration patterns as v1 (`rag/ingestion/pipeline.py` + `rag/ingestion/chunkers/docling.py`). The core logic is not rewritten — it is lifted and adapted for async workers, corpus scoping, and Redis caching. Read those files before implementing each section below.
>
> **Sync vs async rule:** CPU-bound operations that have no async API are run synchronously but offloaded to the threadpool via `asyncio.to_thread()` — the event loop stays unblocked. I/O-bound operations (DB, HTTP, Redis, LLM) must always use a native async library. Trivially fast sync calls (< 1ms, no I/O) may run inline.
>
> | Call | Pattern | Reason |
> |------|---------|--------|
> | `DocumentConverter.convert(path)` | `await asyncio.to_thread(converter.convert, path)` | CPU-bound; no async API |
> | `HybridChunker.chunk()` + `list(iter)` | chained in same `to_thread` block after conversion | CPU-bound; shares the thread |
> | `AutoTokenizer.encode(text)` | inline sync | < 1ms per call; overhead of `to_thread` exceeds benefit |
> | `asyncpg`, `redis.asyncio`, `AsyncOpenAI` | native `await` | async I/O libraries; never block |

### 4.1 Embedder (`knowledge/ingestion/embedder.py`)

- [ ] Port from `rag/ingestion/embedder.py`
- [ ] Add `@functools.lru_cache(maxsize=1000)` on `embed(text: str) → tuple[float, ...]` (L1 cache)
- [ ] Add timeout via `asyncio.wait_for` + `settings.embedding_timeout_s`
- [ ] Add exponential backoff on `RateLimitError`, `APIConnectionError`, `APITimeoutError`
- [ ] Write unit tests: mock `AsyncOpenAI`; verify retry logic and LRU cache hit

### 4.2 Docling Processor (`knowledge/ingestion/docling_processor.py`)

The v1 pipeline uses **two separate converters** for good reason — PDF may need a VLM pipeline for figure description, while all other rich formats (DOCX/PPTX/XLSX/HTML/MD) only need the text layer. Preserve this split exactly.

**Format routing** (copy constants from v1 `rag/ingestion/pipeline.py`):
```python
_PDF_FORMATS        = frozenset({".pdf"})
_STRUCTURED_FORMATS = frozenset({".docx", ".doc", ".pptx", ".ppt",
                                  ".xlsx", ".xls", ".html", ".htm",
                                  ".md", ".markdown"})
_AUDIO_FORMATS      = frozenset({".mp3", ".wav", ".m4a", ".flac"})
# Anything else → plain text read, no Docling
```

- [ ] `class DoclingProcessor` — holds two lazily-initialised converter instances per worker process (not per request); thread-safe because workers are single-threaded async loops
- [ ] `_get_pdf_converter()` — mirrors v1 exactly:
  - `VLM_ENABLED=false` (default): `PdfPipelineOptions(do_ocr=False, do_table_structure=True, do_picture_description=False)`
  - `VLM_ENABLED=true`: same pipeline + `PictureDescriptionApiOptions` pointing at Ollama (`settings.vlm_base_url`, `settings.vlm_model`); prompt: *"Describe this image in detail, explicitly capturing any charts, tables, diagrams, or structural data shown."*
- [ ] `_get_standard_converter()` — DOCX/PPTX/XLSX/HTML/MD: always `PdfPipelineOptions(do_ocr=False, do_picture_description=False, generate_picture_images=False)`; VLM_ENABLED has no effect on this path
- [ ] `async def process(path: Path) → ConversionResult`:
  - Route by extension using the constants above
  - PDF → `await asyncio.to_thread(self._get_pdf_converter().convert, path)`
  - Structured → `await asyncio.to_thread(self._get_standard_converter().convert, path)`
  - Audio → `await asyncio.to_thread(self._convert_audio, path)` (see 4.2a)
  - Plain text / unknown → direct `Path.read_text()` (no Docling)
  - Return `ConversionResult(markdown: str, docling_doc: DoclingDocument | None)`
  - On `DocumentConversionError` → raise `PermanentError` (no retry; DLQ immediately)
  - On any other exception → log + return `ConversionResult(markdown="[Error: ...]", docling_doc=None)` so the document is stored with an error placeholder rather than lost
- [ ] Add `settings.vlm_enabled`, `settings.vlm_base_url`, `settings.vlm_model`, `settings.vlm_timeout`, `settings.vlm_concurrency` to `knowledge/config/settings.py`

#### 4.2a Audio transcription

Mirrors v1 `_transcribe_audio()` exactly — uses Docling's ASR pipeline with Whisper Turbo, not `openai-whisper` directly:

```python
from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline

pipeline_options = AsrPipelineOptions()
pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

converter = DocumentConverter(
    format_options={
        InputFormat.AUDIO: AudioFormatOption(
            pipeline_cls=AsrPipeline,
            pipeline_options=pipeline_options,
        )
    }
)
result = converter.convert(audio_path)
return result.document.export_to_markdown(), result.document
```

- [ ] Wrap in `asyncio.to_thread`; on failure return `("[Error: Could not transcribe ...]", None)` — same graceful degradation as v1
- [ ] Audio conversion is guarded by `extra = "audio"` import check; if `openai-whisper` / FFmpeg not present, return the error placeholder immediately

### 4.3 Chunker (`knowledge/ingestion/chunker.py`)

Port `DoclingHybridChunker` from `rag/ingestion/chunkers/docling.py` with these v1 patterns preserved:

- [ ] `HybridChunker(tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_MODEL), max_tokens=config.max_tokens, merge_peers=True)` — same tokenizer model (`sentence-transformers/all-MiniLM-L6-v2`), same `merge_peers=True`
- [ ] `chunk_document(content, title, source, metadata, docling_doc)`:
  - With `docling_doc`: `chunker.chunk(dl_doc=docling_doc)` → for each chunk call `chunker.contextualize(chunk=chunk)` to get heading hierarchy prepended — **do not skip `contextualize()`**, it is what makes chunks self-contained for retrieval
  - Without `docling_doc`: `_simple_fallback_chunk()` — sliding window with sentence-boundary detection (copy from v1 exactly)
  - Token count via `len(tokenizer.encode(contextualized_text))` — stored on `ChunkData.token_count`
  - Metadata per chunk: `title`, `source`, `chunk_method` (`"hybrid"` or `"simple_fallback"`), `total_chunks`, `token_count`, `has_context`
- [ ] Inject `corpus_id`, `tenant_id`, and `CorpusConfig.metadata_tags` into every chunk's metadata (v2 addition)
- [ ] Fallback is triggered by: no `docling_doc` provided, or `HybridChunker.chunk()` raises any exception — log the exception, fall through to `_simple_fallback_chunk()`
- [ ] `ChunkingConfig` fields: `chunk_size`, `chunk_overlap`, `min_chunk_size`, `max_tokens` — same as v1

### 4.4 Graph Extractor (`knowledge/ingestion/graph_extractor.py`)

> Read `RAGV2_DESIGN.md §Knowledge Graph Extraction` before implementing. The correct API is `run_pipeline(PipelineConfig(...)) → PipelineContext` — there is no `PipelineOrchestrator` class.

- [ ] Guard first: if `corpus_config.enable_graph_extraction is False` → return `None` immediately (no LLM call, no log spam)

- [ ] `async def extract_graph(doc_path: Path, corpus_config: CorpusConfig, settings: Settings) → str | None` — returns Cypher string or `None`

  **Implementation:**
  ```python
  def _run_sync() -> str:
      from docling_graph import PipelineConfig, run_pipeline
      from docling_graph.core.exporters import CypherExporter

      ontology_class = load_ontology(corpus_config.graph_ontology_path)  # LRU-cached

      config = PipelineConfig(
          source=str(doc_path),
          template=ontology_class,
          backend=corpus_config.graph_extraction_backend,         # "llm" | "vlm"
          inference="local",
          provider_override=corpus_config.graph_extraction_provider,  # "ollama"
          model_override=corpus_config.graph_extraction_model,        # "llama3.2:3b"
          processing_mode=corpus_config.graph_processing_mode,        # "many-to-one"
          extraction_contract=corpus_config.graph_extraction_contract, # "staged"
          use_chunking=True,
          chunk_max_tokens=settings.chunk_max_tokens,
          structured_output=True,
          dump_to_disk=False,   # API mode — no files on disk
      )
      context = run_pipeline(config)

      # Extract Cypher from the NetworkX graph in PipelineContext
      exporter = CypherExporter()
      with tempfile.NamedTemporaryFile(suffix=".cypher", delete=False) as f:
          tmp = Path(f.name)
      exporter.export(context.knowledge_graph, tmp)
      cypher = tmp.read_text(encoding="utf-8")
      tmp.unlink()
      return cypher

  return await asyncio.wait_for(
      asyncio.to_thread(_run_sync),
      timeout=settings.graph_extraction_timeout_s,   # default 120s
  )
  ```

- [ ] Wrap call in `try/except`: timeout → log warning, return `None`; any other exception → log error, return `None` — both are soft failures; vector path continues regardless
- [ ] On soft failure: set `chunk_metadata["graph_extraction_failed"] = True` on all chunks for this document; do NOT raise; do NOT go to DLQ
- [ ] `PipelineConfig` is instantiated fresh per call (it is not a singleton — thread-safe)

- [ ] **Ontology loader** (`knowledge/corpus/ontologies/loader.py`):
  - `@functools.lru_cache(maxsize=32)` on `load_ontology(path: str | None) → type[BaseModel]`
  - `path=None` → return `GenericDocument` from `generic.py`
  - Otherwise: `importlib.util.spec_from_file_location()` → exec the Python file → return the last `BaseModel` subclass found (root class convention)
  - Raise `FileNotFoundError` if ontology file not found (propagates as permanent error → DLQ)

- [ ] **Generic ontology** (`knowledge/corpus/ontologies/generic.py`):
  - `GenericEntity(BaseModel)` with `graph_id_fields=["name"]`, fields: `name`, `entity_type`, `description`
  - `GenericDocument(BaseModel)` with `graph_id_fields=["title"]`, edge `"MENTIONS"` → `List[GenericEntity]`
  - This is the fallback when no domain ontology is configured

- [ ] Add to `pyproject.toml` `ingestion` extra: `"docling-graph>=1.5.1"`
- [ ] Add `settings.graph_extraction_timeout_s: float = 120.0` to config

### 4.5 Ingestion Pipeline Orchestrator (`knowledge/ingestion/pipeline.py`)

Mirrors v1 `DocumentIngestionPipeline` but adapted for the Redis worker model and corpus scoping.

**Incremental ingestion** — carry forward v1's hash-based change detection:
- Full mode (`job.mode == "full"`): no hash check; process every file
- Incremental mode (`job.mode == "incremental"`):
  1. Compute `sha256(file_content)` — use SHA-256, not MD5 like v1 (stronger, same cost)
  2. Check `RedisCache.get_fingerprint(sha256)` (L2) → if hit, skip file entirely
  3. On cache miss, check `documents.metadata->>'content_hash'` in PG (handles Redis flush/restart)
  4. If hash matches DB → set Redis fingerprint cache and skip
  5. If hash differs or not found → delete old document+chunks (`ON CONFLICT` upsert also works) → ingest fresh
  6. After successful ingest → `RedisCache.set_fingerprint(sha256)`
  7. Deleted files: `store.get_all_document_sources(corpus_id)` → diff against current file list → delete removed sources

**YAML frontmatter** — carry forward v1's `_extract_document_metadata()`:
- [ ] Parse YAML frontmatter (`---` block) if present; merge into chunk metadata
- [ ] Always include: `file_path`, `file_size`, `ingestion_date`, `content_hash`, `line_count`, `word_count`

**Title extraction** — carry forward v1's `_extract_title()`:
- [ ] Scan first 10 lines for `# ` heading; fall back to `Path(file_path).stem`

**Main flow** — `async def run(job: IngestJob) → IngestResult`:
  1. Emit `PRE_INGEST` hook
  2. `DoclingProcessor.process(path)` → `(markdown, docling_doc)`
  3. Incremental hash check (skip if unchanged)
  4. `asyncio.gather(chunker_task, graph_task)` — parallel
     - chunker_task: `chunker.chunk_document(...)` → `embedder.embed_batch(chunks)` → `vector_store.upsert_chunks(corpus_id, tenant_id)`
     - graph_task (if `enable_graph_extraction`): `graph_extractor.extract(docling_doc)` → `graph_store.upsert_entities()` → `entity_index.upsert()`
  5. Set Redis fingerprint cache
  6. Publish `IngestCompleteEvent` to `knowledge:events`
  7. Update job hash: `HSET job:{id} status "completed" chunks_ingested ... completed_at ...`
  8. Emit `POST_INGEST` hook
  9. Invalidate L2 search cache for `corpus_id` (so next query sees new chunks immediately)

- [ ] `_find_document_files(source_path: Path) → list[Path]` — mirrors v1's glob patterns across all format constants; recursive

### 4.6 Ingestion Worker Entrypoint (`knowledge/ingestion/worker.py`)

- [ ] Instantiate `DoclingProcessor` (singleton — converters are expensive to init), `DoclingHybridChunker`, `Embedder`, `PostgresHybridStore`, `AgeGraphStore`, `RedisCache` at startup
- [ ] Connect all stores, create Redis consumer group, start `consume_loop`
- [ ] `python -m knowledge.ingestion.worker` is the Docker CMD
- [ ] Graceful shutdown on `SIGTERM`: drain current job, stop accepting new messages, close all connections

### 4.7 Models (`knowledge/ingestion/models.py`)

- [ ] `ChunkData` — `content`, `metadata: dict[str, Any]`, `chunk_index`, `token_count`, `start_char`, `end_char` — same fields as v1; add `corpus_id`, `tenant_id`
- [ ] `ChunkingConfig` — `chunk_size`, `chunk_overlap`, `min_chunk_size`, `max_tokens` — same as v1
- [ ] `IngestionConfig` — wraps `ChunkingConfig` + `documents_folder`, `clean_before_ingest`
- [ ] `ConversionResult` — `markdown: str`, `docling_doc: DoclingDocument | None`, `format: str`
- [ ] `SearchResult` — `chunk_id`, `document_id`, `document_title`, `document_source`, `content`, `metadata`, `raw_score`, `raw_score_type`, `confidence: float | None`
- [ ] `Citation` — `chunk_id`, `document_title`, `document_source`, `relevance_score` (= `confidence`), `excerpt`
- [ ] `IngestResult` — `job_id`, `chunks_ingested`, `graph_entities`, `duration_s`, `skipped: bool`, `errors: list[str]`
- [ ] `IngestionResult` — mirrors v1's `IngestionResult`: `document_id`, `title`, `chunks_created`, `processing_time_ms`, `errors`

**Test gate:** `tests/integration/test_ingestion_pipeline.py`:
- Ingest `rag/documents/` sample docs (reuse v1's gold set)
- Verify chunks in DB with correct `corpus_id` and `tenant_id`
- Verify `contextualize()` output differs from raw chunk text (heading hierarchy is present)
- Verify incremental mode skips unchanged file on second run
- Verify graph extraction skipped when `enable_graph_extraction=False`
- Verify audio placeholder returned when ASR deps not installed
- Verify YAML frontmatter fields appear in chunk metadata

---

## Phase 5 — Retrieval Pipeline

> Gate: a query returns ranked, confidence-scored results from the vector store with all 3 cache layers wired.

### 5.1 Semantic Cache (`knowledge/retrieval/semantic_cache.py`)

- [ ] `async def lookup(query_emb, corpus_ids, threshold) → CachedAnswer | None`
- [ ] `async def store(query_text, query_emb, corpus_ids, answer) → None` — JWE-encrypt answer before insert
- [ ] Pruning: on `store()`, check row count; if > `semantic_cache_max_rows`, delete oldest 10%
- [ ] Prometheus counters: `cache_l3_hits_total`, `cache_l3_misses_total`, `cache_l3_similarity_score` histogram
- [ ] Write integration tests against live PG

### 5.2 CrossEncoder Reranker (`knowledge/retrieval/fusion.py`)

- [ ] Implement CrossEncoder reranker (reference v1's `rag/retrieval/rerankers.py` if it exists; otherwise implement from scratch using `sentence-transformers` `CrossEncoder` class — `BAAI/bge-reranker-base` is the default model)
- [ ] After reranking: `result.confidence = sigmoid(cross_encoder_logit)` for each result
- [ ] For standalone semantic search (no reranker): `confidence = raw_cosine_similarity`
- [ ] `RRF fusion`: `score = Σ 1/(60 + rank)` across all search legs; `raw_score_type = "rrf"`
- [ ] Confidence filter post-rerank: `[r for r in reranked if r.confidence >= settings.min_confidence_score]`
- [ ] Write unit tests: `tests/unit/test_fusion.py` — verify RRF math; verify confidence assignment; verify filter

### 5.3 Graph Retriever (`knowledge/retrieval/graph_retriever.py`)

- [ ] Port `NL→Cypher` logic from `kg/` (existing NL query code)
- [ ] `async def query(query_text, corpus_id) → list[SearchResult]`
- [ ] Wrap in circuit breaker (`CircuitBreaker("age_graph")`)
- [ ] On `CircuitOpenError`: return empty list (degrade to `no_graph` mode)

### 5.4 Retriever (`knowledge/retrieval/retriever.py`)

- [ ] `class Retriever`
- [ ] `async def retrieve(query, corpus_ids, k, tenant_id) → list[SearchResult]`:
  1. L2 Redis cache check (exact query hash)
  2. Embed query (L1 LRU hit likely)
  3. L3 semantic cache check
  4. `asyncio.gather(semantic_search, text_search, graph_retrieval [optional])`
  5. RRF fusion
  6. CrossEncoder rerank → populate `confidence`
  7. Confidence filter (`>= min_confidence_score`)
  8. Populate L2 Redis cache (async, non-blocking)
  9. Emit `POST_RETRIEVE` hook
- [ ] `async def retrieve_with_confidence(query, corpus_ids, k, tenant_id) → list[SearchResult]`:
  - Calls `retrieve()`
  - Computes `aggregate_confidence = sum(r.confidence for r in results[:k])`
  - If `aggregate_confidence < settings.retrieval_confidence_threshold` → return `[]` (Layer 1 gate)
- [ ] Prometheus: `cache_l2_hits_total`, `cache_l2_misses_total`, `retrieval_latency_seconds` histogram by stage

### 5.5 Retrieval Worker (`knowledge/retrieval/worker.py`)

- [ ] For async search requests (bulk/background batches)
- [ ] Consumes from `knowledge:search` stream
- [ ] Stores results in Redis hash for polling

**Test gate:** `tests/integration/test_retrieval_pipeline.py` — send a query against ingested data; verify:
- L2 cache miss on first call, hit on second identical call
- Confidence values populated on all results
- Low-confidence results filtered out
- Correct corpus isolation (results from wrong corpus_id never appear)

---

## Phase 6 — Agent & Confidence-Aware Pipeline

> Gate: the 3-layer gate orchestrator correctly abstains or answers; streaming SSE works end-to-end in the browser.
>
> **Architecture anchor:** copy `rag/agent/rag_agent.py` and `rag/api/app.py` as the starting point. The Pydantic AI patterns — `PydanticAgent` singleton, `RAGState` lazy init with `asyncio.Lock`, `@agent.tool` async functions, `contextvars.ContextVar` for per-coroutine tracing, `agent.run()` / `agent.run_stream()` — are all carried forward unchanged. Only the tool implementations and the structured output model change.

### 6.1 Agent Core (`knowledge/agent/agent.py`)

**What to copy directly from `rag/agent/rag_agent.py`:**
- `get_llm_model(model_choice)` — `OpenAIChatModel` + `OpenAIProvider`; reads `settings.llm_*`; Ollama `num_ctx` injected via `model_settings={"extra_body": {"num_ctx": ...}}` when `provider == "ollama"`
- `_trace_context: contextvars.ContextVar` — per-coroutine Langfuse trace ref; safe for concurrent requests
- `RAGState(BaseModel)` with `PrivateAttr` fields: `_store`, `_retriever`, `_initialized`, `_init_lock: asyncio.Lock`; lazy `get_retriever()` initialises in the current event loop
- `agent = PydanticAgent(get_llm_model(), system_prompt=..., model_settings=...)` — module-level singleton
- `traced_agent_run(query, user_id, session_id, message_history)` — wraps `agent.run()` with Langfuse trace; sets `_trace_context`; always calls `state.close()` in `finally`

**What changes in v2:**
- [ ] `RAGState` holds `corpus_ids: list[str]` and `tenant_id: str` (used by all tools for scoping)
- [ ] Agent output type is `GenerationResult` (structured, not raw string):
  ```python
  class CitationCheck(BaseModel):
      is_trustworthy: bool
      uncited_claims: list[str]

  class GenerationResult(BaseModel):
      answer: str
      citations: list[Citation]
      citation_check: CitationCheck
  ```
- [ ] System prompt always includes: *"Every factual statement MUST be supported by one of the provided source chunks, cited inline as [chunk_id]. If you cannot find a supporting chunk for a claim, omit that claim entirely."*
- [ ] Tools (see 6.1a below) return structured context strings scoped to `corpus_ids` + `tenant_id`
- [ ] `traced_agent_run` is used for the blocking `POST /v1/chat` route — identical call signature to v1

#### 6.1a Agent Tools

Port the tool signatures from v1 but scope all retrieval to `ctx.deps.corpus_ids` and `ctx.deps.tenant_id`:

- [ ] `search_knowledge_base(ctx, query, match_count, search_type, metadata_filters)` — use `Retriever` from `RAGState`; return formatted context string with `[chunk_id]` anchors so the LLM can cite
- [ ] `search_knowledge_graph(ctx, query, entity_type, limit)` — use `AgeGraphStore` from `RAGState`; corpus-scoped graph name
- [ ] `search_hybrid_kg(ctx, query, match_count)` — parallel `asyncio.gather(semantic, graph)` then fuse; same pattern as v1's `HybridKGRetriever`
- [ ] `run_graph_query(ctx, cypher)` — pass-through to `AgeGraphStore.run_cypher_query()`; read-only guard unchanged
- [ ] All tools: call `trace_tool_call()` via `_trace_context.get()` when Langfuse is enabled — copy the tracing block from v1 verbatim

### 6.2 Streaming (`knowledge/api/routes/chat.py`)

**Copy the streaming pattern from `rag/api/app.py` verbatim** — only the surrounding pipeline changes:

```python
# Non-streaming: POST /v1/chat
result = await traced_agent_run(query, user_id, session_id, message_history)

# Streaming: GET /v1/chat/stream (SSE)
async def _generate():
    state = RAGState(user_id=user_id, corpus_ids=corpus_ids, tenant_id=tenant_id)
    try:
        async with agent.run_stream(
            query,
            deps=state,
            message_history=message_history or [],
        ) as streamed:
            async for delta in streamed.stream_text(delta=True):
                yield f"data: {json.dumps({'delta': delta})}\n\n"
        # After stream ends, access the structured output and emit citations
        # streamed.output is a GenerationResult; .citations is list[Citation]
        citations = [c.model_dump() for c in streamed.output.citations]
        yield f"data: {json.dumps({'citations': citations, 'done': True})}\n\n"
    except Exception as exc:
        logger.exception("Stream error: %s", exc)
        yield f"data: {json.dumps({'error': 'Internal server error'})}\n\n"
    finally:
        await state.close()

return StreamingResponse(_generate(), media_type="text/event-stream")
```

- [ ] SSE event types: `{"delta": "<text>"}` for tokens, `{"citations": [...], "done": true}` on completion, `{"error": "..."}` on failure, `{"abstained": true, "layer": 1, "reason": "..."}` on pipeline abstention
- [ ] `message_history` is passed through unchanged — same multi-turn support as v1
- [ ] `StreamingResponse` + `media_type="text/event-stream"` — exact same FastAPI pattern as v1

### 6.3 Confidence-Aware Pipeline (`knowledge/agent/pipeline.py`)

Wraps `agent.run()` with the 3-layer gate. The streaming path (`agent.run_stream()`) bypasses the judge gate — streaming is only available on the standard answered path.

- [ ] `class ConfidenceAwarePipeline`
- [ ] `async def run(query, corpus_ids, tenant_id, model_tier, ...) → RAGResponse`:
  - Layer 1: `retrieve_with_confidence()` → if empty → `abstained_retrieval` (no LLM call)
  - Layer 2: `agent.run(query, deps=state)` → check `result.output.citation_check.is_trustworthy` → if `False` → `abstained_citation`
  - Layer 3: `judge(query, context, answer)` → `unsupported` or low `confidence` → `abstained_judge`; `partial` → append uncertainty note
  - On `answered`: populate L3 semantic cache (async, JWE-encrypted); emit `POST_LLM` hook
- [ ] `async def run_stream(query, corpus_ids, tenant_id, ...) → AsyncGenerator`: Layer 1 gate only (retrieval confidence check); if passes, delegate directly to `agent.run_stream()` — judge not called on streaming path (latency trade-off; judge runs offline via eval)
- [ ] `RAGResponse` fields: `answer`, `status`, `confidence`, `citations`, `low_confidence_warning`, `pipeline_latency_ms`, `abstention_layer`, `abstention_reason`
- [ ] Hook points: `PRE_LLM` → cost guard; `POST_RETRIEVE` / `POST_LLM` / `ON_VALIDATION_FAIL` per gate outcome

### 6.4 LLM Judge (`knowledge/agent/judge.py`)

- [ ] `async def judge(query, context, answer) → JudgeResult`
- [ ] Structured output: `JudgeResult(verdict: Literal["supported","partial","unsupported"], confidence: float, reasoning: str)`
- [ ] Uses `nano` tier (`PydanticAgent` with nano model); escalates to `small` tier if `result.output.confidence < 0.5`
- [ ] Circuit breaker wraps both LLM calls
- [ ] Write unit tests: mock both nano and small agents; verify escalation; verify all verdict paths

### 6.5 Model Router (`knowledge/agent/model_router.py`)

- [ ] `async def route(query: str) → RoutingDecision` — uses `nano` tier `PydanticAgent` with structured output
- [ ] 3s `asyncio.wait_for` timeout; on timeout default to `small`
- [ ] `RoutingDecision`: `complexity`, `requires_graph`, `requires_multipass`, `estimated_context_tokens`, `rejected`, `rejection_reason`

### 6.6 Cost Guard (`knowledge/agent/cost_guard.py`)

- [ ] `async def check_cost_circuit_breaker(tenant_id, model_id)` — Redis INCRBYFLOAT; raises `TenantBudgetExceeded` (→ 402) or `SystemBudgetExceeded` (→ 503)
- [ ] Called at `PRE_LLM` hook — before every `agent.run()` / `agent.run_stream()` call

**Test gate:** `tests/integration/test_agent.py` — requires Phase 4 (ingested data) to be complete first. Run confidence-aware pipeline against ingested NeuralFlow docs; verify `status="answered"` with citations on known-good query; verify `status="abstained_retrieval"` against empty corpus; verify streaming yields delta events in correct order.

---

## Phase 7 — Validation & Hooks

> Gate: all validation steps fire in correct order; hooks can intercept pipeline; bad queries are rejected before any DB or LLM call.
>
> **Note:** the Model Router (`knowledge/agent/model_router.py`) is implemented in Phase 6 because it is used inside the agent pipeline. Phase 7 covers only the input validation chain and hook system.

### 7.1 Hook System (`knowledge/hooks/`)

- [ ] `HookRegistry`, `HookPoint` enum (all 13 points), `Hook` type alias
- [ ] `HookContext` dataclass: query, corpus_id, user_id, routing_decision, retrieved_chunks, llm_response, error
- [ ] `fire(point, ctx) → ctx` — runs hooks in priority order; stops on `HookAbort`
- [ ] Register placeholder hooks at app startup: `audit_log_hook`, `pii_redact_hook`, `response_filter_hook`, `metrics_hook`
- [ ] Write unit tests: `tests/unit/test_hooks.py` — verify priority ordering; verify `HookAbort` stops chain

### 7.2 Validation Pipeline (`knowledge/validation/pipeline.py`)

- [ ] V1: Pydantic schema validation (handled by FastAPI automatically; this step fires the hook)
- [ ] V2: Length guard — reject if `len(query) > settings.max_query_chars`
- [ ] V3: Language detection (optional; skip if `allowed_languages` is `*`)
- [ ] V4: Prompt injection detector — regex patterns + embedding-similarity against known attack patterns
- [ ] V5: Content policy check — nano model, structured output `ContentPolicyResult(verdict, confidence, reason)`
- [ ] V6: Corpus access check — JWT roles vs. `CorpusConfig.allowed_roles`; checked before any DB I/O
- [ ] Write unit tests: `tests/unit/test_validation.py` — test each layer independently; stub V5 model

**Test gate:** end-to-end request through validation + hook + router returns correct 422/400/403 for each failure mode.

---

## Phase 8 — API Layer

> Gate: all REST endpoints return correct status codes and response envelopes; streaming SSE works in browser.

### 8.1 FastAPI App Factory (`knowledge/api/app.py`)

- [ ] Lifespan: on startup — connect stores, create consumer groups, load CorpusRegistry, register hooks, start APScheduler
- [ ] Lifespan: on shutdown — drain connections, close Redis pool, stop workers gracefully
- [ ] Middleware stack (in order): CorrelationID → StructuredLog → AuditEmitter → CORS → RateLimiter

### 8.2 Request/Response Schemas (`knowledge/api/schemas.py`)

- [ ] `ChatRequest`: `query`, `corpus_ids`, `model_tier`, `stream`, `session_id: str` (required — generated by the frontend per conversation; UUID format; used for Langfuse tracing, audit log, log correlation, and multi-turn history grouping), `message_history: list | None`
- [ ] `SearchRequest`: `query`, `corpus_ids`, `k`, `metadata_filter`
- [ ] `IngestRequest`: `corpus_id`, `source_path | source_url`, `enable_graph_extraction`, `mode`
- [ ] `ScheduledJobRequest`: `name`, `source_config`, `corpus_id`, `cron_expr`, `mode`
- [ ] `APIResponse[T]`: `request_id`, `data: T | None`, `error: ErrorDetail | None`, `cache_hit`
- [ ] `ErrorDetail`: `code`, `message`, `details`, `retry_after_s`, `doc_url`
- [ ] All versioned; breaking changes get `V2Request` suffix

### 8.3 Routes

- [ ] **`routes/ingest.py`**:
  - `POST /v1/ingest` → validate request, publish `IngestJob` → return `{job_id}`
  - `GET /v1/ingest/{job_id}/status` → `HGETALL job:{job_id}` → return structured status
  - `GET /v1/ingest/{job_id}/stream` → SSE subscription to `knowledge:events` filtered by `job_id`
- [ ] **`routes/search.py`**:
  - `POST /v1/search` → run retriever directly (sync fast-path) → return `SearchResult[]` with citations
- [ ] **`routes/auth.py`** (new route file):
  - `POST /v1/auth/token` → accept `{email, password}` or API key; return signed JWT (access token 15-min TTL + refresh token 7-day TTL)
  - `POST /v1/auth/refresh` → accept refresh token; return new access token; rotate refresh token
- [ ] **`routes/chat.py`**:
  - `POST /v1/chat` → run `ConfidenceAwarePipeline` (blocking) → return `RAGResponse`
  - `POST /v1/chat/stream` → SSE streaming; body is same `ChatRequest` JSON; `Content-Type: text/event-stream`; yield token deltas then citations+done event. **Use POST, not GET** — SSE with GET cannot carry a JSON body; query params are insufficient for corpus_ids + message_history
- [ ] **`routes/corpus.py`**:
  - `GET /v1/corpus` → list corpora from `CorpusRegistry` (filtered by JWT roles)
  - `POST /v1/corpus/{id}/cache/invalidate` → flush L2+L3 for corpus
  - `GET /v1/corpus/{id}/ontology` → return current ontology Python source for corpus (admin)
  - `POST /v1/corpus/{id}/ontology` → upload new Python ontology file (admin); validate it contains a root `BaseModel` subclass and the `edge()` helper; save to `knowledge/corpus/ontologies/{corpus_id}.py`; clear `load_ontology` LRU cache
  - `DELETE /v1/corpus/{id}/ontology` → remove custom ontology; revert to generic default
- [ ] **`routes/evaluate.py`**:
  - `POST /v1/evaluate/run` → publish `EvalJob`; return `run_id`
  - `GET /v1/evaluate/run/{id}` → poll status + aggregated metrics
  - `GET /v1/evaluate/run/{id}/results` → paginated per-sample results
  - `GET /v1/evaluate/compare?a={id}&b={id}` → regression diff
- [ ] **`routes/feedback.py`**:
  - `POST /v1/feedback` → insert `UserFeedback` (background task)
  - `POST /v1/signals` → insert `ImplicitSignal` (service token only)
- [ ] **`routes/scheduler.py`**:
  - `GET /v1/scheduler/jobs` → list scheduled jobs for tenant
  - `POST /v1/scheduler/jobs` → create new scheduled ingest job
  - `PATCH /v1/scheduler/jobs/{id}` → update trigger or source config
  - `DELETE /v1/scheduler/jobs/{id}` → cancel and remove job
  - `POST /v1/scheduler/jobs/{id}/run-now` → immediate one-off trigger
- [ ] **`routes/logs.py`** (admin only):
  - `GET /v1/logs` → query Redis ring buffer `knowledge:logs:recent`; filter by `level`, `service`, `corpus_id`, `request_id`, `since`, `limit`; return JSON array newest-first; each entry includes `trace_url` when available
  - On-demand only — no streaming; clients fetch on page load and on explicit refresh
- [ ] **`routes/health.py`**:
  - `GET /health` → check PG pool, Redis ping, worker heartbeats, DLQ depth, circuit breaker states
  - `GET /metrics` → Prometheus text format (service token auth)
- [ ] Write mocked API tests: `tests/integration/test_api.py` — all routes, all error codes, all SSE events

**Test gate:** `tests/integration/test_api.py` — 100% of routes tested; 0 unhandled exceptions.

---

## Phase 9 — Security Layer

> Gate: JWT auth rejects invalid tokens; RBAC denies cross-corpus access; JWE round-trip works; rate limiting returns 429.

### 9.1 JWT Auth (`knowledge/api/auth.py`)

- [ ] `async def require_jwt(token: str) → TokenClaims` — FastAPI dependency
- [ ] RS256 verification; JWKS fetched from issuer + cached in process for `jwks_cache_ttl_s`
- [ ] Extract `sub`, `roles: list[str]`, `tenant_id` from claims
- [ ] `async def check_corpus_access(tenant_id, corpus_id, role, registry)` — RBAC check
- [ ] Write unit tests: `tests/unit/test_auth.py` — expired token, wrong algorithm, missing role

### 9.2 JWE Helpers (`knowledge/api/auth.py`)

- [ ] `encrypt_answer(plaintext: str, tenant_id: str) → str` — JWE compact serialized
- [ ] `decrypt_answer(jwe: str, tenant_id: str) → str`
- [ ] Use `joserfc`; ECDH-ES+A256KW / A256GCM
- [ ] Per-tenant keys loaded from Secrets Manager (or env in local dev)

### 9.3 Middleware (`knowledge/api/middleware.py`)

- [ ] `CorrelationIDMiddleware` — set `X-Request-ID` header; inject into `contextvars` for structured logging
- [ ] `AuditEmitter` — background task: `INSERT INTO audit_events` after every authenticated request
- [ ] Request logs: `{"level": "INFO", "request_id": "...", "corpus_id": "...", "user_id": "...", "latency_ms": ...}` (structlog)

### 9.4 Rate Limiting (`knowledge/api/quota.py`)

- [ ] Implement `enforce_quota()` as specified in the SaaS Deployment Model section of `RAGV2_DESIGN.md` — Redis INCR counters with 25h TTL; RPM sliding window via 2-minute key expiry
- [ ] Add `slowapi` for inbound rate limiting by JWT `sub`; return 429 + `Retry-After`
- [ ] Quota headers on every response: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, `X-Quota-Daily-Limit`, `X-Quota-Daily-Used`
- [ ] Write unit tests: `tests/unit/test_quota.py` — verify DAILY_QUOTA_EXCEEDED, RATE_LIMIT_EXCEEDED, LLM_NOT_ENABLED_ON_FREE_TIER

**Test gate:** `tests/unit/test_auth.py` + `tests/unit/test_quota.py` — all error paths handled; no 500s for auth/quota failures.

---

## Phase 10 — Ingestion Scheduler

> Gate: a scheduled job fires at its configured cron time, triggers an ingestion job, and updates `next_run_at`; incremental mode skips unchanged files.

### 10.1 Job Store (`knowledge/scheduler/job_store.py`)

- [ ] `ScheduledJob` Pydantic model: `id`, `tenant_id`, `name`, `source_type` (local/url/s3/gcs), `source_config`, `corpus_id`, `cron_expr`, `mode` (full/incremental), `enable_graph_extraction`, `next_run_at`, `last_run_at`, `last_status`, `is_active`
- [ ] CRUD: `create`, `get`, `list_by_tenant`, `update`, `delete`, `get_due_jobs` (WHERE `next_run_at <= NOW() AND is_active`)
- [ ] `compute_next_run_at(cron_expr) → datetime` — using `croniter`

### 10.2 Scheduler Runner (`knowledge/scheduler/runner.py`)

- [ ] Use `APScheduler` (AsyncIOScheduler) — start in FastAPI lifespan
- [ ] Single tick job: every 60s, call `job_store.get_due_jobs()` → for each due job, publish `IngestJob` to Redis stream → update `next_run_at`
- [ ] Respects `scheduler_max_concurrent_jobs` — checks in-flight count via `job:{id}:status` hashes
- [ ] Incremental mode: worker checks fingerprint cache before processing each file in source folder
- [ ] On startup: reload all active scheduled jobs into APScheduler from DB (handles restarts)
- [ ] `POST /v1/scheduler/jobs/{id}/run-now` → bypass `next_run_at` and publish immediately

### 10.3 Source Adapters

- [ ] `LocalFolderSource` — scan directory recursively for supported file types; emit file paths
- [ ] `URLSource` — single URL download to temp file; emit temp path
- [ ] `S3Source` — list bucket prefix; download changed objects (compare S3 ETag to fingerprint cache)
- [ ] `GCSSource` — same pattern as S3 using `google-cloud-storage` (add `google-cloud-storage>=2.0` to a new `scheduler-cloud` optional extra in `pyproject.toml`; also add `boto3` for S3Source to the same extra)

**Test gate:** `tests/unit/test_scheduler.py` — `compute_next_run_at` correct for various cron expressions; `get_due_jobs` only returns jobs past their `next_run_at`; `run-now` publishes to Redis stream.

---

## Phase 11 — Observability

> Gate: Prometheus scrape returns all defined metrics; Langfuse trace appears for each LLM call; alert email sends on circuit open.

### 11.1 Prometheus Metrics (`knowledge/observability/metrics.py`)

- [ ] Define all counters/histograms/gauges from the design (cache hits, latency by stage, token usage, cost, feedback, storage)
- [ ] Instrument: cache layer (L1/L2/L3), retrieval stages, model tier selection, circuit breaker state, DLQ depth, quota enforcement
- [ ] Expose `/metrics` via `GET /metrics` route (Prometheus text format)
- [ ] `pg_exporter` and `redis_exporter` Docker services are defined in Phase 13.3 — no action needed here

### 11.2 Langfuse Tracing (`knowledge/observability/langfuse.py`)

- [ ] `@trace_llm_call(span_name)` decorator for all agent + judge + router LLM calls
- [ ] Capture: model_id, prompt_tokens, completion_tokens, latency_ms, corpus_id, tenant_id
- [ ] Emit to Langfuse only when `settings.langfuse_enabled = True`; no-op otherwise (no-op decorator)

### 11.3 Alert System (`knowledge/observability/alerts.py`)

- [ ] `async def send_alert(severity, code, detail) → None` — SMTP send via `aiosmtplib`
- [ ] Non-blocking: always wrapped in `asyncio.create_task`
- [ ] Fallback: if SMTP unreachable → write to `logs/alerts.jsonl` + stderr
- [ ] Email template: Subject `[RAG] {severity} — {code}`, Body: time, corpus, tenant, request_id, trace_url
- [ ] Register at: circuit breaker OPEN transition, DLQ push, system budget breach, P99 latency breach

### 11.4 Grafana Dashboard (`infra/grafana/dashboards/rag_v2.json`)

- [ ] Pre-build 7-row dashboard JSON (export from Grafana UI after wiring Prometheus):
  - Row 1: Retrieval Quality Trends
  - Row 2: Generation Quality (faithfulness, abstention rate)
  - Row 3: Answer Correctness
  - Row 4: Latency Breakdown heatmap + SLA compliance stat
  - Row 5: Cost (token usage bar + daily cost line + cost/1K queries)
  - Row 6: Online Feedback (satisfaction score, tag distribution, lowest-rated traces)
  - Row 7: Storage (PG table bytes area chart, Redis memory gauge)

**Test gate:** `docker compose --profile observability up` → Prometheus scrape returns 200; Grafana dashboard loads; send a test query → trace appears in Langfuse.

---

## Phase 12 — Evaluation System

> Gate: offline eval run completes end-to-end; regression detection fires correctly; CI blocks on metric regression.

### 12.1 Gold Dataset (`knowledge/evaluation/datasets.py` + `data/`)

- [ ] `GoldSample` Pydantic model as designed
- [ ] Extract existing gold samples from `rag/tests/retrieval/test_retrieval_metrics.py` (`GOLD_DATASET` list) and `rag/tests/retrieval/test_legal_retrieval.py` (`LEGAL_GOLD_DATASET` list) — they are Python lists, not JSONL files; script them to emit `.jsonl` files into `evaluation/data/`
- [ ] `GoldDataset.load(corpus_id)` — from JSONL file in `evaluation/data/` or DB
- [ ] `GoldDataset.save_to_db(conn)` — upsert into `gold_samples` table

### 12.2 Retrieval Metrics (`knowledge/evaluation/metrics/retrieval.py`)

- [ ] Port from `rag/tests/retrieval/test_retrieval_metrics.py`
- [ ] `hit_rate_at_k`, `mrr_at_k`, `ndcg_at_k`, `precision_at_k`, `recall_at_k`
- [ ] Add confidence distribution: `mean_confidence`, `min_confidence`, `low_confidence_flag`

### 12.3 Faithfulness (`knowledge/evaluation/metrics/faithfulness.py`)

- [ ] Claim decomposition via nano model → list of atomic claims
- [ ] Per-claim NLI verification via nano model → `supported: bool`
- [ ] `faithfulness = count(supported) / count(claims)`

### 12.4 Answer Relevance (`knowledge/evaluation/metrics/answer_relevance.py`)

- [ ] Generate 3 reverse questions via nano model
- [ ] Embed original query + reverse questions; cosine similarity mean

### 12.5 Answer Correctness (`knowledge/evaluation/metrics/correctness.py`)

- [ ] BLEU-4 (`nltk`), ROUGE-1/2/L (`rouge-score`), METEOR (`nltk`), BERTScore-F (`bert-score`), semantic similarity (cosine)

### 12.6 Performance Metrics (`knowledge/evaluation/metrics/performance.py`)

- [ ] Latency span recording (retrieval, rerank, LLM first token, generation, total)
- [ ] Token counting; `estimate_cost(model_id, prompt_tokens, completion_tokens)` with pricing table from design

### 12.7 Runner + Reporter (`knowledge/evaluation/runner.py`, `reporter.py`)

- [ ] `runner.py` — consumes from `knowledge:eval` stream; runs samples concurrently (semaphore-limited); inserts results; publishes `EvalCompleteEvent`
- [ ] `reporter.py` — `generate_report(run_id, baseline_run_id)`:
  - Compute deltas; compare to `REGRESSION_TOLERANCE` per metric
  - Store report JSON in `eval_runs.report_json`
  - Generate Markdown summary (for GitHub PR comment)
  - Emit `eval_metric{metric, corpus, run_id}` Prometheus gauges
- [ ] `pipeline_status`, `abstention_layer`, confidence distribution fields in `EvalResult`

### 12.8 CI Integration

- [ ] `eval-worker` Docker service is already defined in Phase 13.2 — no action needed here
- [ ] Add GitHub Actions step: offline eval → `--fail-on-regression` → block merge

**Test gate:** trigger eval run via API against ingested gold dataset; `reporter.py` outputs Markdown with correct delta calculations.

---

## Phase 13 — Docker Compose & Infra

> Gate: `docker compose up` brings up all core services; `docker compose --profile observability up` adds monitoring.

### 13.1 Dockerfile (`backend/Dockerfile`)

Multi-stage build — one file, three runnable targets: `api`, `ingest-worker`, `retrieval-worker` (and `eval-worker`). Workers share the same base layer as the API but have a different CMD and a tighter extras set.

```dockerfile
# ── Stage 1: base — Python + uv ──────────────────────────────────────────────
FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1
RUN pip install --no-cache-dir uv
WORKDIR /app

# ── Stage 2: api-deps — install API extras ────────────────────────────────────
FROM base AS api-deps
COPY pyproject.toml uv.lock ./
# reranker + observability + mcp; no audio/ui
RUN uv sync --extra ingestion --extra observability --extra reranker --extra mcp --no-dev

# ── Stage 3: worker-deps — install worker extras ──────────────────────────────
FROM base AS worker-deps
COPY pyproject.toml uv.lock ./
# audio needed for Whisper ASR on ingest workers
RUN uv sync --extra ingestion --extra audio --extra observability --no-dev

# ── Stage 4: api — production API image ──────────────────────────────────────
FROM api-deps AS api
COPY knowledge/ ./knowledge/
# gunicorn manages worker processes; each runs UvicornWorker (async, handles SSE)
# --timeout 0: disable worker timeout for SSE long-poll routes (Nginx handles upstream timeout)
# --workers: set via GUNICORN_WORKERS env (default 2 for < 4 vCPU; 2×vCPU for larger)
ENV GUNICORN_WORKERS=2 \
    GUNICORN_TIMEOUT=0 \
    GUNICORN_GRACEFUL_TIMEOUT=30 \
    GUNICORN_KEEPALIVE=5
EXPOSE 8000
CMD gunicorn knowledge.api.app:app \
      -k uvicorn.workers.UvicornWorker \
      --workers ${GUNICORN_WORKERS} \
      --bind 0.0.0.0:8000 \
      --timeout ${GUNICORN_TIMEOUT} \
      --graceful-timeout ${GUNICORN_GRACEFUL_TIMEOUT} \
      --keep-alive ${GUNICORN_KEEPALIVE} \
      --access-logfile - \
      --error-logfile -

# ── Stage 5: ingest-worker ────────────────────────────────────────────────────
FROM worker-deps AS ingest-worker
COPY knowledge/ ./knowledge/
CMD ["python", "-m", "knowledge.ingestion.worker"]

# ── Stage 6: retrieval-worker ─────────────────────────────────────────────────
FROM worker-deps AS retrieval-worker
COPY knowledge/ ./knowledge/
CMD ["python", "-m", "knowledge.retrieval.worker"]

# ── Stage 7: eval-worker ──────────────────────────────────────────────────────
FROM worker-deps AS eval-worker
COPY knowledge/ ./knowledge/
CMD ["python", "-m", "knowledge.evaluation.runner"]
```

**Key decisions:**
- `--timeout 0` on Gunicorn disables the worker timeout. Without this, Gunicorn kills workers that hold open SSE connections beyond 30s. Nginx `proxy_read_timeout` is the real deadline.
- `GUNICORN_WORKERS` is an env var so it can be tuned per deployment size without rebuilding the image (set to `2×vCPU` in K8s via `valueFrom.resourceFieldRef`).
- Workers use the same `knowledge/` source copy; only the CMD and pip extras differ — no code duplication.
- Add `gunicorn` to `pyproject.toml` core deps (not an extra — every image needs it).

- [ ] Add `gunicorn>=23.0` to `pyproject.toml` core dependencies
- [ ] Verify `gunicorn` + `uvicorn.workers.UvicornWorker` round-trips SSE correctly in local smoke test

---

### 13.2 Docker Compose (`backend/docker-compose.yml`)

```yaml
services:

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./infra/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./infra/certs:/certs:ro
    depends_on: [api, frontend]
    restart: unless-stopped

  api:
    build:
      context: .
      target: api          # gunicorn + UvicornWorker
    env_file: .env
    environment:
      GUNICORN_WORKERS: "2"
    depends_on: [postgres, redis, ollama]
    restart: unless-stopped
    # no published port — nginx proxies to api:8000 on the internal network

  ingest-worker:
    build:
      context: .
      target: ingest-worker
    env_file: .env
    deploy:
      replicas: 2
    depends_on: [postgres, redis, ollama]
    restart: unless-stopped

  retrieval-worker:
    build:
      context: .
      target: retrieval-worker
    env_file: .env
    deploy:
      replicas: 2
    depends_on: [postgres, redis, ollama]
    restart: unless-stopped

  eval-worker:
    build:
      context: .
      target: eval-worker
    env_file: .env
    depends_on: [postgres, redis, ollama]
    restart: unless-stopped

  postgres:
    image: apache/age:latest          # pgvector + Apache AGE
    environment:
      POSTGRES_DB: ragv2
      POSTGRES_USER: ragv2
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: unless-stopped
    # no published port — only API and workers reach it internally

  redis:
    image: redis:7-alpine
    command: redis-server --save 60 1 --appendonly yes
    volumes:
      - redisdata:/data
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollamamodels:/root/.ollama
    ports:
      - "11434:11434"               # published for local model pulls
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  frontend:
    build:
      context: ../frontend
      target: runner
      args:
        NEXT_PUBLIC_API_BASE_URL: ""   # empty: nginx handles /api/v1 rewrite on same origin
    restart: unless-stopped
    # no published port — nginx proxies to frontend:3000

volumes:
  pgdata:
  redisdata:
  ollamamodels:
```

---

### 13.3 Docker Compose Observability (`backend/docker-compose.observability.yml`)

```yaml
# Extend base compose: docker compose -f docker-compose.yml -f docker-compose.observability.yml up

services:

  langfuse:
    image: langfuse/langfuse:latest
    environment:
      DATABASE_URL: postgresql://langfuse:${LANGFUSE_DB_PASSWORD}@langfuse-postgres:5432/langfuse
      NEXTAUTH_SECRET: ${LANGFUSE_NEXTAUTH_SECRET}
      SALT: ${LANGFUSE_SALT}
    depends_on: [langfuse-postgres]
    restart: unless-stopped

  langfuse-postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: langfuse
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: ${LANGFUSE_DB_PASSWORD}
    volumes:
      - langfuse_pgdata:/var/lib/postgresql/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./infra/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
    volumes:
      - ./infra/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana_data:/var/lib/grafana
    depends_on: [prometheus]
    restart: unless-stopped

  pg-exporter:
    image: quay.io/prometheuscommunity/postgres-exporter:latest
    environment:
      DATA_SOURCE_NAME: postgresql://ragv2:${POSTGRES_PASSWORD}@postgres:5432/ragv2?sslmode=disable
    depends_on: [postgres]
    restart: unless-stopped

  redis-exporter:
    image: oliver006/redis_exporter:latest
    environment:
      REDIS_ADDR: redis://redis:6379
    depends_on: [redis]
    restart: unless-stopped

volumes:
  langfuse_pgdata:
  grafana_data:
```

---

### 13.4 Nginx Config (`backend/infra/nginx/nginx.conf`)

```nginx
events { worker_connections 1024; }

http {
    upstream api    { server api:8000; }
    upstream frontend { server frontend:3000; }

    # HTTP → HTTPS redirect
    server {
        listen 80;
        return 301 https://$host$request_uri;
    }

    server {
        listen 443 ssl;
        ssl_certificate     /certs/cert.pem;
        ssl_certificate_key /certs/key.pem;
        ssl_protocols       TLSv1.3;
        ssl_ciphers         HIGH:!aNULL:!MD5;

        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

        # SSE routes — disable buffering, extend read timeout
        location ~ ^/api/v1/(chat/stream|ingest/[^/]+/stream) {
            proxy_pass         http://api;
            proxy_buffering    off;
            proxy_read_timeout 3600s;
            proxy_set_header   X-Request-ID $request_id;
            proxy_set_header   Host $host;
        }

        # All other API routes
        location /api/v1/ {
            proxy_pass         http://api;
            proxy_read_timeout 60s;
            proxy_set_header   X-Request-ID $request_id;
            proxy_set_header   Host $host;
        }

        # Frontend — everything else
        location / {
            proxy_pass         http://frontend;
            proxy_read_timeout 30s;
            proxy_set_header   Host $host;
        }
    }
}
```

---

### 13.5 Makefile targets

- [ ] `make dev` — `docker compose up --build`
- [ ] `make dev-obs` — `docker compose -f docker-compose.yml -f docker-compose.observability.yml up --build`
- [ ] `make migrate` — run all SQL files in `migrations/` in order against `DATABASE_URL`
- [ ] `make test` — `pytest tests/unit/ tests/integration/ -v`
- [ ] `make test-unit` — `pytest tests/unit/ -v` (no external deps)
- [ ] `make lint` — `ruff check --fix && ruff format knowledge/`
- [ ] `make pull-models` — `ollama pull llama3.2:3b nomic-embed-text qwen2.5:0.5b llama3.1:70b`
- [ ] `make chaos-kill-redis` / `make chaos-kill-ollama` / `make chaos-kill-postgres`

- [ ] Write `install.sh` (Linux/macOS) and `install.ps1` (Windows)
- [ ] Write `.env.example` with all required variables

**Test gate:** `make dev` → `curl -k https://localhost/health` returns `{"status": "healthy"}`.

---

## Phase 14 — Frontend

> Gate: chat UI works end-to-end in browser (SSE streaming, citations, feedback); ingestion panel submits jobs and shows SSE progress; scheduled jobs CRUD works.

### 14.1 Project Setup

- [ ] `cd frontend && npx create-next-app@latest . --typescript --tailwind --app --src-dir --eslint`
- [ ] Install: `zustand`, `@radix-ui/react-*` (dialog, tabs, tooltip, select), `react-markdown`, `remark-gfm`, `recharts` (charts), `react-dropzone` (file upload), `react-hot-toast` (notifications), `date-fns` (date formatting), `cronstrue` (human-readable cron display), `cron-parser` (next-run calculation), `lucide-react` (icons)
- [ ] Configure Tailwind dark mode: `darkMode: "class"`; add custom CSS variables for design tokens
- [ ] Set up `next.config.ts` with API proxy rewrites: `/api/v1 → backend:8000/api/v1`

### 14.2 Auth & API Client (`src/lib/`)

- [ ] `auth.ts` — login form → `POST /api/v1/auth/token` (see Phase 8 `routes/auth.py`) → store access token in memory + refresh token in httpOnly cookie; auto-refresh via `POST /api/v1/auth/refresh` on 401
- [ ] `api.ts` — typed fetch wrapper; adds `Authorization: Bearer <token>` header; maps `error.code` to typed errors; returns `APIResponse<T>.data` or throws
- [ ] `sse.ts` — `async function* streamSSE(url, init)` — reads `ReadableStream`; yields parsed SSE events by type

### 14.3 State Management (`src/store/chatStore.ts`)

- [ ] Zustand store: `conversations: Conversation[]`, `activeConversationId`, `selectedCorpusIds: string[]`, `modelTier: "auto" | "small" | "large"`, `isDark: boolean`
- [ ] `Conversation` type includes `session_id: string` — a UUID generated once when the conversation is created (`crypto.randomUUID()`), persisted in the store, and sent with every `ChatRequest` in that conversation. A new conversation always gets a new session_id. The `user_id` is extracted from the decoded JWT access token (`sub` claim) by `auth.ts` and injected by `api.ts` into every authenticated request header or body as needed.
- [ ] Actions: `sendMessage`, `appendToken`, `setCitations`, `setAbstention`, `setFeedback`, `newConversation` (generates new `session_id`), `loadConversation`

### 14.4 Chat Components

- [ ] `ChatShell.tsx` — full-page flex layout; mounts sidebar + message list + citation panel
- [ ] `MessageList.tsx` — virtualized scroll; auto-scroll on new messages; `aria-live="polite"`
- [ ] `MessageBubble.tsx` — user/assistant variant; renders markdown via `react-markdown` + `remark-gfm`; shows `PipelineStatusBadge` + `LowConfidenceWarning` on assistant messages
- [ ] `StreamingMessage.tsx` — renders in-progress tokens with blinking cursor; transitions to `MessageBubble` on stream end
- [ ] `CitationPanel.tsx` — slide-in panel; renders `CitationCard[]`; collapses to icon on tablet
- [ ] `CitationCard.tsx` — title, excerpt (truncated to 200 chars), `ConfidenceBadge`, link to source
- [ ] `ConfidenceBadge.tsx` — progress bar: green ≥ 0.7, amber 0.4–0.69, red < 0.4; shows numeric score
- [ ] `FeedbackBar.tsx` — thumbs buttons; on thumbs-down show tag picker; submit via `POST /v1/feedback`
- [ ] `CostBadge.tsx` — small inline badge below each assistant message: `$0.0007 · 1,637 tok · small · 843ms`; hidden when `estimated_cost_usd === 0` (local Ollama); always shown to `admin` role, hidden by default for `reader` role (toggle in settings)
- [ ] `DebugPanel.tsx` — collapsible panel below each assistant message (admin/dev mode only, toggled via `?debug=1` query param or user preference stored in `localStorage`):
  - **Pipeline latency breakdown**: bar chart of `pipeline_latency_ms` stages (retrieval / rerank / generation / judge)
  - **Token usage**: prompt tokens, completion tokens, estimated cost
  - **Model tier**: which tier the router selected and why (routing decision)
  - **Cache hit**: L2 / L3 / none
  - **Confidence scores**: aggregate retrieval confidence, judge verdict + confidence
  - **Trace link**: `[Open in Langfuse →]` button linking to `RAGResponse.trace_url`
  - **Request ID**: copyable UUID for log correlation
- [ ] `InputBar.tsx` — `<textarea>` auto-resize; `CorpusSelector` (multi-select combobox); `ModelTierPicker`; Enter to send; Shift+Enter for newline
- [ ] `useChat.ts` hook — manages `sendMessage`: `POST /v1/chat` or SSE streaming; handles abstention responses; extracts `estimated_cost_usd`, `pipeline_latency_ms`, `trace_url`, `request_id` from response for `CostBadge` and `DebugPanel`

### 14.5 Ingest Components

- [ ] `IngestPage.tsx` — three-tab layout
- [ ] `UploadDropzone.tsx` — `react-dropzone`; multiple files; show accepted MIME types; URL input fallback
- [ ] `JobStatusCard.tsx` — SSE-driven: subscribes to `/v1/ingest/{job_id}/stream`; updates progress bar token by token
- [ ] `SchedulerPanel.tsx` — `ScheduleList` + [+ New Job] button → `ScheduleForm` modal
- [ ] `ScheduleForm.tsx` — job name, source type radio, source path/URL, corpus selector, cron expression input with next-run preview (`cron-parser` computes next run, `cronstrue` renders it as human-readable text), mode (incremental/full), graph toggle; validates cron syntax before submit
- [ ] `useIngest.ts` — `submitJob(request) → job_id`; `watchJob(job_id)` via SSE; `listScheduledJobs()`; `createScheduledJob()`; `deleteScheduledJob()`
- [ ] `useScheduler.ts` — CRUD wrappers; optimistic update of local job list

### 14.6 Corpus Components

- [ ] `CorpusList.tsx` — fetch `/v1/corpus`; render `CorpusCard[]` in grid
- [ ] `CorpusCard.tsx` — display chunk count, last ingest, graph toggle, invalidate cache button
- [ ] `CorpusCreateModal.tsx` — name, source folders, graph toggle; submits to create flow
- [ ] `OntologyUploader.tsx` — visible only when graph extraction is enabled for the corpus:
  - Displays current ontology filename (or "Generic default" if none set)
  - File upload button accepting `.py` files only; rejects anything else
  - On upload: calls `POST /v1/corpus/{id}/ontology`; shows validation errors if the file is not a valid template (missing `edge()`, no root `BaseModel`, etc.)
  - [Remove] button calls `DELETE /v1/corpus/{id}/ontology` and reverts to generic
  - [View] button opens current ontology source in a read-only code modal (`<pre>` block)
  - Tooltip explains: "An ontology defines the entity types and relationships docling-graph will extract. Without one, a generic entity extractor is used."

### 14.7 Eval Components

- [ ] `EvalDashboard.tsx` — fetch latest run; render metric stats + trend charts (recharts LineChart)
- [ ] `RunTriggerForm.tsx` — corpus, model tier, k; submit → `POST /v1/evaluate/run`; poll status
- [ ] `MetricsTable.tsx` — compare current vs baseline; colour deltas green/red
- [ ] `RegressionDiff.tsx` — side-by-side diff of all metrics; highlight regressions
- [ ] `LatencyHeatmap.tsx` — recharts Heatmap of stage × percentile

### 14.8 Log Viewer (`/logs` — admin role only)

- [ ] `LogViewerPage.tsx` — full-page log browser; accessible at `/logs`; redirects non-admin users to `/chat`
- [ ] `LogFilterBar.tsx` — filter controls: level dropdown (DEBUG/INFO/WARNING/ERROR), service multi-select, corpus_id input, request_id search input, time range picker (last 1h / 6h / 24h); [Refresh] button
- [ ] `LogTable.tsx` — virtualized table of log entries (newest first); columns: timestamp, level (colour-coded pill), service, route, latency_ms, cost badge, user_id, session_id, status; click row to expand
- [ ] `LogEntryDetail.tsx` — expanded view of a single log entry: all fields as a formatted JSON block; [Open Trace →] button that links to `trace_url` (Langfuse) when present; [Copy Request ID] button
- [ ] `useLogViewer.ts` hook — calls `GET /api/v1/logs` with current filter state on mount and on [Refresh]; no polling (on-demand only); memoises last 500 entries in local state
- [ ] Add `logs/` to the app router pages

### 14.9 Admin Components

- [ ] `TenantTable.tsx` — list tenants; show tier badge, quota, billing status
- [ ] `QuotaEditor.tsx` — inline edit quota fields; `PATCH /v1/admin/tenants/{id}/quota`
- [ ] `BudgetGauge.tsx` — circular gauge: spent / limit; warning at 80%, error at 100%

### 14.9 UI Polish

- [ ] Dark/light mode toggle — `ThemeToggle.tsx` sets `<html class="dark">`; persists to `localStorage`
- [ ] Loading skeletons for all data-fetch components
- [ ] Error boundary per page section; `ErrorBanner` shows `error.code` + `error.message`
- [ ] `EmptyState` for no conversations, no corpus, no eval runs
- [ ] Toast notifications (`react-hot-toast`) for: job submitted, job completed, cache invalidated, feedback submitted
- [ ] Mobile responsive: sidebar hamburger, citation bottom sheet, input bar stacked layout

**Test gate:** spin up `make dev`; open browser → send chat message → SSE streaming works; submit a file for ingestion → progress bar updates live via SSE; create a scheduled job → appears in list with correct next-run time.

---

## Phase 15 — CI/CD & Cloud IaC

> Gate: GitHub Actions pipeline runs full test suite on every PR; staging deploy triggered on main merge; Helm chart deploys API + workers to K8s.

### 15.0 Frontend Deployment — Docker / Node.js (primary)

The frontend runs as a containerized Next.js Node.js server — same Docker Compose and K8s workflow as the backend. No platform lock-in.

#### Dockerfile (`frontend/Dockerfile`)

Multi-stage build: deps → builder → runner.

```dockerfile
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --frozen-lockfile

FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_PUBLIC_API_BASE_URL
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
```

Enable `output: "standalone"` in `next.config.ts` — produces a self-contained `server.js` with minimal node_modules; image stays small (~150 MB).

#### Local dev (Docker Compose)

- [ ] Add `frontend` service to `docker-compose.yml`:
  ```yaml
  frontend:
    build:
      context: ./frontend
      target: runner
      args:
        NEXT_PUBLIC_API_BASE_URL: http://api:8000
    ports: ["3000:3000"]
    depends_on: [api]
  ```
- [ ] Local dev without Docker: `cd frontend && npm run dev` — `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` in `.env.local`
- [ ] Hot reload in Docker: use `target: deps` + volume-mount source for development containers (optional; direct `npm run dev` is faster)

#### Nginx reverse proxy

- [ ] Add frontend upstream to `infra/nginx/nginx.conf`: proxy `app.ragv2.com/` → `frontend:3000`; proxy `/api/v1/` → `api:8000`
- [ ] Single Nginx entry point handles both; no CORS headers needed (same origin from browser's perspective)
- [ ] SSE routes (`/v1/chat/stream`, `/v1/ingest/{id}/stream`) need `proxy_buffering off; proxy_read_timeout 3600s` in Nginx

#### K8s Helm (cloud production)

- [ ] Add `frontend` Deployment to Helm chart; HPA on CPU (min 2, max 5 — frontend is stateless and light)
- [ ] `ConfigMap` for `NEXT_PUBLIC_API_BASE_URL` (internal service URL)
- [ ] Ingress rule: `app.ragv2.com` → `frontend:3000`; TLS via cert-manager

#### Environment variables

| Variable | Where set | Notes |
|----------|-----------|-------|
| `NEXT_PUBLIC_API_BASE_URL` | Build arg + env | Public; safe to expose; injected at build time |
| `JWT_SECRET` | Server-only `.env.local` / K8s secret | Never `NEXT_PUBLIC_*`; used only in Next.js API route for token refresh |
| `NEXTAUTH_URL` | Server-only | If using NextAuth.js for auth UI |

#### CI build

- [ ] `docker build --build-arg NEXT_PUBLIC_API_BASE_URL=$STAGING_API_URL -t rag-frontend:$SHA ./frontend`
- [ ] Push to ECR / Artifact Registry alongside backend images
- [ ] Staging deploy: `helm upgrade` updates `frontend` Deployment image tag

#### Test gate

- [ ] `npm run build` succeeds with zero type errors and zero ESLint errors
- [ ] `docker build` produces image < 200 MB
- [ ] `docker run -p 3000:3000 rag-frontend:latest` → `curl localhost:3000` returns HTML

---

### 15.1 GitHub Actions (`.github/workflows/`)

- [ ] `ci.yml` — on PR to `main`:
  1. `uv sync --extra all`
  2. `ruff check && mypy backend/knowledge/`
  3. `pytest tests/unit/ tests/integration/ -m "not integration"` (unit only in CI; integration needs services)
  4. `docker build --target api -t rag-api:pr-{sha} ./backend`
  5. `docker build --target ingest-worker -t rag-ingest-worker:pr-{sha} ./backend` and `docker build --target retrieval-worker -t rag-retrieval-worker:pr-{sha} ./backend`
  6. Offline eval against staging (optional gate — see Phase 12)
  7. Load regression: `locust --headless --users 5 --run-time 3m --fail-on-error-rate 0.01`
- [ ] `deploy-staging.yml` — on push to `main`: `helm upgrade --install --namespace staging`
- [ ] `deploy-prod.yml` — manual trigger with `environment: production` (requires approval)

### 15.2 Helm Chart (`infra/helm/rag-v2/`)

- [ ] Deployment: `api` (HPA on CPU; min 2, max 10)
- [ ] Deployment: `ingest-worker` (HPA on Redis stream depth; min 2, max 20)
- [ ] Deployment: `retrieval-worker` (HPA on Redis stream depth; min 2, max 10)
- [ ] Deployment: `eval-worker` (1 replica; no HPA)
- [ ] `ConfigMap`: non-secret settings from `values.yaml`
- [ ] `ExternalSecret`: DB password, JWT private key, SMTP password from Secrets Manager

### 15.3 Terraform Module (`infra/terraform/`)

- [ ] `aurora_postgres.tf` — Aurora PostgreSQL Multi-AZ + read replica; pgvector extension
- [ ] `elasticache_redis.tf` — 3-shard cluster mode; Multi-AZ
- [ ] `ecs_or_eks.tf` — placeholder; exact cloud TBD (EKS or ECS Fargate)
- [ ] `secrets_manager.tf` — store JWT private keys, SMTP credentials

**Test gate:** `helm lint infra/helm/rag-v2/` passes; CI workflow completes on a sample PR.

---

## Phase 16 — Load & Chaos Testing

> Gate: all Phase 1 SLA numbers are validated as measurements, not hypotheses. Every chaos scenario passes its acceptance criteria.

### 16.1 Locust Setup (`tests/load/locustfile.py`)

- [ ] `RAGUser` with tasks: search (weight 5), chat (weight 3), ingest_small_doc (weight 1)
- [ ] Use gold queries from evaluation dataset
- [ ] Parameterize JWT and staging URL via environment

### 16.2 Baseline Load Matrix

Run each scenario, record P50/P95/P99, commit CSV:

- [ ] Baseline search only — 1 RPS / 5 min → target: P95 < 600ms, 0% errors
- [ ] Baseline chat (small model) — 1 RPS / 5 min → target: P95 < 2000ms, 0% errors
- [ ] Ramp to breaking point — 1→20 RPS / 10 min → record RPS where error rate > 1%
- [ ] Sustained peak — 5 RPS / 30 min → P95 < 2000ms, error rate < 0.1%
- [ ] Burst — 0→15 RPS spike for 60s → recovery within 2 min; 0 DLQ entries
- [ ] Cache warmup — 1 RPS, 100 unique queries → L2 hit rate > 10% by end
- [ ] Cache cold — 5 RPS, 1000 unique queries → P95 < 2000ms (no cache)

### 16.3 Chaos Scenarios

For each: run at 3 RPS background; kill component; verify degraded mode header; restart component; verify recovery:

- [ ] Redis kill → `no_cache` mode; DB-backed rate limiting; no 500s
- [ ] Ollama kill → `search_only`; chat returns `503 LLM_CIRCUIT_OPEN`; circuit opens < 60s; alert email sent
- [ ] Ollama recovery → circuit OPEN→HALF-OPEN→CLOSED within 90s of restart
- [ ] PostgreSQL kill → `unavailable`; all 503s; no data corruption; jobs re-queue on recovery
- [ ] AGE graph kill → `no_graph`; vector+text path still returns results
- [ ] All ingest workers killed → queue grows; no data loss; jobs resume on worker restart

### 16.4 Resource Exhaustion Tests

- [ ] DB connection pool exhaustion — 20 RPS / 10 min; verify pool waiters visible in `/health`; no crashes
- [ ] Redis memory ceiling — fill semantic cache; pruning fires; no OOM
- [ ] Tenant budget exhaustion — exhaust Pro tier; chat returns 402; search continues; alert sent
- [ ] LLM context overflow — 50 queries with 8000+ token context; `context_truncated: true`; no 500s

### 16.5 Deliverables

- [ ] Commit `tests/load/results/baseline-{date}.md` (summary; not raw CSV)
- [ ] Grafana dashboard screenshot committed to `tests/load/results/grafana-{date}.png`
- [ ] All chaos acceptance criteria documented in `tests/load/results/chaos-results-{date}.md`

---

## Dependency Map (Phase Order)

```
Phase 0 (scaffold)
  └── Phase 1 (config + migrations)
        └── Phase 2 (storage)
              └── Phase 3 (message bus)
                    └── Phase 4 (ingestion)
                          └── Phase 5 (retrieval)
                                └── Phase 7 (validation + hooks)   ← hooks required by agent pipeline
                                      └── Phase 6 (agent + pipeline)
                                            └── Phase 8 (API)
                                                  ├── Phase 9 (security)
                                                  ├── Phase 10 (scheduler)
                                                  └── Phase 11 (observability)
                                                        └── Phase 12 (evaluation)
                                                              └── Phase 13 (docker + infra)
                                                                    └── Phase 14 (frontend)
                                                                          └── Phase 15 (CI/CD)
                                                                                └── Phase 16 (load testing)
```

> **Note on Phase 6 / Phase 7 order:** Phase 7 (hooks + validation) must be implemented before Phase 6 (agent) because the `ConfidenceAwarePipeline` fires `HookRegistry` at every gate. Build the hook scaffolding first — even with placeholder no-op hooks — so Phase 6 can wire into it without circular dependency.

---

## Checklist Summary

| Phase | Description | Estimated Effort |
|-------|-------------|-----------------|
| 0 | Housekeeping & scaffold | 0.5 day |
| 1 | Config + DB migrations | 1 day |
| 2 | Storage layer | 2 days |
| 3 | Message bus | 1 day |
| 4 | Ingestion pipeline | 2 days |
| 5 | Retrieval pipeline | 2 days |
| 6 | Agent + confidence-aware pipeline | 2 days |
| 7 | Validation + hooks + model router | 1 day |
| 8 | API layer | 2 days |
| 9 | Security layer | 1.5 days |
| 10 | Ingestion scheduler | 1 day |
| 11 | Observability | 1 day |
| 12 | Evaluation system | 2 days |
| 13 | Docker Compose + infra | 1 day |
| 14 | Frontend (Next.js + Tailwind) | 5 days |
| 15 | CI/CD + Cloud IaC | 1.5 days |
| 16 | Load & chaos testing | 2 days |
| **Total** | | **~29 days** |

> Estimates assume solo developer, all design decisions already made (they are), and no rework. Phases 2–8 are the critical path — front-load them.
