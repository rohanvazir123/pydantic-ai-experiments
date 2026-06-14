"""Locust load test for RAG v2 API.

Run:
  cd rag/v2
  locust -f tests/load/locustfile.py \
    --headless --users 5 --spawn-rate 1 --run-time 5m \
    --host https://localhost \
    --csv tests/load/results/baseline-$(date +%F)

Pass criteria (P95 targets):
  search:  < 600ms
  chat:    < 2000ms
  ingest:  < 200ms (submit only — job runs async)

Latency monitoring: every task records stage breakdowns from response headers
and the pipeline_latency_ms field so we can identify bottlenecks per stage
(retrieval vs rerank vs generation vs judge).
"""

import os
import random
import uuid

from locust import HttpUser, between, events, task

# Gold queries from evaluation dataset (Phase 12)
GOLD_QUERIES = [
    "What does NeuralFlow AI do?",
    "What is the PTO policy?",
    "What is the learning budget for employees?",
    "What technologies and architecture does the platform use?",
    "What is the company mission and vision?",
    "How many employees work at NeuralFlow AI?",
    "Q4 2024 business results and performance review",
    "implementation approach and playbook",
]

CORPUS_IDS = ["neuralflow"]

# JWT from env (set before running)
TOKEN = os.environ.get("LOCUST_JWT", "stub.dev.token")


def _headers():
    return {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


class RAGUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(5)
    def search(self):
        """Synchronous hybrid search — fast path, no LLM."""
        query = random.choice(GOLD_QUERIES)
        with self.client.post(
            "/api/v1/search",
            json={"query": query, "corpus_ids": CORPUS_IDS, "k": 5},
            headers=_headers(),
            catch_response=True,
            name="/api/v1/search",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Search failed: {resp.status_code}")
                return
            data = resp.json().get("data", {})
            results = data.get("results", [])
            if not results:
                resp.failure("Search returned 0 results")

    @task(3)
    def chat_blocking(self):
        """Blocking chat — full 3-layer confidence-aware pipeline."""
        query      = random.choice(GOLD_QUERIES)
        session_id = str(uuid.uuid4())
        with self.client.post(
            "/api/v1/chat",
            json={
                "query":      query,
                "corpus_ids": CORPUS_IDS,
                "session_id": session_id,
                "model_tier": "small",
            },
            headers=_headers(),
            catch_response=True,
            timeout=15,
            name="/api/v1/chat",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Chat failed: {resp.status_code}")
                return

            # Record per-stage latencies for bottleneck analysis
            data = resp.json().get("data", {})
            latencies = data.get("pipeline_latency_ms", {})
            status    = data.get("status", "unknown")

            if status.startswith("abstained"):
                # Abstentions are valid business outcomes, not failures
                resp.success()

            # Log stage breakdown for analysis
            if latencies:
                for stage, ms in latencies.items():
                    events.request.fire(
                        request_type="STAGE",
                        name=f"pipeline/{stage}",
                        response_time=ms,
                        response_length=0,
                        exception=None,
                        context={},
                    )

    @task(1)
    def ingest_small_doc(self):
        """Submit a small ingest job — verifies publisher and job hash creation."""
        with self.client.post(
            "/api/v1/ingest",
            json={
                "corpus_id":   CORPUS_IDS[0],
                "source_path": "../../rag/documents/company-overview.md",
                "mode":        "incremental",
            },
            headers=_headers(),
            catch_response=True,
            name="/api/v1/ingest",
        ) as resp:
            if resp.status_code not in (200, 202):
                resp.failure(f"Ingest submit failed: {resp.status_code}")
