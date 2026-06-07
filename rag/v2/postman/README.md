# postman/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [How to Use](#how-to-use)
- [Auto-capture Scripts](#auto-capture-scripts)

---

## What This Is

Postman collection for manual API testing without a UI. Import the collection and environment into Postman, set your `baseUrl`, log in once to capture the JWT, then run requests in any order.

---

## Files

| File | Purpose |
|------|---------|
| `RAG_v2.postman_collection.json` | All API endpoints grouped by feature area |
| `RAG_v2_local.postman_environment.json` | Local dev environment variables |

---

## How to Use

1. Open Postman → **Import** → select both JSON files
2. Select the **RAG v2 — local dev** environment from the environment dropdown
3. Run **Auth → POST /auth/token** — the test script auto-saves the JWT to the `token` variable
4. Run any other request — `Authorization: Bearer {{token}}` is applied automatically

---

## Auto-capture Scripts

The collection includes Postman test scripts on key requests that automatically set collection variables:

| Request | Variable set |
|---------|-------------|
| `POST /auth/token` | `token` ← `response.data.access_token` |
| `POST /ingest` | `job_id` ← `response.data.job_id` |

After submitting an ingest job, immediately run **GET /ingest/{{job_id}}/status** to poll progress — no manual copy-paste needed.
