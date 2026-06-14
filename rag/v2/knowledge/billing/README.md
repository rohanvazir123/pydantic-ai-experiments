# knowledge/billing/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [Quota Enforcement](#quota-enforcement)

---

## What This Is

Tenant lifecycle and billing. Provisions new tenants, enforces per-tenant quotas and LLM budgets, emits billing events for each LLM call, and handles GDPR data erasure.

---

## Files

| File | Purpose |
|------|---------|
| `provisioner.py` | `TenantProvisioner`: onboard (create tenant + quotas + default corpus + JWT keypair), offboard (cascade delete + GDPR), user-level right to erasure |
| `metering.py` | `BillingEvent` emit after each LLM call; nightly Stripe flush cron |

---

## Quota Enforcement

Quotas are enforced in Redis on the hot path — not in PostgreSQL. Redis is the fast guard; PostgreSQL is the audit trail.

```python
# Fires at PRE_VALIDATE hook
async def enforce_quota(tenant_id: str, request_type: str) -> None:
    # INCR daily counter, INCR RPM sliding-window counter
    # Raises QuotaExceeded on breach → 429 response
    ...
```

Budget headers on every response:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 47
X-Quota-Daily-Limit: 10000
X-Quota-Daily-Used: 3241
```

Tenants: `free` (search-only, 500 queries/day), `pro` ($299/mo, LLM enabled), `enterprise` (custom). See `RAGV2_DESIGN.md §SaaS Deployment Model` for full tier definitions.
