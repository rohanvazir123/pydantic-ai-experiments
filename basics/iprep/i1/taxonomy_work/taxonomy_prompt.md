You are helping design an explainable taxonomy for a B2B SaaS transcript intelligence dataset.

Context:
The product analyzes meeting transcripts from support calls, external customer calls, and internal company calls.
The assignment requires:
1. Categorizing transcripts by topic or theme.
2. Generating sentiment analysis across call types.
3. Explaining trends in a way useful to product, support, sales, customer success, and engineering leaders.

Input:
I am providing observed topic tags extracted from meeting summaries, plus key moment signal types.

Task:
Create a compact taxonomy for classifying meetings.

Requirements:
- Create 6 to 10 high-level business themes.
- Themes should be useful for executive/product/support/sales/engineering analysis.
- Do not create one category per raw topic.
- Group semantically similar topics under broader themes.
- Use the counts to prioritize categories that explain the most meetings.
- Do not invent categories unsupported by the observed topic tags.
- If a category is based on low-frequency tags, mark it as lower confidence in taxonomy_notes.
- For each theme, provide theme_name, business_meaning, stakeholder_who_cares, keyword_hints, example_raw_topics, and common_risks_or_opportunities.
- Also create a call_type inference scheme with 4 to 7 call types.
- For each call type, provide call_type, meaning, keyword_hints, example_raw_topics, and ambiguity_notes.
- Prefer explainability over perfection.
- Avoid overly narrow keywords unless they strongly indicate the category.

Important:
The output will be reviewed by a human and then saved as taxonomy.json for deterministic code.
Make keyword_hints concise lowercase strings suitable for simple substring matching.

Return JSON only with this structure:

{
  "themes": [
    {
      "theme_name": "...",
      "business_meaning": "...",
      "stakeholder_who_cares": ["..."],
      "keyword_hints": ["..."],
      "example_raw_topics": ["..."],
      "common_risks_or_opportunities": ["..."]
    }
  ],
  "call_types": [
    {
      "call_type": "...",
      "meaning": "...",
      "keyword_hints": ["..."],
      "example_raw_topics": ["..."],
      "ambiguity_notes": "..."
    }
  ],
  "taxonomy_notes": ["..."],
  "suggested_review_steps": ["..."]
}

Observed topic tags with counts:
- compliance: 23
- compliance reporting: 19
- renewal: 17
- churn risk: 14
- onboarding: 9
- incident response: 8
- outage: 8
- incident communication: 7
- product launch: 7
- feature request: 6
- platform outage: 6
- identity management: 5
- sprint planning: 5
- pricing: 5
- service outage: 5
- multi-framework support: 5
- outage post-mortem: 5
- product roadmap: 5
- infrastructure reliability: 4
- billing dispute: 4
- audit preparation: 4
- technical issue: 4
- workaround: 4
- product demo: 4
- roadmap planning: 4
- product bug: 4
- hipaa compliance: 4
- pci dss: 4
- incident review: 4
- support response time: 4
- customer communication: 3
- healthcare: 3
- competitive threat: 3
- customer retention: 3
- competitive evaluation: 3
- sla breach: 3
- hipaa: 3
- backup and recovery: 3
- soc 2: 3
- ot security: 3
- sprint retrospective: 3
- false positives: 3
- data retention: 3
- alert fatigue: 3
- product feedback: 3
- siem integration: 3
- platform reliability: 3
- pipeline reliability: 2
- iso 27001: 2
- soc 2 audit: 2
- backup performance: 2
- escalation: 2
- overage charges: 2
- contract negotiation: 2
- launch readiness: 2
- product outage: 2
- sla compliance: 2
- customer escalation: 2
- competitive displacement: 2
- post-mortem: 2
- identity & access management: 2
- alert latency: 2
- pipeline failure: 2
- technical debt: 2
- logvault integration: 2
- backup failure: 2
- roadmap prioritization: 2
- renewal risk: 2
- outage follow-up: 2
- scim provisioning: 2
- financial services: 2
- circuit breaker: 2
- competitor evaluation: 2
- product update: 2
- audit workflow: 2
- data backfill: 2
- threat monitoring: 2
- feature gap: 2
- soc 2 compliance: 2
- identity access management: 2
- reliability engineering: 2
- multi-framework compliance: 2
- incident communication failure: 2
- early access program: 2
- pci dss compliance: 2
- outage remediation: 1
- security monitoring: 1
- postmortem planning: 1
- seat overage: 1
- migration cleanup: 1
- billing adjustment: 1
- provisioning: 1
- kafka configuration: 1
- comply v2 launch: 1
- resource allocation: 1
- timeline risk: 1
- multi-framework coverage: 1
- cloud expansion: 1
- access control: 1
- change management: 1
- incident tracking: 1
- vendor management: 1
- security training: 1
- agent version update: 1
- notification gap: 1
- product add-on: 1
- api rate limiting: 1
- documentation: 1
- incident impact: 1
- vendor review: 1
- internal monitoring failure: 1
- granular restore: 1
- rto compliance: 1
- aws ebs: 1
- compliance automation: 1
- platform integration: 1
- threat monitoring failure: 1
- incident escalation: 1
- proactive communication failure: 1
- scim / provisioning: 1
- adaptive mfa: 1
- compliance integration: 1
- competitive landscape: 1
- service credits: 1
- sla review: 1
- data loss: 1
- pdf rendering: 1
- new release issue: 1
- audit deadline: 1
- proof of concept: 1
- audit automation: 1
- aws s3 connector: 1
- 403 error: 1
- iam permissions: 1
- service control policies: 1
- connector configuration: 1
- product visibility: 1
- product upsell: 1
- estimation: 1
- comply integration: 1
- api rate limits: 1
- alert tuning: 1
- alert routing: 1
- post-mortem follow-up: 1
- compliance bug: 1
- load testing: 1
- runbook: 1
- win-loss analysis: 1
- competitive intelligence: 1
- reliability / outage impact: 1
- sales process: 1
- collateral gaps: 1
- q1 review: 1
- audit readiness: 1
- product integration: 1
- sla: 1
- cmmc: 1
- contract renewal: 1
- architecture gap: 1
- deployment: 1
- identity roadmap: 1
- mfa policy engine: 1
- sso connectors: 1
- qa process: 1
- infrastructure bottleneck: 1
- expansion: 1
- healthcare security: 1
- pipeline architecture: 1
- single point of failure: 1
- ci/cd improvement: 1
- capacity planning: 1
- hipaa reporting: 1
- compliance tooling: 1
- feature gaps: 1
- roadmap: 1
- upgrade rollout: 1
- hybrid environment: 1
- data protection: 1
- retail seasonality: 1
- product release: 1
- upgrade migration: 1
- product migration: 1
- outage impact: 1
- alert misconfiguration: 1
- incident follow-up: 1
- mfa enforcement: 1
- support escalation: 1
- identity deployment: 1
- privileged access management: 1
- mfa migration: 1
- compliance audit: 1
- entra id integration: 1
- bug fix: 1
- architecture: 1
- ingestion pipeline: 1
- alert deduplication: 1
- qa: 1
- outage recovery: 1
- incident root cause: 1
- data integrity: 1
- post-incident report: 1
- architectural remediation: 1
- saml certificate rotation: 1
- authentication failure: 1
- customer reliability concerns: 1
- identity federation: 1
- okta integration: 1
- deprovisioning: 1
- role management: 1
- session timeout policy: 1
- infrastructure deployment: 1
- qa and testing: 1
- observability and alerting: 1
- rollback planning: 1
- ldap sync failure: 1
- ssl certificate: 1
- active directory: 1
- trust recovery: 1
- sla negotiation: 1
- post-incident review: 1
- compliance risk: 1
- internal communications: 1
- product early access: 1
- invoice overage: 1
- backup policy: 1
- account management: 1
- usage monitoring: 1
- deployment planning: 1
- reliability incident: 1
- cloud infrastructure: 1
- integrations: 1
- detect pipeline failure: 1
- pci compliance: 1
- fraud monitoring: 1
- competitive analysis: 1
- product gaps: 1
- compliance features: 1
- pricing pressure: 1
- hipaa audit prep: 1
- evidence management: 1
- reporting: 1
- infrastructure architecture: 1
- detection tuning: 1
- feature requests: 1
- custom compliance templates: 1
- api integration: 1
- event ingestion pipeline: 1
- ot/scada monitoring: 1
- sla credits: 1
- mfa failure: 1
- totp: 1
- okta sso integration: 1
- p1 escalation: 1
- user lockout: 1
- reliability concern: 1
- custom reporting: 1
- compliance frameworks: 1
- hybrid infrastructure: 1
- aws integration: 1
- adoption metrics: 1
- cross-sell: 1
- gdpr roadmap: 1
- customer feedback: 1
- competitive positioning: 1
- sales strategy: 1
- incident remediation: 1
- hybrid backup strategy: 1
- disaster recovery testing: 1
- data retention policies: 1
- healthcare data management: 1
- quarterly business review: 1
- disaster recovery: 1
- throughput degradation: 1
- platform incident: 1
- sql server: 1
- agent update: 1
- quarterly planning: 1
- incremental backup performance: 1
- restore ux improvement: 1
- multi-region replication: 1
- canary deployment: 1
- tech debt: 1
- sso outage: 1
- identity module regression: 1
- saml configuration: 1
- patch deployment: 1
- qa coverage: 1
- customer churn: 1
- upsell opportunity: 1
- compliance roadmap: 1
- data backup: 1
- threat detection: 1
- support issue: 1
- evidence collection: 1
- design review: 1
- pdf export: 1
- performance testing: 1
- billing discrepancy: 1
- proration error: 1
- license upgrade: 1
- credit request: 1
- product features: 1
- multi-framework reporting: 1
- detect outage follow-up: 1
- documentation gap: 1
- regulatory compliance: 1
- infrastructure rebalancing: 1
- product gap: 1
- configuration review: 1
- provisioning sync delay: 1
- identity policy propagation: 1
- shared cluster performance: 1
- compliance requirements: 1
- sso configuration: 1
- mfa policy: 1
- contractor access controls: 1
- okta migration: 1
- shared workstation session management: 1
- mfa policy exceptions: 1
- sso integration: 1
- rbac enhancements: 1
- qa sign-off: 1
- safari bug: 1
- hipaa validation: 1
- incident communications: 1
- performance degradation: 1
- architecture scalability: 1
- incident risk: 1
- sales alignment: 1
- backup infrastructure: 1
- upsell: 1
- technical incident: 1
- api outage: 1
- healthcare data: 1
- health checks: 1
- chaos engineering: 1
- service credit: 1
- patch management: 1
- behavioral detection: 1
- customer retention risk: 1
- baseline configuration: 1
- circuit breaker implementation: 1
- monitoring gaps: 1
- knowledge sharing: 1
- soc 2 type ii audit: 1
- audit evidence packaging: 1
- product dogfooding: 1
- customer renewals: 1
- incident post-mortem: 1
- product expansion: 1
- early access: 1

Observed key moment types with counts:
- concern: 84
- positive_pivot: 76
- churn_signal: 61
- technical_issue: 54
- feature_gap: 51
- action_item: 43
- praise: 23
- pricing_offer: 10
