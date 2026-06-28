# Query Normalization and Schema Enrichment

Two preprocessing steps that most NL2SQL teams skip and then wonder why retrieval quality is poor and cache hit rates are low. Query normalization happens at query time; schema enrichment happens at indexing time. Both are essential.

## Table of Contents

- [Why These Are Skipped and Why That Hurts](#why-these-are-skipped-and-why-that-hurts)
- [Query Normalization](#query-normalization)
  - [What to Normalize](#what-to-normalize)
  - [Normalization Pipeline](#normalization-pipeline)
  - [Concrete Examples](#concrete-examples)
  - [Impact on Cache Hit Rate](#impact-on-cache-hit-rate)
- [Schema Enrichment](#schema-enrichment)
  - [What Production Schemas Look Like Without Enrichment](#what-production-schemas-look-like-without-enrichment)
  - [Enrichment Layers](#enrichment-layers)
  - [Automated Enrichment Pipeline](#automated-enrichment-pipeline)
  - [Business Glossary Integration](#business-glossary-integration)
  - [Keeping Enrichment Fresh](#keeping-enrichment-fresh)
- [How They Work Together](#how-they-work-together)

---

## Why These Are Skipped and Why That Hurts

**Query normalization is skipped** because it seems unnecessary — "the LLM understands natural language, right?" It does, but the embedding model and the cache do not. Two users asking the same question in slightly different ways get:
- Different embeddings → different retrieval results → inconsistent answers
- Different cache keys → no cache hit → pay for LLM generation twice

**Schema enrichment is skipped** because it requires effort upfront. Production schemas are often built by engineers for engineers — column names like `cust_ord_trans_amt`, `flg_b2b_ind`, `dt_creat_ts` are completely opaque to an embedding model. Without enrichment, the embedding index is built on meaningless abbreviations and the schema retrieval step fails silently.

The result of skipping both: a system that works on the demo schema and fails on the real one.

---

## Query Normalization

### What to Normalize

Normalization is not about changing what the user asked — it is about removing surface variation that means the same thing so that routing, retrieval, and caching can treat equivalent queries identically.

**1. Case and whitespace:**
```
"Show me Revenue"  →  "show me revenue"
"show  me  revenue"  →  "show me revenue"  (collapse multiple spaces)
"Show me revenue\n"  →  "show me revenue"  (strip trailing newlines/tabs)
```

**2. Punctuation:**
```
"Show me revenue, by region."  →  "show me revenue by region"
"What's the ARR?"  →  "what is the arr"
"top-10 customers"  →  "top 10 customers"
```

**3. Abbreviation and acronym expansion:**
```
"q3 rev"     →  "q3 revenue"
"ytd sales"  →  "year to date sales"
"mtd arr"    →  "month to date annual recurring revenue"
"# of cust"  →  "number of customers"
"avg order val"  →  "average order value"
```
Maintain a tenant-specific abbreviation dictionary. Common ones are universal (YTD, MTD, ARR, MRR, LTV, CAC); others are tenant-specific ("NB" = new business, "PS" = professional services).

**4. Synonym normalisation:**
```
"clients"   →  "customers"   (if schema uses "customers")
"revenue"   →  "revenue"     (no change — already canonical)
"bookings"  →  "revenue"     (if company defines bookings as revenue)
"headcount" →  "employees"   (if schema uses "employees")
"reps"      →  "sales_representatives"
```
This is the most impactful normalisation. Synonyms that resolve to the same schema concept should be mapped to a single canonical term before embedding. This dramatically improves retrieval recall and cache hit rate.

**5. Number normalisation:**
```
"top 10 customers"   →  "top 10 customers"  (keep)
"top ten customers"  →  "top 10 customers"  (convert word to digit)
"last 7 days"        →  "last 7 days"       (keep)
"last seven days"    →  "last 7 days"       (convert)
```

**6. Typo correction:**
```
"reveunue"    →  "revenue"
"custoemrs"   →  "customers"
"quaterly"    →  "quarterly"
```
Use a lightweight spell checker (pyspellchecker, SymSpell) applied only to known business terms, not arbitrary text. Do not over-correct — "arr" is not a typo for "are".

**7. Pronoun and ellipsis resolution (for multi-turn):**
```
Turn 1: "Show me Q3 revenue by region"
Turn 2: "Now filter to EMEA only"  →  "Show me Q3 revenue by region filtered to EMEA"
Turn 3: "And break it down by month"  →  "Show me Q3 revenue by region filtered to EMEA broken down by month"
```
Resolve pronouns and elliptical references against the conversation context before normalising.

---

### Normalization Pipeline

```python
class QueryNormalizer:

    def __init__(self, tenant_config: TenantConfig):
        self.abbreviations = tenant_config.abbreviation_dict    # {"ytd": "year to date", ...}
        self.synonyms      = tenant_config.synonym_dict         # {"clients": "customers", ...}
        self.spell_checker = DomainSpellChecker(tenant_config.domain_vocab)

    def normalize(self, query: str, conversation_context: list[str] | None = None) -> str:
        # Step 1: resolve pronouns if multi-turn
        if conversation_context:
            query = self._resolve_pronouns(query, conversation_context)

        # Step 2: lowercase + strip
        query = query.lower().strip()

        # Step 3: punctuation — keep apostrophes for contractions, remove rest
        query = re.sub(r"'s\b", "", query)          # "company's" → "company"
        query = re.sub(r"'(re|ve|ll|d)\b", " \\1", query)  # "what're" → "what re"
        query = re.sub(r"[^\w\s]", " ", query)      # remove all other punctuation
        query = re.sub(r"\s+", " ", query).strip()  # collapse whitespace

        # Step 4: typo correction (domain vocabulary only)
        query = self.spell_checker.correct(query)

        # Step 5: number words to digits
        query = word_to_number(query)               # "ten" → "10"

        # Step 6: abbreviation expansion
        tokens = query.split()
        tokens = [self.abbreviations.get(t, t) for t in tokens]
        query = " ".join(tokens)

        # Step 7: synonym normalisation (after expansion, so "ytd arr" → "year to date annual recurring revenue" first)
        tokens = query.split()
        tokens = [self.synonyms.get(t, t) for t in tokens]
        query = " ".join(tokens)

        return query

# Example:
normalizer = QueryNormalizer(tenant_config)
normalizer.normalize("What's the YTD ARR by region?")
# → "what is the year to date annual recurring revenue by region"

normalizer.normalize("top-10 custoemrs last Q")
# → "top 10 customers last quarter"
```

---

### Concrete Examples

| Raw query | Normalized query | What changed |
|-----------|-----------------|--------------|
| "Show me Q3 rev" | "show me q3 revenue" | abbreviation expansion, lowercase |
| "What's the YTD sales by region?" | "what is the year to date sales by region" | contraction, abbreviation |
| "top ten clients" | "top 10 customers" | word-to-number, synonym |
| "reveunue last quarter" | "revenue last quarter" | typo correction |
| "Show me the # of deals closed" | "show me the number of deals closed" | punctuation, abbreviation |
| "ARR MTD" | "annual recurring revenue month to date" | double abbreviation |
| "How r we doing?" | "how are we doing" | contraction expansion |

---

### Impact on Cache Hit Rate

Cache hit rate before and after normalization (real-world pattern):

```
Without normalization:
  "show me revenue last quarter"          → cache miss
  "Show Me Revenue Last Quarter"          → cache miss (different case)
  "show me rev last q"                    → cache miss (abbreviation)
  "show me revenue last qtr"              → cache miss (abbreviation variant)
  
  4 semantically identical queries, 4 cache misses, 4 LLM calls
  Effective cache hit rate for this query cluster: 0%

With normalization (all normalize to "show me revenue last quarter"):
  Query 1: cache miss → LLM call → cached
  Query 2: cache hit
  Query 3: cache hit
  Query 4: cache hit
  
  Effective cache hit rate for this query cluster: 75%
```

At scale, normalization alone typically improves cache hit rate by 15–25 percentage points. At 10,000 queries/day and $0.014/LLM call:
- Without: 2,000 cache hits (20%) → 8,000 LLM calls → $112/day
- With: 4,500 cache hits (45%) → 5,500 LLM calls → $77/day
- Savings: $35/day = $1,050/month from normalization alone

---

## Schema Enrichment

### What Production Schemas Look Like Without Enrichment

A schema pulled directly from `information_schema` for a 15-year-old enterprise system:

```sql
TABLE: cust_mstr
  id              BIGINT
  cust_id_ext     VARCHAR(50)
  cust_nm_full    VARCHAR(200)
  cust_nm_shrt    VARCHAR(50)
  cust_typ_cd     CHAR(2)
  cust_seg_cd     CHAR(3)
  dt_creat_ts     TIMESTAMP
  dt_upd_ts       TIMESTAMP
  flg_actv_ind    SMALLINT
  flg_b2b_ind     SMALLINT
  own_reg_cd      CHAR(5)
  src_sys_cd      CHAR(10)

TABLE: ord_trans_fct
  ord_id          BIGINT
  cust_id         BIGINT
  prod_sk         BIGINT
  ord_dt          DATE
  ord_amt_lc      DECIMAL(18,4)
  ord_amt_usd     DECIMAL(18,4)
  ord_qty         INTEGER
  disc_pct        DECIMAL(5,4)
  chnl_cd         CHAR(3)
  sts_cd          CHAR(2)
```

Every column name is an abbreviation. The embedding model has no signal to distinguish `cust_nm_full` (customer full name) from `cust_nm_shrt` (customer short name). `flg_b2b_ind` is completely opaque. `own_reg_cd` is meaningful only if you know the owning region code system.

An NL2SQL system built on this schema without enrichment will fail on almost every query.

### Enrichment Layers

**Layer 1 — Table-level description:**
```json
{
  "table_name": "cust_mstr",
  "display_name": "Customers",
  "description": "Master customer record. One row per customer account. Contains customer identity, segmentation, and status. Links to order history via cust_id.",
  "business_domain": "CRM",
  "common_use_cases": ["customer lookup", "segmentation analysis", "active customer count"],
  "synonyms": ["accounts", "clients", "buyers", "counterparties"]
}
```

**Layer 2 — Column-level descriptions:**
```json
[
  {
    "column": "cust_typ_cd",
    "display_name": "Customer Type",
    "description": "Two-character code classifying the customer type. Values: 'EN'=Enterprise, 'MM'=Mid-Market, 'SM'=SMB, 'CO'=Consumer.",
    "sample_values": ["EN", "MM", "SM", "CO"],
    "synonyms": ["account type", "customer segment", "tier"],
    "is_filter_column": true,
    "cardinality": "low"
  },
  {
    "column": "flg_actv_ind",
    "display_name": "Active Customer Flag",
    "description": "Boolean flag. 1 = active customer (placed an order in the last 12 months), 0 = inactive.",
    "sample_values": [0, 1],
    "synonyms": ["active", "is active", "customer status"],
    "default_filter": "WHERE flg_actv_ind = 1",
    "note": "Most queries about 'customers' should implicitly filter flg_actv_ind = 1 unless the user specifies 'all customers' or 'inactive customers'."
  },
  {
    "column": "own_reg_cd",
    "display_name": "Owning Region Code",
    "description": "Five-character code for the sales region that owns this account. Values defined in region_ref table. Common values: 'NAMRC'=North America, 'EMEAI'=EMEA, 'APJAP'=Asia Pacific.",
    "sample_values": ["NAMRC", "EMEAI", "APJAP", "LATAM"],
    "synonyms": ["region", "territory", "sales region", "geography"],
    "is_filter_column": true,
    "join_hint": "JOIN region_ref ON cust_mstr.own_reg_cd = region_ref.reg_cd"
  }
]
```

**Layer 3 — Business glossary KPI definitions:**
```json
[
  {
    "term": "ARR",
    "full_name": "Annual Recurring Revenue",
    "definition": "Sum of ord_trans_fct.ord_amt_usd for orders where ord_trans_fct.sts_cd IN ('AC', 'RN') (active or renewal), annualised. Formula: SUM(ord_amt_usd * 12 / contract_months).",
    "sql_template": "SUM(CASE WHEN sts_cd IN ('AC', 'RN') THEN ord_amt_usd * 12.0 / contract_months ELSE 0 END)",
    "required_tables": ["ord_trans_fct"],
    "synonyms": ["annual recurring revenue", "subscription revenue", "recurring revenue"]
  },
  {
    "term": "churn",
    "full_name": "Customer Churn",
    "definition": "Customers who had sts_cd='AC' in the prior period and have sts_cd='CH' or no active orders in the current period.",
    "sql_template": "/* see churn calculation template churn_v3.sql */",
    "required_tables": ["cust_mstr", "ord_trans_fct"],
    "synonyms": ["attrition", "lost customers", "cancelled", "churned accounts"]
  }
]
```

**Layer 4 — Join path catalog:**
```json
[
  {
    "from_table": "cust_mstr",
    "to_table": "ord_trans_fct",
    "join_condition": "cust_mstr.id = ord_trans_fct.cust_id",
    "join_type": "LEFT",
    "description": "Links customers to their order history. Use LEFT JOIN to include customers with no orders.",
    "cardinality": "one-to-many"
  },
  {
    "from_table": "cust_mstr",
    "to_table": "region_ref",
    "join_condition": "cust_mstr.own_reg_cd = region_ref.reg_cd",
    "join_type": "INNER",
    "description": "Resolves region code to full region name and hierarchy.",
    "cardinality": "many-to-one"
  }
]
```

---

### Automated Enrichment Pipeline

Manual annotation of 400 tables × 50 columns is not feasible. Automate 80%, validate 20%.

```python
class SchemaEnricher:

    def enrich_column(self, table: str, column: ColumnMeta, sample_values: list) -> ColumnEnrichment:

        # Step 1: expand abbreviation in column name
        expanded_name = self._expand_column_name(column.name)
        # "cust_nm_full" → "customer name full"
        # "flg_b2b_ind"  → "flag business to business indicator"
        # "ord_amt_usd"  → "order amount usd"

        # Step 2: infer column purpose from name + type + sample values
        purpose = self._infer_purpose(expanded_name, column.dtype, sample_values)
        # BOOLEAN FLAG, AMOUNT_USD, DATE, CATEGORY_CODE, IDENTIFIER, etc.

        # Step 3: generate natural language description via LLM
        description = self.llm.complete(
            f"Table: {table} (context: {self.table_descriptions[table]})\n"
            f"Column: {column.name} (expanded: {expanded_name})\n"
            f"Type: {column.dtype}\n"
            f"Sample values: {sample_values[:10]}\n"
            f"Other columns in table: {self._sibling_column_names(table)}\n\n"
            f"Write a one-sentence description of what this column contains. "
            f"If it is a code or flag, list what the values mean."
        )

        # Step 4: extract synonyms from description
        synonyms = self._extract_synonyms(description, expanded_name)

        # Step 5: infer default filter rule
        default_filter = self._infer_default_filter(column, sample_values, description)
        # e.g., "flg_actv_ind = 1" for active flag columns

        return ColumnEnrichment(
            display_name=expanded_name.title(),
            description=description,
            synonyms=synonyms,
            sample_values=sample_values[:5],
            default_filter=default_filter,
            confidence=self._confidence_score(description, sample_values),
        )

    def _expand_column_name(self, name: str) -> str:
        """Expand abbreviations in snake_case column names."""
        ABBREV = {
            "cust": "customer", "ord": "order", "prod": "product",
            "amt": "amount",    "qty": "quantity", "dt": "date",
            "ts": "timestamp",  "cd": "code",  "nm": "name",
            "flg": "flag",      "ind": "indicator", "pct": "percent",
            "lc": "local currency", "usd": "us dollars",
            "creat": "created", "upd": "updated", "actv": "active",
            "mstr": "master",   "fct": "fact",  "ref": "reference",
            "sk": "surrogate key", "ext": "external", "shrt": "short",
            "seg": "segment",   "typ": "type",  "reg": "region",
            "src": "source",    "sys": "system", "own": "owning",
        }
        parts = name.lower().split("_")
        return " ".join(ABBREV.get(p, p) for p in parts)
```

**Cost of automated enrichment:**
- One LLM call per column (~200 input tokens + 100 output tokens)
- At GPT-4o-mini rates: ~$0.0001/column
- For 400 tables × 50 columns = 20,000 columns: $2.00 total
- This is not a cost problem. Run it.

**Human review queue:**
Enrich automatically, then route low-confidence descriptions to a human reviewer. Confidence is low when:
- The column name is a pure code with no recognisable parts (`flg_zz9_ind`)
- Sample values are all NULL or a single constant
- The generated description contains hedging language ("possibly", "may represent")

Target: human review for 10–15% of columns. For 20,000 columns: 2,000–3,000 reviews. Assign to data stewards over 2–3 weeks.

---

### Business Glossary Integration

The glossary bridges user language to schema. Every KPI, business term, and metric definition lives here:

```python
class BusinessGlossary:

    def resolve(self, term: str) -> list[ColumnMapping]:
        """
        Given a user term, return the SQL expression(s) it maps to.
        Returns multiple if the term is ambiguous across contexts.
        """
        term_normalized = normalize(term)

        # Exact match first
        if term_normalized in self.glossary:
            return [self.glossary[term_normalized]]

        # Fuzzy match on synonyms
        candidates = [
            entry for entry in self.glossary.values()
            if term_normalized in entry.synonyms
            or fuzzy_match(term_normalized, entry.synonyms) > 0.85
        ]

        return candidates

# Glossary entry structure:
@dataclass
class GlossaryEntry:
    term:          str           # canonical term: "arr"
    full_name:     str           # "Annual Recurring Revenue"
    synonyms:      list[str]     # ["annual recurring revenue", "subscription revenue"]
    sql_expression: str          # the SQL to compute this metric
    required_tables: list[str]   # tables that must be in the FROM clause
    filters:       list[str]     # implicit WHERE conditions
    version:       int           # increment when definition changes
    owner:         str           # who is responsible for this definition
    last_reviewed: date          # when was this last validated
```

**Why glossary versioning matters:**
When the definition of "ARR" changes (e.g., the finance team changes the formula), all cached SQL that computed ARR using the old formula becomes wrong. The glossary version is a component of the cache key — changing the version automatically invalidates all cached queries that used this term.

---

### Keeping Enrichment Fresh

Schema enrichment is not a one-time job. Three things change:

1. **New columns added:** Trigger enrichment automatically on any `ALTER TABLE ADD COLUMN` event.
2. **Column values change:** If `cust_typ_cd` adds a new value `'GV'` (Government), the description listing valid values is now incomplete. Re-enrich any column whose distinct values have changed by > 10% since last enrichment.
3. **Business definitions evolve:** Glossary terms must have owners who review them quarterly. A glossary entry with `last_reviewed` > 90 days triggers a review notification.

```python
def schema_change_handler(event: SchemaChangeEvent):
    if event.type == "COLUMN_ADDED":
        enrich_column(event.table, event.column)

    if event.type == "COLUMN_MODIFIED":
        re_enrich_column(event.table, event.column)
        invalidate_cache_for_column(event.table, event.column)

    if event.type == "TABLE_ADDED":
        enrich_table(event.table)
        update_embedding_index(event.table)
```

---

## How They Work Together

Query normalization and schema enrichment are complementary:

```
User query: "Show me YTD ARR by region"

Step 1 — Normalize:
  "show me year to date annual recurring revenue by region"
  (abbreviations expanded, lowercase)

Step 2 — Cache lookup:
  Cache key: hash("show me year to date annual recurring revenue by region" + schema_v14)
  → cache miss (first time)

Step 3 — Schema retrieval (enriched schema):
  Query embedding of normalized query matches:
  - "ord_trans_fct" (enriched: "order transaction fact — contains annual recurring revenue, subscription status")
  - "cust_mstr"     (enriched: "customer master — customer segments, owning region")
  - "region_ref"    (enriched: "region reference — region names, hierarchy")
  
  Without enrichment, "annual recurring revenue" would NOT match "ord_trans_fct" because 
  the raw schema only has "ord_amt_usd" and "sts_cd" — no mention of ARR anywhere.

Step 4 — Glossary resolution:
  "annual recurring revenue" → glossary entry for ARR
  SQL expression: SUM(CASE WHEN sts_cd IN ('AC', 'RN') THEN ord_amt_usd * 12.0 / contract_months END)
  Required tables: ord_trans_fct

Step 5 — SQL generation (with enriched schema in prompt):
  The LLM sees column descriptions, not raw names.
  It knows "own_reg_cd" is the region filter, "flg_actv_ind = 1" is the active customer filter.
  It generates correct SQL on the first attempt.

Step 6 — Cache write:
  Store result against the normalized query key.
  Next user who asks "YTD ARR by region?", "yr to date ARR by reg?", or 
  "annual recurring revenue this year by geography?" all normalize to the same key → cache hit.
```

Without normalization: different users, same intent, 0% cache hit rate.
Without enrichment: schema retrieval fails silently, LLM hallucinates column names.
With both: first query is answered correctly, subsequent similar queries are answered from cache.
