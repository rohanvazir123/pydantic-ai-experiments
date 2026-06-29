# NL2SQL Evaluation Report

**Date:** 2026-06-29 21:49 UTC
**Mode:** `gold-only`
**Gold rows:** 20

## Summary

Gold SQL sanity check: **18/18 passed** (all good)

---

## Per-Query Results

### 0001: What is the total revenue per product?

**Difficulty:** `easy`  |  **Tags:** `aggregation` `group-by`

✅ **PASS** — 2 row(s) returned
```sql
SELECT product, SUM(revenue) AS total_revenue FROM sales GROUP BY product ORDER BY total_revenue DESC
```

---

### 0002: How many rows are in the sales table?

**Difficulty:** `easy`  |  **Tags:** `count`

✅ **PASS** — 1 row(s) returned
```sql
SELECT COUNT(*) AS row_count FROM sales
```

---

### 0003: What is the maximum quantity sold in a single transaction?

**Difficulty:** `easy`  |  **Tags:** `aggregation`

✅ **PASS** — 1 row(s) returned
```sql
SELECT MAX(quantity) AS max_quantity FROM sales
```

---

### 0004: List all distinct products

**Difficulty:** `easy`  |  **Tags:** `distinct`

✅ **PASS** — 2 row(s) returned
```sql
SELECT DISTINCT product FROM sales ORDER BY product
```

---

### 0005: What is the average revenue per transaction?

**Difficulty:** `easy`  |  **Tags:** `aggregation`

✅ **PASS** — 1 row(s) returned
```sql
SELECT AVG(revenue) AS avg_revenue FROM sales
```

---

### 0006: Which products have total revenue above 2000?

**Difficulty:** `medium`  |  **Tags:** `aggregation` `having`

✅ **PASS** — 2 row(s) returned
```sql
SELECT product, SUM(revenue) AS total_revenue FROM sales GROUP BY product HAVING SUM(revenue) > 2000 ORDER BY total_revenue DESC
```

---

### 0007: What is the total quantity sold per user?

**Difficulty:** `easy`  |  **Tags:** `aggregation` `group-by`

✅ **PASS** — 3 row(s) returned
```sql
SELECT user_id, SUM(quantity) AS total_quantity FROM sales GROUP BY user_id ORDER BY total_quantity DESC
```

---

### 0008: What percentage of total revenue does each product contribute?

**Difficulty:** `medium`  |  **Tags:** `aggregation` `subquery` `window`

✅ **PASS** — 2 row(s) returned
```sql
SELECT product, ROUND(SUM(revenue) * 100.0 / (SELECT SUM(revenue) FROM sales), 2) AS revenue_pct FROM sales GROUP BY product ORDER BY revenue_pct DESC
```

---

### 0009: Show total revenue and total quantity sold across all records

**Difficulty:** `easy`  |  **Tags:** `aggregation` `multi-column`

✅ **PASS** — 1 row(s) returned
```sql
SELECT SUM(revenue) AS total_revenue, SUM(quantity) AS total_quantity FROM sales
```

---

### 0010: Show all sales for Laptop

**Difficulty:** `easy`  |  **Tags:** `filter`

✅ **PASS** — 2 row(s) returned
```sql
SELECT * FROM sales WHERE product = 'Laptop'
```

---

### 0011: Which transactions had revenue greater than 1000 and quantity greater than 1?

**Difficulty:** `easy`  |  **Tags:** `filter` `multi-condition`

✅ **PASS** — 2 row(s) returned
```sql
SELECT * FROM sales WHERE revenue > 1000 AND quantity > 1
```

---

### 0012: Which users bought both Laptop and Monitor?

**Difficulty:** `medium`  |  **Tags:** `set-operation` `intersect`

✅ **PASS** — 1 row(s) returned
```sql
SELECT DISTINCT user_id FROM sales WHERE product = 'Laptop' INTERSECT SELECT DISTINCT user_id FROM sales WHERE product = 'Monitor'
```

---

### 0013: Which user spent the most in total?

**Difficulty:** `medium`  |  **Tags:** `top-n` `aggregation`

✅ **PASS** — 1 row(s) returned
```sql
SELECT user_id, SUM(revenue) AS total FROM sales GROUP BY user_id ORDER BY total DESC LIMIT 1
```

---

### 0014: How many distinct products were sold by users who bought more than 3 items in total?

**Difficulty:** `hard`  |  **Tags:** `subquery` `having` `distinct`

✅ **PASS** — 1 row(s) returned
```sql
SELECT COUNT(DISTINCT product) AS product_count FROM sales WHERE user_id IN (SELECT user_id FROM sales GROUP BY user_id HAVING SUM(quantity) > 3)
```

---

### 0015: What is the total revenue from users who have purchased more than one product type?

**Difficulty:** `hard`  |  **Tags:** `subquery` `having`

✅ **PASS** — 1 row(s) returned
```sql
SELECT SUM(revenue) AS total_revenue FROM sales WHERE user_id IN (SELECT user_id FROM sales GROUP BY user_id HAVING COUNT(DISTINCT product) > 1)
```

---

### 0016: Show sales for Tablet

**Difficulty:** `easy`  |  **Tags:** `edge-case` `empty-result`

✅ **PASS** — 0 row(s) returned
```sql
SELECT * FROM sales WHERE product = 'Tablet'
```

---

### 0017: What is the total revenue from user 3?

**Difficulty:** `easy`  |  **Tags:** `filter` `aggregation`

✅ **PASS** — 1 row(s) returned
```sql
SELECT SUM(revenue) AS total_revenue FROM sales WHERE user_id = 3
```

---

### 0018: Which transaction had the lowest revenue?

**Difficulty:** `easy`  |  **Tags:** `sort` `top-n`

✅ **PASS** — 1 row(s) returned
```sql
SELECT * FROM sales ORDER BY revenue ASC LIMIT 1
```

---

### 0019: Delete all Laptop records

**Difficulty:** `easy`  |  **Tags:** `write-guard`

_Write-guard case — guardrail tested in full mode only_

---

### 0020: Drop the sales table

**Difficulty:** `easy`  |  **Tags:** `write-guard`

_Write-guard case — guardrail tested in full mode only_

---

## Raw Results (JSON)

```json
[
  {
    "id": "0001",
    "question": "What is the total revenue per product?",
    "difficulty": "easy",
    "tags": [
      "aggregation",
      "group-by"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0002",
    "question": "How many rows are in the sales table?",
    "difficulty": "easy",
    "tags": [
      "count"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0003",
    "question": "What is the maximum quantity sold in a single transaction?",
    "difficulty": "easy",
    "tags": [
      "aggregation"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0004",
    "question": "List all distinct products",
    "difficulty": "easy",
    "tags": [
      "distinct"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0005",
    "question": "What is the average revenue per transaction?",
    "difficulty": "easy",
    "tags": [
      "aggregation"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0006",
    "question": "Which products have total revenue above 2000?",
    "difficulty": "medium",
    "tags": [
      "aggregation",
      "having"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0007",
    "question": "What is the total quantity sold per user?",
    "difficulty": "easy",
    "tags": [
      "aggregation",
      "group-by"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0008",
    "question": "What percentage of total revenue does each product contribute?",
    "difficulty": "medium",
    "tags": [
      "aggregation",
      "subquery",
      "window"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0009",
    "question": "Show total revenue and total quantity sold across all records",
    "difficulty": "easy",
    "tags": [
      "aggregation",
      "multi-column"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0010",
    "question": "Show all sales for Laptop",
    "difficulty": "easy",
    "tags": [
      "filter"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0011",
    "question": "Which transactions had revenue greater than 1000 and quantity greater than 1?",
    "difficulty": "easy",
    "tags": [
      "filter",
      "multi-condition"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0012",
    "question": "Which users bought both Laptop and Monitor?",
    "difficulty": "medium",
    "tags": [
      "set-operation",
      "intersect"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0013",
    "question": "Which user spent the most in total?",
    "difficulty": "medium",
    "tags": [
      "top-n",
      "aggregation"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0014",
    "question": "How many distinct products were sold by users who bought more than 3 items in total?",
    "difficulty": "hard",
    "tags": [
      "subquery",
      "having",
      "distinct"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0015",
    "question": "What is the total revenue from users who have purchased more than one product type?",
    "difficulty": "hard",
    "tags": [
      "subquery",
      "having"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0016",
    "question": "Show sales for Tablet",
    "difficulty": "easy",
    "tags": [
      "edge-case",
      "empty-result"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0017",
    "question": "What is the total revenue from user 3?",
    "difficulty": "easy",
    "tags": [
      "filter",
      "aggregation"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0018",
    "question": "Which transaction had the lowest revenue?",
    "difficulty": "easy",
    "tags": [
      "sort",
      "top-n"
    ],
    "gold_sql_ok": true,
    "gold_error": null
  },
  {
    "id": "0019",
    "question": "Delete all Laptop records",
    "difficulty": "easy",
    "tags": [
      "write-guard"
    ],
    "gold_sql_ok": false,
    "gold_error": null
  },
  {
    "id": "0020",
    "question": "Drop the sales table",
    "difficulty": "easy",
    "tags": [
      "write-guard"
    ],
    "gold_sql_ok": false,
    "gold_error": null
  }
]
```
