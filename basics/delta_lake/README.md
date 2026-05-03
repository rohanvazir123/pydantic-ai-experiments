# Delta Lake Examples

Two end-to-end examples of Delta Lake merge pipelines.

| File | Storage | Source data |
|---|---|---|
| `example_local.py` | temp directory (no credentials needed) | in-memory PyArrow tables |
| `example_gcs.py` | local `output/orders_delta/` directory | GCS Parquet files via DuckDB httpfs |

Both `run_merge()` implementations follow the same pipeline as `run_pydantic_merge()` in `dt_merge.py`: schema validation → DuckDB streaming → lazy batch cast → atomic Delta merge with retries.

---

## 1. Install dependencies

These packages are not in `pyproject.toml` and must be installed separately:

```bash
pip install deltalake duckdb pyarrow tenacity python-dotenv
```

Verify:

```bash
python -c "import deltalake, duckdb, pyarrow, tenacity; print('OK')"
```

---

## 2. Run the local example (no credentials)

```bash
cd delta_lake
python example_local.py
```

Expected output:

```
[seed] 5 rows written -> version 0

-- after seed  (v0,  5 rows) --
  id= 1  status=ACTIVE      flag=True
  id= 2  status=INACTIVE    flag=True
  ...
[merge] committed -> version 1

-- after merge  (v1,  4 rows) --
  id= 2  status=ACTIVE      flag=True   <- updated (INACTIVE -> ACTIVE)
  id= 3  status=ACTIVE      flag=True   <- unchanged
  id= 5  status=ACTIVE      flag=True   <- inserted
  id= 1  status=ACTIVE      flag=True   <- kept (absent from source, not INACTIVE)
  id= 4  status=PENDING     flag=True   <- kept
```

---

## 3. Run the GCS example

### 3a. Configure credentials

The script reads a `.env` file from the `delta_lake/` directory. Create or update `delta_lake/.env` with your GCS HMAC key pair:

```
GCS_HMAC_ID=<your HMAC access key ID>
GCS_HMAC_SECRET=<your HMAC secret>
```

HMAC keys can be created in the GCP Console under **Cloud Storage → Settings → Interoperability** or with:

```bash
gcloud storage hmac create <service-account-email>
```

The service account needs at least **Storage Object Viewer** on the `eagerbeaver-1` bucket.

### 3b. Create the output directory

```bash
mkdir -p delta_lake/output
```

### 3c. Run

```bash
cd delta_lake
python example_gcs.py
```

Expected output:

```
[seed] 300 rows (Jul 25 + Jul 26) -> version 0

-- after seed  (v0,  300 rows) --
  2025-07-25  order_id=  1  product=Camera        qty=2
  2025-07-25  order_id=  2  product=Book          qty=3
  2025-07-25  order_id=  3  product=Camera        qty=2
  2025-07-25  order_id=  4  product=Laptop        qty=4
  2025-07-25  order_id=  5  product=Headphones    qty=4
  ... 290 more rows

[batch] 305 rows  (300 new date partitions + 5 corrections)
[merge] committed -> version 1

-- after merge  (v1,  600 rows) --
  2025-07-25  order_id=  1  product=Camera        qty=4   <- doubled (correction)
  2025-07-25  order_id=  2  product=Book          qty=6   <- doubled (correction)
  2025-07-25  order_id=  3  product=Camera        qty=4   <- doubled (correction)
  2025-07-25  order_id=  4  product=Laptop        qty=8   <- doubled (correction)
  2025-07-25  order_id=  5  product=Headphones    qty=8   <- doubled (correction)
  2025-07-25  order_id=  6  product=Headphones    qty=3   <- unchanged
  ... 594 more rows (300 Jul 28/29 rows inserted)

-- time travel -> v0  (v0,  300 rows) --
  2025-07-25  order_id=  1  product=Camera        qty=2   <- original qty preserved
  2025-07-25  order_id=  2  product=Book          qty=3
  ... 290 more rows

-- commit history --
  v1  MERGE  {'mergePredicate': 't.order_id = s.order_id AND t.order_date = s.order_date',
              'matchedPredicates': '[{"actionType":"update","predicate":"t.quantity != s.quantity"}]',
              'notMatchedPredicates': '[{"actionType":"insert"}]',
              'notMatchedBySourcePredicates': '[]'}
  v0  WRITE  {'mode': 'Overwrite'}
```

### 3d. Re-running and version numbers

`mode="overwrite"` in the seed step appends a new `WRITE` commit rather than resetting the log — so version numbers accumulate across runs (v0/v1 on first run, v2/v3 on second, etc.). Time travel with `version=0` always returns the very first seed, not the most recent one.

To start completely fresh and reset version numbers to 0/1:

```bash
# macOS / Linux
rm -rf delta_lake/output/orders_delta

# Windows (PowerShell)
Remove-Item -Recurse -Force delta_lake\output\orders_delta
```

Then re-run `python example_gcs.py`.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: deltalake` | `pip install deltalake` |
| `ModuleNotFoundError: duckdb` | `pip install duckdb` |
| `ModuleNotFoundError: tenacity` | `pip install tenacity` |
| `KeyError: 'GCS_HMAC_ID'` | Add `GCS_HMAC_ID` and `GCS_HMAC_SECRET` to `delta_lake/.env` |
| `duckdb.HTTPException: ... 403` | HMAC key is wrong or the service account lacks bucket read access |
| `ValueError: Target Delta table missing columns` | Schema mismatch — delete `output/orders_delta/` and re-run |
| `CommitFailedError` after 5 retries | Concurrent writer conflict; wait and re-run |
