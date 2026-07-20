

###  Steps involved in the overall flow along with the tech stack

#### Query Preprocessing

- [User query] => If selected from FAQ drop down, return cached SQL query

Run the SQL query if Run query is selected (true by default)
Observability logs in another UI tab

- [User query] => [Query Router] => reject query 
    ```
    [Query Router] => [ Query normalization] => [ Query disambiguation] 
    For Query normalization use NLP libraries like Spacy
    Normalize the NL query: remove whitespace, emojis, retain case
    Resolve nl dates into YYYY-MM-DD format
    Prompt:
    Example instruction in the prompt:
    “Use table and column names exactly as provided in the schema context. Do not change their case.”
    When you use dates in SQL, always format them as ‘YYYY-MM-DD’
    ```

Query disambiguation is hybrid = rules based + Secondary LLM (qwen2.5:3B)
If clarifying info is needed, prompt user for clarifications by sending it to the UI


- At this point,  query is clean


#### Schema discovery 

Goal:  As opposed dumping the entire schema graph into the context, generate schema tailored for the query

Runs in the background as a cron job or if schema changes/driven by DB CDC events

Output JSON schema in CHUNKS - example CHUNK is given below:

```
Output JSON schema chunks like this

{ 
    “database_name”: “Ariel_Inc_Products", 
    schema_name: “Products_schema”,
    “Tables”: [
        
        { 
            table_name": "Products", 
            "columns": [ 
                { "column_name": "PRODUCT_ID", "data_type": "KEY", "description": "Unique identifier for each Product", "sample_values": [1, 2, 3] }, 
                { "column_name": "PRODUCT_CATEGORY", "data_type": "INT", "description": "PRODUCT_CATEGORY KEY", "sample_values": [10, 20], "tags":[<>], "db_column_name":"<actual name>" },
                {"links": ["orders"]}
            ]
        },
        { 
            table_name": "Order", 
            "columns": [ 
                { "column_name": "ORDER_ID", "data_type": "KEY", "description": "Unique identifier for each order", "sample_values": [15, 25, 35
                ] }, 
                { "column_name": "PRODUCT", "data_type": "INT", "description": "PRODUCT KEY for the order", "sample_values": [1, 2], tags":[<>],
                "db_column_name":"<actual name>" },
                {"links": ["products", "revenue"]}
            ]
        }
    ]
} 

Column names like amt, flg, cd are meaningless to an LLM. so tbhe db_column_name is different  from the [search] column name
Each row is a CHUNK
Use Postgres and the tsvector and pgvector extensions
Generate pgvector out of this JSON schema chunk and store it in embedding column
Generate tsvector out of it and store it in content_tsv column
Need to store metadata like <db_name>: <schema_name>:< table_name>:<column_name> for ANN search using query embedding as key to get top-50 JSON schema chunks in the format <db_name>: <schema_name>:< table_name>:<column_name> 
Follow up with a light reranker (secondary LLM) to get top-10 chunks
```

#### Context Assembly

- Use Pydantic AI agent loop to orchestrate NL to SQL translation
Use qwen2.5-coder:7B model as query translator.

- LLM context = System Prompt + User prompt

- Do not expose tools to LLM unless you need to, plan on LLM resolving the NL query quickly

- Generate context in parts and assemble it

    ```
    User prompt = <query> ....</query> + <schema> ... </schema> + <guardrails/>
    Guard rails could be hard requirements like filter by region, limits (like size, nested queries, page size, Never generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE. )
    Static guard rails check:

    Reject any query containing:
    DDL/DML keywords: INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, etc.
    Multiple statements separated by ; when only a single statement is expected.
    Suspicious constructs: --, /* ... */ comments, xp_ stored procs, etc.
    Enforce query complexity limits:
    Max query length (tokens/characters).
    Max depth of nested subqueries (if you parse or regex-check).

    Use SQLglot for schema validation

    ```


### Caching
Use Redis

Selected NL Query (exact match) -> SQL

NL query => schema

NL Query => SQL 

SQL Query => SQL response


### Evaluation 

- Runs in 3 modes:
```
Mode 1: Pre-deployment gate (runs in CI on every PR)        
Mode 2: Scheduled regression run (daily, against production)
Mode 3: Shadow eval (every production query, async)
```          
