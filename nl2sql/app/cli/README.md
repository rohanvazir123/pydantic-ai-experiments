# NL-to-SQL CLI

No standalone CLI scripts here yet. Use the Streamlit app or REST API for interactive querying.

## Quick programmatic usage

```python
import asyncio
import duckdb
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from nl2sql.nlp_sql_postgres_v2 import ConversationManager
from rag.config.settings import load_settings

async def main():
    settings = load_settings()
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL postgres; LOAD postgres;")
    conn.execute(f"ATTACH '{settings.database_url}' AS rag_db (TYPE postgres, READ_ONLY)")

    llm = OpenAIModel(settings.llm_model, base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    agent = Agent(model=llm, result_type=str)
    manager = ConversationManager(conn=conn, agent=agent, schema_text="...")

    result = await manager.run_query("How many documents are stored?")
    print(result.sql)
    print(result.rows)

asyncio.run(main())
```
