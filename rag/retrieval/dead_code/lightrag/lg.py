import os
import asyncio
from lightrag import LightRAG
from lightrag.llm.openai import openai_embed, gpt_4o_mini_complete

# Import Docling structural layout pipeline components
from docling.document_converter import DocumentConverter

# 1. Setup your unified PostgreSQL multi-model environment coordinates
os.environ["POSTGRES_HOST"] = "localhost"
os.environ["POSTGRES_PORT"] = "5432"
os.environ["POSTGRES_USER"] = "postgres"
os.environ["POSTGRES_PASSWORD"] = "your_secure_password"
os.environ["POSTGRES_DATABASE"] = "your_rag_db"

# 2. Instantiate LightRAG targeting your Postgres extensions
rag = LightRAG(
    working_dir="./rag_storage",
    embedding_func=openai_embed,
    llm_model_func=gpt_4o_mini_complete,
    kv_storage="PGKVStorage",
    vector_storage="PGVectorStorage",
    graph_storage="PGGraphStorage",
    doc_status_storage="PGDocStatusStorage",
)

# 3. Instantiate the standard Docling converter engine
docling_converter = DocumentConverter()


async def pipeline_process_and_ingest(file_path: str):
    """
    Parses complex multi-format documents using Docling and
    streams the structural output directly to LightRAG storage.
    """
    # Step A: Convert the source file into a Docling structured document object
    # Handles layout elements, image placement anchors, and table coordinates
    conversion_result = docling_converter.convert(file_path)

    # Step B: Export the complete layout as clean, structured Markdown text
    # This preserves structural elements like '| col 1 | col 2 |' tables for the graph LLM
    markdown_text = conversion_result.document.export_to_markdown()

    # Step C: Stream directly to LightRAG's ingestion engine
    # LightRAG handles the chunk splitting and entity graph extraction behind the scenes
    await rag.ainsert([markdown_text])


async def main():
    # Verify underlying Apache AGE containers and pgvector schemas are ready
    await rag.initialize_storages()

    # Execute an ingestion test run
    await pipeline_process_and_ingest("financial_report_with_complex_tables.pdf")


if __name__ == "__main__":
    asyncio.run(main())
