@echo off
echo Creating RAG project directory structure...

REM Root rag directory
mkdir rag

REM Ingestion
mkdir rag\ingestion
mkdir rag\ingestion\chunkers

REM Storage
mkdir rag\storage
mkdir rag\storage\vector_store

REM Retrieval
mkdir rag\retrieval

REM Agent
mkdir rag\agent

REM Optional: config / utils
mkdir rag\config

REM Create __init__.py files (important for Python imports)
type nul > rag\__init__.py

type nul > rag\ingestion\__init__.py
type nul > rag\ingestion\chunkers\__init__.py

type nul > rag\storage\__init__.py
type nul > rag\storage\vector_store\__init__.py

type nul > rag\retrieval\__init__.py

type nul > rag\agent\__init__.py

type nul > rag\config\__init__.py

echo Done!
echo RAG directory structure created successfully.
pause
