"""Set env vars before any pipeline import so agent instantiation succeeds in unit tests."""
import os

os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")
