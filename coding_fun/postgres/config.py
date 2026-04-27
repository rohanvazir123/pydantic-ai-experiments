"""
Local PostgreSQL connection settings.

Priority order for each variable:
  1. Environment variable  (PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE)
  2. .env file in this directory  (key=value, one per line)
  3. Built-in default

Create a .env file from .env.example to avoid exporting env vars by hand:
    copy .env.example .env   # Windows
    cp .env.example .env     # Unix
Then fill in PG_PASSWORD (and any other values that differ from the defaults).
"""

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Parse a .env file and populate os.environ without overriding existing vars."""
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

HOST = os.getenv("PG_HOST", "localhost")
PORT = int(os.getenv("PG_PORT", "5432"))
USER = os.getenv("PG_USER", "postgres")
PASSWORD = os.getenv("PG_PASSWORD", "")
DATABASE = os.getenv("PG_DATABASE", "postgres")


def dsn() -> str:
    """Return a libpq-style DSN string for asyncpg."""
    if PASSWORD:
        return f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
    return f"postgresql://{USER}@{HOST}:{PORT}/{DATABASE}"
