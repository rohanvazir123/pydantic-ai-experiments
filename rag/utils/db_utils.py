"""
Shared database utility functions.

Module: rag.utils.db_utils
==========================

Provides helpers shared across the codebase for database connectivity.
"""

from urllib.parse import parse_qs, urlparse


def parse_database_url(database_url: str) -> dict:
    """
    Parse a PostgreSQL DATABASE_URL into pgvector connection parameters.

    Handles standard PostgreSQL URLs including the ``options`` query parameter.

    Args:
        database_url: PostgreSQL connection string, e.g.
            ``postgresql://user:pass@host/dbname?sslmode=require``

    Returns:
        Dict with keys: user, password, host, port, dbname, and optionally options.

    Raises:
        ValueError: If database_url is empty or None.
    """
    if not database_url:
        raise ValueError("DATABASE_URL not configured")

    parsed = urlparse(database_url)

    config: dict = {
        "user": parsed.username or "",
        "password": parsed.password or "",
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "dbname": parsed.path.lstrip("/") if parsed.path else "",
    }

    # Preserve options query parameter if present
    query_params = parse_qs(parsed.query)
    if "options" in query_params:
        config["options"] = query_params["options"][0]

    return config
