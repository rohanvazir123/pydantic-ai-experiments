# Copyright 2024 The Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
