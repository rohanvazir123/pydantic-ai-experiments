"""Make the sibling exercise modules importable from this central tests dir.

Each exercise keeps its source next to its README (``rate_limiter/``,
``moving_average/``, ``worker_queues/``); the tests all live here. Adding those
dirs to ``sys.path`` lets the test files do bare imports like
``import token_bucket_fixed`` without packaging.
"""

import sys
from pathlib import Path

_TELEMETRY = Path(__file__).resolve().parent.parent  # basics/telemetry
for _src in ("rate_limiter", "moving_average", "worker_queues", "state_machine"):
    sys.path.insert(0, str(_TELEMETRY / _src))
