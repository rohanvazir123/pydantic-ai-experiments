"""Exponential backoff with partial jitter.

Note on naming: the design doc uses the term "full jitter" but this
implementation applies 15% jitter (not 100%), making it *partial* jitter.
This bounds worst-case delay while still preventing synchronised retry storms
across N workers.

Default schedule (base=5s, multiplier=2, max=125s):
    attempt=1 →  5s  ± 0.75s
    attempt=2 → 10s  ± 1.50s
    attempt=3 → 20s  ± 3.00s  (capped far below max)
"""

import random


def exponential_backoff(
    attempt: int,
    base_s: float = 5.0,
    multiplier: float = 2.0,
    max_s: float = 125.0,
    jitter_factor: float = 0.15,
) -> float:
    """Return seconds to sleep before attempt number `attempt` (1-indexed).

    Args:
        attempt:       retry attempt number; 1 = first retry after first failure.
        base_s:        base delay in seconds.
        multiplier:    exponential growth factor.
        max_s:         hard cap on the raw (pre-jitter) delay.
        jitter_factor: fraction of raw delay added as uniform random jitter.

    Returns:
        Seconds to sleep — always ≥ base_s, ≤ max_s + jitter.
    """
    attempt = max(attempt, 1)
    raw = min(base_s * (multiplier ** (attempt - 1)), max_s)
    jitter = random.uniform(0, jitter_factor * raw)
    return raw + jitter
