"""Pytest configuration for the agent-complexity examples.

Adds this directory to ``sys.path`` so the example modules (which import
``config`` / ``utils`` as top-level modules) are importable from the test
suite, and defines the ``ollama`` marker used to gate live-inference tests.

Live tests (marked ``@pytest.mark.ollama``) hit a real Ollama daemon and are
therefore slow and non-deterministic. They are **skipped by default** so the
core suite is fast and reliably green. Opt in with:

    RUN_OLLAMA=1 pytest            # or:
    pytest --run-ollama
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import pytest

# Make `config`, `utils`, and the `lN_*` example modules importable.
sys.path.insert(0, str(Path(__file__).parent))

# Collected latency samples: {level_label: [seconds, ...]}. Populated by the
# `latency` fixture (see below) and summarized at the end of the session.
_LATENCIES: dict[str, list[float]] = defaultdict(list)


@pytest.fixture
def latency(request: pytest.FixtureRequest):
    """Time a test and record its wall-clock latency under a level label.

    A test declares its label via ``@pytest.mark.level("L3 tool-calling")``;
    the fixture times the test body and stashes the elapsed seconds so the
    session-end summary can report per-level latency (and percentiles when a
    level has enough samples — e.g. under repetition or ``pytest-repeat``).
    """
    marker = request.node.get_closest_marker("level")
    label = marker.args[0] if marker and marker.args else request.node.name
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    _LATENCIES[label].append(elapsed)
    print(f"\n[latency] {label}: {elapsed:.2f}s")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-ollama",
        action="store_true",
        default=False,
        help="Run live tests against a local Ollama daemon.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "ollama: live test that requires a running Ollama daemon"
    )
    config.addinivalue_line(
        "markers", "level(label): label a live test with its complexity level for latency reporting"
    )


def _pct(values: list[float], p: float) -> float:
    """Nearest-rank percentile (fine for the small samples we collect)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(p / 100 * len(ordered) + 0.5) - 1))
    return ordered[k]


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Print a per-level latency table after a (live) run and save it to disk."""
    if not _LATENCIES:
        return
    lines = [
        "",
        "Per-level latency (wall-clock, local Ollama)",
        f"{'level':<26} {'n':>3} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}",
        "-" * 64,
    ]
    for label in sorted(_LATENCIES):
        s = _LATENCIES[label]
        lines.append(
            f"{label:<26} {len(s):>3} "
            f"{statistics.median(s):>7.2f}s {_pct(s, 95):>7.2f}s "
            f"{_pct(s, 99):>7.2f}s {max(s):>7.2f}s"
        )
    report = "\n".join(lines)
    terminalreporter.write_line(report)
    out = Path(__file__).parent / ".sample_runs" / "latency_tests.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report.strip() + "\n", encoding="utf-8")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-ollama") or os.getenv("RUN_OLLAMA") == "1":
        return
    skip = pytest.mark.skip(reason="live Ollama test; use --run-ollama or RUN_OLLAMA=1")
    for item in items:
        if "ollama" in item.keywords:
            item.add_marker(skip)
