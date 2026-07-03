"""
Simulated runtime shared by Levels 4 and 5.

Levels 4 (agent harness) and 5 (multi-agent) both give agents access to a small
"runtime": a sandboxed view of the ``knowledge/`` filesystem plus a fake billing
API. Keeping that runtime here means:

  * l4 and l5 don't duplicate it, and
  * the logic is *pure* (takes a ``root: Path``, no ``RunContext``), so the test
    suite can exercise reads, globs, greps, and the path-traversal guard
    deterministically without a model or a network.

The ``@agent.tool`` wrappers in the example files are thin: they just pass
``ctx.deps.root`` into these functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import ModelRetry

if TYPE_CHECKING:
    from pathlib import Path


def safe_path(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root``, rejecting escapes (path traversal).

    Raises:
        ModelRetry: if the resolved path would fall outside ``root``. Raising
            ``ModelRetry`` (rather than a hard error) lets the agent recover by
            trying a valid path on its next step.
    """
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ModelRetry(
            f"Access denied: '{relative}' is outside the knowledge base. "
            "Use paths relative to the knowledge base root."
        )
    return candidate


def list_files_text(root: Path, glob: str = "**/*.md") -> str:
    """List files under ``root`` matching ``glob`` (default: all .md files)."""
    root = root.resolve()
    matches = sorted(
        p.relative_to(root).as_posix() for p in root.glob(glob) if p.is_file()
    )
    return "\n".join(matches) if matches else f"No files match '{glob}'."


def read_file_text(root: Path, path: str) -> str:
    """Read a single file under ``root`` by relative path.

    Raises:
        ModelRetry: if the file is missing or escapes the sandbox.
    """
    target = safe_path(root, path)
    if not target.is_file():
        raise ModelRetry(f"File not found: '{path}'. Call list_files to see options.")
    return target.read_text(encoding="utf-8")


def search_files_text(root: Path, term: str, limit: int = 40) -> str:
    """Grep ``root`` for ``term``; return up to ``limit`` ``path:line`` snippets."""
    root = root.resolve()
    hits: list[str] = []
    for file in sorted(root.glob("**/*.md")):
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            if term.lower() in line.lower():
                rel = file.relative_to(root).as_posix()
                hits.append(f"{rel}:{i}: {line.strip()}")
    return "\n".join(hits[:limit]) if hits else f"No matches for '{term}'."


def payment_gateway_text(transaction_date: str, amount: float) -> str:
    """Simulated payment-processor lookup: status + refund eligibility."""
    return (
        f"Payment Gateway Response for {transaction_date} — ${amount:.2f}:\n"
        "- Transaction ID: txn_8f3k2j1\n"
        "- Status: SETTLED\n"
        "- Refund eligible: YES\n"
        "- Original payment method: Visa ending in 4242\n"
        "- Settlement date: 2025-02-02"
    )


def refund_text(amount: float, reason: str, customer_id: str) -> str:
    """Simulated refund processing through the payment gateway."""
    return (
        "Refund processed successfully:\n"
        f"- Customer: {customer_id}\n"
        f"- Amount: ${amount:.2f}\n"
        f"- Reason: {reason}\n"
        "- Refund ID: ref_9x2m4p7\n"
        "- ETA: 3-5 business days"
    )
