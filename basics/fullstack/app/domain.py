"""Pure business logic — no I/O, no framework imports.

Everything here is deterministic and trivially unit-testable, and is safe to call
from inside a Temporal workflow (which forbids I/O and non-determinism).
"""

from __future__ import annotations

from enum import StrEnum

# Orders above this total require human approval before they can be confirmed.
HIGH_VALUE_THRESHOLD_CENTS = 100_000  # $1,000.00


class OrderStatus(StrEnum):
    PENDING = "pending"                    # created, workflow not yet advanced
    PROCESSING = "processing"              # workflow running
    AWAITING_APPROVAL = "awaiting_approval"  # high-value, suspended for a human
    CONFIRMED = "confirmed"                # terminal — success
    REJECTED = "rejected"                  # terminal — invalid, denied, or SLA breach


class Decision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


def line_total_cents(quantity: int, unit_price_cents: int) -> int:
    """Total price for a line item."""
    return quantity * unit_price_cents


def validate_order(item: str, quantity: int, unit_price_cents: int) -> list[str]:
    """Return a list of validation errors; empty list means the order is valid."""
    errors: list[str] = []
    if not item or not item.strip():
        errors.append("item must not be empty")
    if quantity <= 0:
        errors.append("quantity must be > 0")
    if unit_price_cents < 0:
        errors.append("unit_price_cents must be >= 0")
    return errors


def is_high_value(total_cents: int) -> bool:
    """High-value orders route to the human-approval gate."""
    return total_cents > HIGH_VALUE_THRESHOLD_CENTS
