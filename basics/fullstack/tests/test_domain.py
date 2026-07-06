"""Unit tests for pure domain logic (no I/O)."""

from __future__ import annotations

import pytest

from app import domain


def test_line_total() -> None:
    assert domain.line_total_cents(3, 500) == 1500
    assert domain.line_total_cents(0, 500) == 0


@pytest.mark.parametrize(
    ("item", "qty", "price", "expected_errors"),
    [
        ("Widget", 1, 100, 0),
        ("", 1, 100, 1),
        ("   ", 1, 100, 1),
        ("Widget", 0, 100, 1),
        ("Widget", -2, 100, 1),
        ("Widget", 1, -1, 1),
        ("", 0, -1, 3),
    ],
)
def test_validate_order(item: str, qty: int, price: int, expected_errors: int) -> None:
    assert len(domain.validate_order(item, qty, price)) == expected_errors


def test_is_high_value_boundary() -> None:
    assert domain.is_high_value(domain.HIGH_VALUE_THRESHOLD_CENTS) is False  # exactly at limit
    assert domain.is_high_value(domain.HIGH_VALUE_THRESHOLD_CENTS + 1) is True
    assert domain.is_high_value(0) is False
