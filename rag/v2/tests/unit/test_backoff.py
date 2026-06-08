"""Unit tests for knowledge.bus.backoff."""

import pytest
from knowledge.bus.backoff import exponential_backoff


class TestExponentialBackoff:
    def test_attempt_1_equals_base(self) -> None:
        # With jitter_factor=0 the result is exactly base_s
        result = exponential_backoff(1, base_s=5.0, jitter_factor=0.0)
        assert result == pytest.approx(5.0)

    def test_attempt_2_doubles(self) -> None:
        result = exponential_backoff(2, base_s=5.0, multiplier=2.0, jitter_factor=0.0)
        assert result == pytest.approx(10.0)

    def test_attempt_3_quadruples(self) -> None:
        result = exponential_backoff(3, base_s=5.0, multiplier=2.0, jitter_factor=0.0)
        assert result == pytest.approx(20.0)

    def test_max_cap_respected(self) -> None:
        # attempt=10 would give 5 * 2^9 = 2560 but max_s=125
        result = exponential_backoff(10, base_s=5.0, max_s=125.0, jitter_factor=0.0)
        assert result == pytest.approx(125.0)

    def test_jitter_within_bounds(self) -> None:
        for attempt in range(1, 4):
            for _ in range(50):
                raw = min(5.0 * (2.0 ** (attempt - 1)), 125.0)
                result = exponential_backoff(attempt, base_s=5.0, jitter_factor=0.15)
                assert raw <= result <= raw + 0.15 * raw + 1e-9

    def test_zero_jitter_is_deterministic(self) -> None:
        r1 = exponential_backoff(2, jitter_factor=0.0)
        r2 = exponential_backoff(2, jitter_factor=0.0)
        assert r1 == r2

    def test_attempt_below_1_treated_as_1(self) -> None:
        result = exponential_backoff(0, base_s=5.0, jitter_factor=0.0)
        assert result == pytest.approx(5.0)

    def test_result_always_positive(self) -> None:
        for attempt in range(1, 10):
            assert exponential_backoff(attempt) > 0
