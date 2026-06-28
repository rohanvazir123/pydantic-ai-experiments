"""Unit + perf tests for knowledge.validation.pii_scanner.

Split into two classes:
  TestPIIScanner      — correctness: detects known PII, passes clean text
  TestPIIScannerPerf  — timing: each scan must complete within budget
"""

import time

import pytest


# ── Correctness ───────────────────────────────────────────────────────────────

class TestPIIScanner:
    @pytest.mark.asyncio
    async def test_detects_email(self) -> None:
        from knowledge.validation.pii_scanner import scan_pii
        types = await scan_pii("Contact me at john.doe@example.com for details.")
        assert "EMAIL_ADDRESS" in types

    @pytest.mark.asyncio
    async def test_detects_phone(self) -> None:
        from knowledge.validation.pii_scanner import scan_pii
        types = await scan_pii("Call us at 555-867-5309.")
        assert "PHONE_NUMBER" in types

    @pytest.mark.asyncio
    async def test_detects_credit_card(self) -> None:
        from knowledge.validation.pii_scanner import scan_pii
        types = await scan_pii("My card number is 4111 1111 1111 1111.")
        assert "CREDIT_CARD" in types

    @pytest.mark.asyncio
    async def test_clean_text_returns_empty(self) -> None:
        from knowledge.validation.pii_scanner import scan_pii
        types = await scan_pii("The PTO policy allows 15 days per year.")
        assert types == []

    @pytest.mark.asyncio
    async def test_has_pii_true(self) -> None:
        from knowledge.validation.pii_scanner import has_pii
        assert await has_pii("Reach me at user@company.com") is True

    @pytest.mark.asyncio
    async def test_has_pii_false(self) -> None:
        from knowledge.validation.pii_scanner import has_pii
        assert await has_pii("No personal data here.") is False

    @pytest.mark.asyncio
    async def test_empty_string_is_clean(self) -> None:
        from knowledge.validation.pii_scanner import scan_pii
        assert await scan_pii("") == []

    @pytest.mark.asyncio
    async def test_returns_list_of_strings(self) -> None:
        from knowledge.validation.pii_scanner import scan_pii
        result = await scan_pii("Call 555-123-4567.")
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)


# ── Performance ───────────────────────────────────────────────────────────────
# Thresholds are deliberately generous to avoid flakiness on CI.
# The goal is catching regressions (e.g. accidental model download on every call),
# not benchmarking absolute throughput.

_INIT_BUDGET_S  = 5.0   # first call: loads spaCy model — allow up to 5s
_WARM_BUDGET_S  = 0.5   # subsequent calls: singleton reused — must be < 500ms
_LONG_BUDGET_S  = 1.0   # longer text scan budget


class TestPIIScannerPerf:
    @pytest.mark.asyncio
    async def test_first_call_within_init_budget(self) -> None:
        """First call initialises the AnalyzerEngine (spaCy load). Must finish within budget."""
        # Reset singleton so this test is self-contained
        import knowledge.validation.pii_scanner as mod
        mod._analyzer = None

        from knowledge.validation.pii_scanner import scan_pii
        t0 = time.perf_counter()
        await scan_pii("hello")
        elapsed = time.perf_counter() - t0
        assert elapsed < _INIT_BUDGET_S, (
            f"PII scanner init took {elapsed:.2f}s > budget {_INIT_BUDGET_S}s"
        )

    @pytest.mark.asyncio
    async def test_warm_call_under_500ms(self) -> None:
        """After initialisation, a scan on a short string must complete in < 500ms."""
        from knowledge.validation.pii_scanner import scan_pii
        await scan_pii("warmup")   # ensure singleton initialised

        t0 = time.perf_counter()
        await scan_pii("Contact hr@company.com for questions.")
        elapsed = time.perf_counter() - t0
        assert elapsed < _WARM_BUDGET_S, (
            f"Warm PII scan took {elapsed:.3f}s > budget {_WARM_BUDGET_S}s"
        )

    @pytest.mark.asyncio
    async def test_long_answer_under_1s(self) -> None:
        """A 500-word answer scan must complete in < 1s."""
        from knowledge.validation.pii_scanner import scan_pii
        text = ("The quarterly business review covers revenue, headcount growth, "
                "and strategic initiatives. " * 25)  # ~500 words

        t0 = time.perf_counter()
        await scan_pii(text)
        elapsed = time.perf_counter() - t0
        assert elapsed < _LONG_BUDGET_S, (
            f"Long-text PII scan took {elapsed:.3f}s > budget {_LONG_BUDGET_S}s"
        )

    @pytest.mark.asyncio
    async def test_repeated_calls_stable(self) -> None:
        """10 back-to-back scans should all complete within budget (no memory leak / reload)."""
        from knowledge.validation.pii_scanner import scan_pii
        texts = [
            "No PII here.",
            "Call 555-000-1234 please.",
            "Normal business text about quarterly goals.",
            "Send to admin@corp.com.",
            "The policy applies to all employees.",
        ] * 2  # 10 calls

        times = []
        for text in texts:
            t0 = time.perf_counter()
            await scan_pii(text)
            times.append(time.perf_counter() - t0)

        # No single call (after warmup) should spike above 1s
        for i, t in enumerate(times[1:], 1):
            assert t < _LONG_BUDGET_S, f"Call {i} took {t:.3f}s > {_LONG_BUDGET_S}s"
