"""Shared fixtures for incident response workflow tests."""
from __future__ import annotations

import pytest

from .models import IncidentAlert

ALERT_HIGH = IncidentAlert(
    alert_id="alert-001",
    service="payment-service",
    error_rate=0.45,
    latency_p99_ms=2500,
    description="Spike in 5xx errors after deployment v2.3.1",
)

ALERT_CRITICAL = IncidentAlert(
    alert_id="alert-002",
    service="auth-service",
    error_rate=0.80,
    latency_p99_ms=5000,
    description="Complete auth failure — all login requests failing",
)
