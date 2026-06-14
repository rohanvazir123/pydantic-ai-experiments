"""SMTP alert sender — fires on circuit open, DLQ push, budget breach.

All alerts go to rohan.vazirani@gmail.com (from settings.alert_email).
Non-blocking: always wrapped in asyncio.create_task().
Fallback: if SMTP unreachable, writes to logs/alerts.jsonl + stderr.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from knowledge.config.settings import Settings, load_settings

logger = logging.getLogger(__name__)

AlertSeverity = Literal["CRITICAL", "WARNING", "INFO"]

_FALLBACK_FILE = Path(__file__).parent.parent.parent / "logs" / "alerts.jsonl"


async def send_alert(
    severity: AlertSeverity,
    code: str,
    detail: dict,
    settings: Settings | None = None,
) -> None:
    """Send an alert email. Non-blocking — always call via asyncio.create_task().

    Falls back to logs/alerts.jsonl when SMTP is unreachable.
    """
    _settings = settings or load_settings()

    subject = f"[RAG] {severity} — {code}"
    body = (
        f"Time:     {datetime.now(UTC).isoformat()}\n"
        f"Severity: {severity}\n"
        f"Code:     {code}\n"
    )
    for k, v in detail.items():
        body += f"{k.capitalize()}: {v}\n"

    try:
        from email.message import EmailMessage

        import aiosmtplib

        if not _settings.smtp_host or not _settings.smtp_user:
            raise RuntimeError("SMTP not configured")

        msg = EmailMessage()
        msg["From"]    = _settings.smtp_from
        msg["To"]      = _settings.alert_email
        msg["Subject"] = subject
        msg.set_content(body)

        await aiosmtplib.send(
            msg,
            hostname=_settings.smtp_host,
            port=_settings.smtp_port,
            username=_settings.smtp_user,
            password=_settings.smtp_password,
            start_tls=True,
            timeout=10,
        )
        logger.info("Alert sent: %s — %s", severity, code)

    except Exception as exc:
        # Fallback to JSONL file
        logger.warning("SMTP failed (%s) — writing alert to %s", exc, _FALLBACK_FILE)
        _write_fallback(severity, code, detail, body)


def _write_fallback(severity: str, code: str, detail: dict, body: str) -> None:
    try:
        _FALLBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        entry = json.dumps({
            "ts": datetime.now(UTC).isoformat(),
            "severity": severity, "code": code, **detail,
        })
        with _FALLBACK_FILE.open("a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception as exc:
        logger.error("Alert fallback write failed: %s\n%s", exc, body)


def alert(severity: AlertSeverity, code: str, **detail: object) -> None:
    """Fire-and-forget alert. Call from sync or async context."""
    asyncio.create_task(send_alert(severity, code, dict(detail)))
