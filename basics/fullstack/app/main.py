"""Single entrypoint: `python -m app.main api|worker` (same image, two roles).

`api` serves the production ASGI app (`app.api.asgi:app`) via uvicorn — fine for dev
and single-process. For multi-worker production, run gunicorn directly:

    gunicorn app.api.asgi:app -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000
"""

from __future__ import annotations

import asyncio
import sys

import uvicorn

from app.config import get_settings
from app.temporal import worker as worker_mod


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "api"
    if mode == "api":
        settings = get_settings()
        uvicorn.run("app.api.asgi:app", host=settings.api_host, port=settings.api_port)
    elif mode == "worker":
        asyncio.run(worker_mod.main())
    else:
        print("usage: python -m app.main [api|worker]")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
