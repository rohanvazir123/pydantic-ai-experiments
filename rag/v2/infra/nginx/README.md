# infra/nginx/

## Table of Contents

- [What This Is](#what-this-is)
- [SSE Configuration](#sse-configuration)

---

## What This Is

Nginx reverse proxy configuration. Terminates TLS, redirects HTTP → HTTPS, and proxies to the API and frontend containers. Nginx is the single entry point — no ports are published directly from the API or frontend containers.

---

## SSE Configuration

SSE (Server-Sent Events) routes require special Nginx settings to prevent buffering:

```nginx
location ~ ^/api/v1/(chat/stream|ingest/[^/]+/stream) {
    proxy_buffering    off;
    proxy_read_timeout 3600s;
    ...
}
```

Without `proxy_buffering off`, Nginx buffers the entire SSE stream before forwarding — the client receives nothing until the connection closes. Without a long `proxy_read_timeout`, Nginx closes SSE connections after 60s.
