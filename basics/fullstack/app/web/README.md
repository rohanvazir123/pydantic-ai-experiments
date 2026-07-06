# app/web/

Server-rendered frontend assets (Jinja2 templates).

## Table of Contents

- [Purpose](#purpose)
- [Templates](#templates)

## Purpose

The minimal HTML UI rendered by the FastAPI web routes — no SPA, no build step.

## Templates

| File | Rendered by | Shows |
|------|-------------|-------|
| `templates/index.html` | `GET /` | New-order form + table of all orders. |
| `templates/order.html` | `GET /orders/{id}` | One order's details + status; approval hint for high-value orders. |
