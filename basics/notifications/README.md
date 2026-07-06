# Notification / real-time transport patterns — quick reference

Four ways a server pushes/serves updates, cheapest → richest:
**HTTP req/res** (pull, one-shot) · **long poll** (pull, held open) ·
**SSE** (server→client stream, one-way) · **WebSocket** (full duplex).

Each pattern below shows **client** and **server**, in **JS and Python**.
Server handle is instantiated once: `app = FastAPI()` (Python) · `const app = express()` (JS).

| Pattern | Direction | Connection | Use when |
|---------|-----------|------------|----------|
| HTTP req/res | client→server→client | new per request | plain CRUD, on-demand reads |
| Long poll | client asks, server holds | reopened each cycle | near-real-time on legacy/proxied infra, no SSE/WS |
| SSE | server→client only | one long-lived HTTP | token/status streams, notifications (voice/chat token push) |
| WebSocket | bidirectional | one long-lived TCP | live chat, barge-in, presence, anything client also sends |

---

## 1. HTTP request/response

```js
// CLIENT (JS)
const res = await fetch('/api/orders/42');
console.log(await res.json());
```
```py
# CLIENT (Python, httpx)
r = await client.get('/api/orders/42')
print(r.json())
```
```js
// SERVER (JS, Express)
app.get('/api/orders/:id', (req, res) => res.json({ id: req.params.id, status: 'confirmed' }));
```
```py
# SERVER (Python, FastAPI)
@app.get('/api/orders/{oid}')
async def get_order(oid: int):
    return {'id': oid, 'status': 'confirmed'}
```

## 2. Long polling — client re-asks, server holds the request until there's news

```js
// CLIENT (JS)
while (true) {
  const res = await fetch('/api/updates?since=' + cursor); // server blocks until data/timeout
  const data = await res.json();
  if (data.events) handle(data.events);
  cursor = data.cursor;
}
```
```py
# CLIENT (Python, httpx)
while True:
    r = await client.get('/api/updates', params={'since': cursor})
    data = r.json()
    if data['events']:
        handle(data['events'])
    cursor = data['cursor']
```
```js
// SERVER (JS, Express) — hold ~50s (under the 60s infra timeout), else return empty
app.get('/api/updates', async (req, res) => {
  const events = await waitForEvent(req.query.since, 50_000); // resolves early on new data
  res.json({ events, cursor: nextCursor(events) });
});
```
```py
# SERVER (Python, FastAPI)
@app.get('/api/updates')
async def updates(since: str):
    events = await wait_for_event(since, timeout=50)  # under the 60s infra timeout; returns early on data
    return {'events': events, 'cursor': next_cursor(events)}
```

## 3. SSE — one-way server stream over plain HTTP (`text/event-stream`)

```js
// CLIENT (JS) — EventSource: named events + auto-reconnect built in
const es = new EventSource('/api/updates');
es.onmessage = (event) => console.log(JSON.parse(event.data));
es.addEventListener('done', () => es.close());
```
```py
# CLIENT (Python, httpx-sse)
async with aconnect_sse(client, 'GET', '/api/updates') as es:
    async for sse in es.aiter_sse():
        if sse.event == 'done':
            break
        print(json.loads(sse.data))
```
```js
// SERVER (JS, Node) — frame = "data: ...\n\n"
app.get('/api/updates', async (req, res) => {
  res.set({ 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
  for await (const ev of eventSource()) res.write(`data: ${JSON.stringify(ev)}\n\n`);
  res.write('event: done\ndata: {}\n\n');
});
```
```py
# SERVER (Python, FastAPI)
from fastapi.responses import StreamingResponse

async def gen():
    async for ev in event_source():
        yield f"data: {json.dumps(ev)}\n\n"
    yield "event: done\ndata: {}\n\n"

@app.get('/api/updates')
async def updates():
    return StreamingResponse(gen(), media_type='text/event-stream')
```

## 4. WebSocket — full duplex; client can also send

```js
// CLIENT (JS)
const ws = new WebSocket('wss://api.example.com/socket');
ws.onmessage = (event) => console.log(JSON.parse(event.data));
ws.send(JSON.stringify({ type: 'ping' })); // client can also send
```
```py
# CLIENT (Python, websockets)
async with websockets.connect('wss://api.example.com/socket') as ws:
    await ws.send(json.dumps({'type': 'ping'}))  # client can also send
    async for msg in ws:
        print(json.loads(msg))
```
```js
// SERVER (JS, ws)
new WebSocketServer({ port: 8080 }).on('connection', (ws) => {
  ws.on('message', (data) => ws.send(JSON.stringify({ echo: JSON.parse(data) })));
});
```
```py
# SERVER (Python, FastAPI)
@app.websocket('/socket')
async def socket(ws: WebSocket):
    await ws.accept()
    while True:
        msg = await ws.receive_json()      # read what the client sends
        await ws.send_json({'echo': msg})  # push back anytime
```

---

## Picking one (how to choose)

- **Reachability:** SSE & long-poll are plain HTTP — sail through proxies/firewalls. WS needs an `Upgrade` and proxy support (`proxy_buffering off` for SSE too).
- **Reconnect:** SSE auto-reconnects with `Last-Event-ID`; WS you reconnect yourself.
- **Scale cost:** each SSE/WS pins a connection (and often a worker) — plan for connection count, not just RPS. Long poll trades held connections for repeated requests.
- **Voice/chat mapping:** stream agent tokens/status → **SSE**; interactive turn + barge-in (client interrupts) → **WebSocket**.

## Gotchas

**Long poll**
- **Hold < the tightest timeout in the path** (LB/proxy/router) or you get 504s instead of clean empties; here 50s under a 60s ceiling.
- **Client timeout must exceed the hold** or the client aborts a valid in-progress request.
- **Async server only.** A held request on a sync worker-per-request model (gunicorn sync, default `--timeout 30`) ties up a whole worker → pool exhaustion. Cheap on uvicorn/async (just a suspended coroutine).

**SSE**
- **~6 connections/domain on HTTP/1.1** (browser cap) — 6 tabs and the 7th hangs. HTTP/2 multiplexing removes it.
- **Disable proxy buffering** (`proxy_buffering off` / `X-Accel-Buffering: no`) or events are buffered and never flush to the client.
- **One-way + text only.** Client can't reply on the same channel (needs a separate POST); no binary frames.
- **Send heartbeats** (`: ping\n\n`) so idle intermediaries don't close the stream; handle `Last-Event-ID` on reconnect for resume/dedup.

**WebSocket**
- **No auto-reconnect** — you implement backoff + resume yourself.
- **Broadcast needs a backplane.** With N server instances, a message on server A won't reach clients on server B without a Redis/pub-sub fan-out; sticky sessions usually required.
- **Auth on connect, not per-message**, and browsers can't set headers on the `WebSocket` constructor — pass the token via query param / subprotocol / cookie.
- **Heartbeats still required** (ping/pong) — idle timeouts apply to WS too; and stateful connections make deploys harder (drain on restart).

## Which transport for which system

> These three are **client-facing** transports (browser/mobile ↔ your server). When the
> consumer is another **server**, the answer is usually **webhooks or a message queue**,
> not any of them.

Decision tree:
- **Server-to-server?** → **webhook / queue** (durable, retried, signed) — not SSE/WS/long-poll.
- **Browser-facing?** one-way live view → **SSE**; interactive / client sends a lot → **WebSocket**; just "is it done?" → **request/response** (poll if needed).

**Payment processor (Stripe)** — optimizes for durability, not latency; never miss "charge succeeded."
- Backbone = **webhooks**: signed (HMAC `Stripe-Signature`), retried with backoff for ~3 days, idempotent by event id. A dropped socket frame is gone; a webhook is redelivered until you `200`.
- Checkout client = **request/response + redirects** (3DS); the browser **polls** your API for final status.
- SSE appears only as a *live view* on the merchant dashboard. **No WebSocket** — there's no high-frequency client→server traffic to justify its stateful cost.
- Lesson: the **durable event log is the truth; any socket is just a reconcilable window** onto it.

**RAG / LLM chatbot** — chose **SSE over WebSocket**.
- Traffic is overwhelmingly **server→client**: streaming generated tokens as they're produced. One-way.
- The client sends infrequently (one prompt per turn) — a plain `POST /chat` handles that; no need for a persistent bidirectional channel for an occasional send.
- SSE is simpler and more robust: plain HTTP (sails through proxies/CDNs), built-in auto-reconnect (`Last-Event-ID`), no `Upgrade` handshake, no sticky sessions or pub/sub backplane.
- WebSocket would be overkill — stateful connections + a backplane + manual reconnect for what is essentially one-way streaming.
- Pattern: `POST /chat` (prompt) → **SSE** stream of `token` / `tool` / `done` events; cancel = close the stream or a separate `POST /cancel`.
- **When WS would flip it:** genuine bidirectional real-time — voice with **barge-in** (client interrupts mid-generation), or two-way typing/presence. For text chat with token streaming, SSE wins.

**Loan approval (long-running, human-in-the-loop, multi-day SLA)** — match transport to *process lifetime*.
- Applicant status → **polling** `GET` + out-of-band notify (email/push, or a **webhook** for partner systems); optional **short-lived SSE** only during the seconds-long auto-decision window.
- Underwriter queue → **SSE** push of newly-queued items + a plain `PUT` for the decision.
- Internal step→step → the **workflow engine / queue**, not a client transport.
- **Anti-pattern:** holding a WS/SSE across a days-long human step — decouple durable backend from ephemeral view.

**Three axes that decide it:**
1. **Direction / interactivity** — don't pay WebSocket's stateful cost unless the client genuinely sends.
2. **Delivery guarantee** — SSE/WS are best-effort and ephemeral; *must-not-lose* ⇒ durable webhook/queue + a `GET` to reconcile.
3. **Process vs connection lifetime** — match them; a long process ⇒ decouple, don't hold a socket.

One-liner: **latency-critical + interactive → WebSocket; live one-way view → SSE; must-not-lose or server-to-server → webhook/queue + a `GET` to reconcile; everything else → request/response.**
