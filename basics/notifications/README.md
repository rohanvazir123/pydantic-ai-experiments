# Notification / real-time transport patterns — quick reference

Four ways a server pushes/serves updates, cheapest → richest:
**HTTP req/res** (pull, one-shot) · **long poll** (pull, held open) ·
**SSE** (server→client stream, one-way) · **WebSocket** (full duplex).

Each pattern below shows **client** and **server**, in **JS and Python**.

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
// SERVER (JS, Express) — hold ~25s waiting for an event, else return empty
app.get('/api/updates', async (req, res) => {
  const events = await waitForEvent(req.query.since, 25_000); // resolves early on new data
  res.json({ events, cursor: nextCursor(events) });
});
```
```py
# SERVER (Python, FastAPI)
@app.get('/api/updates')
async def updates(since: str):
    events = await wait_for_event(since, timeout=25)  # returns early when something arrives
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

## Picking one (the interview answer)

- **Reachability:** SSE & long-poll are plain HTTP — sail through proxies/firewalls. WS needs an `Upgrade` and proxy support (`proxy_buffering off` for SSE too).
- **Reconnect:** SSE auto-reconnects with `Last-Event-ID`; WS you reconnect yourself.
- **Scale cost:** each SSE/WS pins a connection (and often a worker) — plan for connection count, not just RPS. Long poll trades held connections for repeated requests.
- **Voice/chat mapping:** stream agent tokens/status → **SSE**; interactive turn + barge-in (client interrupts) → **WebSocket**.
