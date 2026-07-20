// Run with: node --inspect-brk 03-event-loop-blocking.js
// The heartbeat below should tick every 200ms. Watch what happens to its
// timing once the "heavy job" kicks off.

const path = require("path");
const { Worker } = require("worker_threads");

let tick = 0;
const heartbeat = setInterval(() => {
  console.log(`heartbeat #${++tick} at ${new Date().toISOString()}`);
}, 200);

setTimeout(() => {
  console.log("Starting heavy job...");
  const worker = new Worker(path.join(__dirname, "heavy-worker.js"), {
    workerData: { durationMs: 1500 },
  });

  worker.on("message", (result) => {
    console.log("Heavy job done:", result);
    clearInterval(heartbeat);
    console.log("Done.");
  });

  worker.on("error", (err) => {
    console.error("Worker error:", err);
  });
}, 1000);
