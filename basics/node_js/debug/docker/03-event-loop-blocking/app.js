// Simulates a health-check tick alongside batch ingestion.
// The health tick should never noticeably stall.

const HEALTH_INTERVAL_MS = 200;

let tick = 0;
const health = setInterval(() => {
  console.log(`[health] ok tick #${++tick}`);
}, HEALTH_INTERVAL_MS);

function buildHugeContractBlob() {
  const docs = [];
  for (let i = 0; i < 200_000; i++) {
    docs.push({ id: i, clause: `Clause text for contract ${i}`.repeat(5) });
  }
  return JSON.stringify(docs);
}

function parseIncomingBatch() {
  const start = Date.now();
  let parsed;
  do {
    const blob = buildHugeContractBlob();
    parsed = JSON.parse(blob);
  } while (Date.now() - start < 1500);
  return parsed.length;
}

setTimeout(() => {
  console.log("Incoming batch received, parsing...");
  const count = parseIncomingBatch();
  console.log(`Parsed ${count} contract records.`);
}, 1000);

setTimeout(() => {
  clearInterval(health);
  console.log("Done.");
}, 3000);
