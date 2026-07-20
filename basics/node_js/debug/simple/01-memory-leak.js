// Run with: node --inspect-brk 01-memory-leak.js
// Then attach VS Code ("Launch Current File" or attach on 9229) and watch
// process.memoryUsage().heapUsed climb every tick and never level off.

const requestLog = [];

function handleRequest(id) {
  const payload = Buffer.alloc(1_000_000); // pretend response body
  requestLog.push({ id, payload, handledAt: Date.now() });
  return payload.length;
}

let counter = 0;
const timer = setInterval(() => {
  handleRequest(counter++);
  const { heapUsed } = process.memoryUsage();
  console.log(
    `request #${counter} | heapUsed: ${(heapUsed / 1024 / 1024).toFixed(1)} MB | log size: ${requestLog.length}`
  );

  if (counter >= 30) {
    clearInterval(timer);
    console.log("Done. Heap grew roughly linearly with request count — why?");
  }
}, 100);
