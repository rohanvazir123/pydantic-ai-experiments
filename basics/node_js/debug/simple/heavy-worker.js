const { parentPort, workerData } = require("worker_threads");

function heavyJob(durationMs) {
  const end = Date.now() + durationMs;
  let total = 0;
  let i = 0;
  while (Date.now() < end) {
    total += Math.sqrt(i) * Math.sin(i);
    i++;
  }
  return total;
}

const result = heavyJob(workerData.durationMs);
parentPort.postMessage(result);
