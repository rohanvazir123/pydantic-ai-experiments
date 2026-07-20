// Simulates a worker pulling extraction jobs off a queue. One bad job
// should not be allowed to take the whole worker down.

const jobs = [
  { id: 1, prompt: "extract parties" },
  { id: 2, prompt: "extract effective date" },
  { id: 3, prompt: "BROKEN_PROMPT" },
  { id: 4, prompt: "extract termination clause" },
];

function callLLM(job) {
  if (job.prompt === "BROKEN_PROMPT") {
    return Promise.reject(new Error(`job ${job.id}: LLM call failed`));
  }
  return Promise.resolve({ jobId: job.id, result: `parsed(${job.prompt})` });
}

async function runWorker() {
  console.log("Worker starting,", jobs.length, "jobs queued...");
  for (const job of jobs) {
    const result = await callLLM(job);
    console.log("Completed:", result);
  }
  console.log("All jobs complete.");
}

runWorker();
