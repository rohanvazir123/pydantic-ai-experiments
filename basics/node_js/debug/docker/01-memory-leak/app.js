// Simulates a document-processing worker with a result cache.
// Watch heapUsed and cache size as documents come in.

const processedDocs = new Map();

function processDocument(doc) {
  const extracted = { ...doc, extractedText: "x".repeat(500_000) };
  processedDocs.set(doc.id, extracted);
  return extracted;
}

let docId = 0;
const timer = setInterval(() => {
  processDocument({ id: docId++, title: `contract-${docId}.pdf` });
  const { heapUsed } = process.memoryUsage();
  console.log(
    `doc #${docId} | heapUsed: ${(heapUsed / 1024 / 1024).toFixed(1)} MB | cache size: ${processedDocs.size}`
  );

  if (docId >= 40) {
    clearInterval(timer);
    console.log("Done.");
  }
}, 150);
