// Run with: node --inspect-brk 04-async-bug.js
// Supposed to process items, then print a summary once everything is done.
// The summary always prints empty (or incomplete) instead.

async function processItem(item) {
  await new Promise((resolve) => setTimeout(resolve, Math.random() * 200));
  console.log(`processed: ${item}`);
  return item.toUpperCase();
}

async function processAll(items) {
  const results = [];
  await Promise.all(items.map(async (item) => {
    const result = await processItem(item);
    results.push(result);
  }));
  console.log("Summary:", results);
}

processAll(["a", "b", "c", "d"]);
