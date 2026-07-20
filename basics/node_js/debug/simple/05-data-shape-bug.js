// Run with: node --inspect-brk 05-data-shape-bug.js
// Splits 10 items into batches of 3. Every item should appear exactly once
// across all batches, so the flattened total should be 10. It isn't.

function makeBatches(items, batchSize) {
  const batches = [];
  for (let i = 0; i < items.length; i += batchSize) {
    batches.push(items.slice(i, i + batchSize));
  }
  console.log("Batches created:", batches);
  return batches;
}

const items = Array.from({ length: 10 }, (_, i) => `item-${i}`);
const batches = makeBatches(items, 3);

console.log("Batches:", batches);
console.log("Total items across batches:", batches.flat().length, "(expected 10)");
