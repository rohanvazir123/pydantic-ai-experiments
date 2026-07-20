// Run with: node --inspect-brk 02-unhandled-rejection.js
// Expected: prints "Batch complete" with all 5 records. It doesn't.

function fetchRecord(id) {
  return (id === 3
    ? Promise.reject(new Error(`record ${id} not found`))
    : Promise.resolve({ id, value: id * 10 })
  ).catch((err) => {
    console.error(`Error fetching record ${id}:`, err.message);
    return Promise.resolve({ id, value: null });
  });
}

async function processBatch(ids) {
  const results = [];
  for (const id of ids) {
    const record = await fetchRecord(id);
    results.push(record);
  }
  return results;
}

processBatch([1, 2, 3, 4, 5]).then((results) => {
  console.log("Batch complete:", results);
});
