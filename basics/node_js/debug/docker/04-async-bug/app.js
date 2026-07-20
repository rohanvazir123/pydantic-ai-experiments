// Classifies a batch of contracts and reports how many were classified.
// Expected: "Classified 4 of 4". It reports something else.

const contracts = ["NDA-001", "MSA-014", "SOW-002", "NDA-005"];

async function classifyContract(id) {
  await new Promise((resolve) => setTimeout(resolve, 100 + Math.random() * 200));
  return { id, type: id.startsWith("NDA") ? "NDA" : "Other" };
}

async function classifyBatch(ids) {
  const classified = [];
  ids.forEach(async (id) => {
    const result = await classifyContract(id);
    classified.push(result);
  });
  return classified;
}

classifyBatch(contracts).then((results) => {
  console.log(`Classified ${results.length} of ${contracts.length} contracts:`, results);
});
