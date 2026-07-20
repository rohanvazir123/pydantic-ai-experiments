// Summarizes parties extracted from a set of documents by an upstream LLM
// step. One document's extraction came back malformed.

const extractionResults = [
  { documentId: "doc-1", parties: ["Acme Corp", "Globex"] },
  { documentId: "doc-2", parties: ["Initech", "Umbrella"] },
  { documentId: "doc-3" },
  { documentId: "doc-4", parties: ["Stark Industries"] },
];

function summarizeParties(extraction) {
  const [primary, ...others] = extraction.parties;
  return `${extraction.documentId}: primary=${primary}, others=${others.length}`;
}

function processExtractions(results) {
  return results.map(summarizeParties);
}

console.log(processExtractions(extractionResults));
