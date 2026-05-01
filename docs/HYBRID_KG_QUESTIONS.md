# 100 Questions Requiring Hybrid Search + Knowledge Graph (Apache AGE)

These questions cannot be answered by either hybrid vector search or graph traversal alone.
Each one requires both legs: dense + BM25 retrieval over chunk embeddings **and** Cypher
graph traversal over the Apache AGE `legal_graph` built from the CUAD corpus
(509 contracts → 13,262 entities + 13,603 relationships).

---

## Party & Contract Relationship Queries (1–15)

1. Which contracts have Amazon as a party, and what termination clauses apply to them?
2. Find all agreements where the same two parties appear in both a licensing deal and a services agreement.
3. Which entities appear as a party in more than five contracts, and what types of IP clauses do those contracts contain?
4. What are the governing-law clauses in all contracts where Apple is mentioned as a licensee?
5. Which contracts connect a software company based in California with a distributor governed by New York law?
6. Find all parties that are listed in contracts with perpetual license grants — what other clause types appear in those same contracts?
7. Which company names appear as parties in contracts that also contain a non-compete restriction clause and a Delaware jurisdiction clause?
8. What is the renewal term structure for all contracts where Google LLC is a named party?
9. List all contracts where a "licensor" party relationship exists alongside an uncapped liability clause.
10. Which parties co-appear in both a technology transfer agreement and a manufacturing agreement?
11. Which contracts have three or more distinct parties, and what unusual clause types appear only in those multi-party agreements?
12. Find all contracts where a parent company and a wholly owned subsidiary both appear as separate parties.
13. Which parties are linked by both a PARTY_TO relationship in the KG and appear in the same chunk discussing "change of control"?
14. What are the indemnification obligations in contracts where the licensor is headquartered outside the United States?
15. Find all contracts that list a pharmaceutical company as a party and also contain a most-favored-nation clause.

---

## Jurisdiction & Governing Law (16–28)

16. Which jurisdictions govern the most contracts in the corpus, and what IP clause patterns are most common under each jurisdiction?
17. Find all contracts governed by English law that also contain a restriction on sublicensing.
18. Which contracts have a governing law clause pointing to Delaware but were signed by parties incorporated in another state?
19. List all termination-for-convenience clauses in contracts governed by California law.
20. Are there contracts where the dispute resolution clause specifies a different jurisdiction than the governing law clause?
21. Which jurisdiction sees the highest frequency of uncapped indemnification clauses?
22. Find contracts governed by New York law that also have a non-solicitation clause — what parties are involved?
23. Which contracts have no explicit governing law clause, and what other clause types are present in them?
24. How do liability caps differ between contracts governed by Delaware vs. those governed by California?
25. Which governing-law jurisdictions are exclusively used by technology licensing agreements vs. supply agreements?
26. Find all contracts where jurisdiction is a European country and the licensor is a US entity.
27. Which contracts pair a Texas governing law clause with an IP ownership clause that favors the licensee?
28. What termination triggers appear most often in contracts governed by UK law?

---

## IP & License Clause Deep Dives (29–42)

29. Find all contracts that grant an exclusive license but also contain a clause allowing the licensor to use competing technology.
30. Which contracts grant a license to "all improvements" and also contain a revenue-sharing clause?
31. What are the scope limitations in IP clauses for contracts where both parties are publicly traded companies?
32. Find contracts where the license grant is field-of-use restricted AND the governing law is Delaware — what are the exact field restrictions?
33. Which contracts contain both a patent license clause and a trademark license clause? Are the terms symmetric?
34. Find all agreements where the IP clause grants rights that survive termination — what other survival clauses are present?
35. Which contracts have an IP assignment (not just a license) clause, and who are the assignors and assignees?
36. What source-code escrow provisions appear in contracts that also contain a perpetual license grant?
37. Find all contracts where the licensee gets rights to sublicense AND the contract contains a most-favored-nation clause.
38. Which contracts have an IP clause that excludes pre-existing IP, and what is the governing law for those agreements?
39. Find all agreements where the IP clause's effective date is different from the contract execution date — what other date entities link to those contracts?
40. Which contracts have both a joint-development IP clause and a separate ownership clause that conflicts with it?
41. Find all contracts that contain a "work-for-hire" clause — which parties are listed as the employer vs. contractor?
42. What license grant language is used in contracts where the licensee is also listed as a distributor in the KG?

---

## Termination & Renewal (43–55)

43. Find all contracts where the termination clause includes a cure period and the governing law is California — what is the typical cure window?
44. Which contracts have both an automatic renewal clause and a termination-for-breach clause? How do these interact?
45. Find all contracts with a termination-for-insolvency clause — which parties are involved and what other protective clauses exist?
46. Which contracts allow termination without cause with less than 30 days' notice?
47. Find contracts where termination triggers the assignment of IP back to the licensor — what IP was originally licensed?
48. Which contracts have a "change of control" termination right — who holds that right, and what is the governing law?
49. Find all evergreen contracts (no fixed end date) that also have a termination-for-convenience clause — what parties are in those agreements?
50. Which contracts have a post-termination license clause, and how long does the license survive after termination?
51. Find all contracts where the renewal term differs from the initial term — what clause types are associated with these renewal differences?
52. Which contracts allow partial termination (termination of specific licensed products/services), and who can invoke it?
53. Find all contracts with a right of first refusal upon termination — which parties hold the ROFR, and what asset does it cover?
54. Which contracts have a termination clause triggered by a regulatory event (FDA, FCC, etc.)?
55. Find all contracts where termination for breach requires a minimum dollar threshold to be crossed.

---

## Restriction & Non-Compete Clauses (56–65)

56. Which contracts contain a non-compete clause that explicitly names competing products or companies?
57. Find all agreements where the restriction clause is broader than the scope of the license grant — is this pattern jurisdiction-specific?
58. Which parties appear as the restricted party in more than two separate non-compete agreements?
59. Find contracts where a non-solicitation clause is the only restriction and no non-compete exists — what types of agreements are these?
60. Which contracts have a geographic restriction clause that limits activity in specific countries — list the countries mentioned.
61. Find all contracts where a restriction clause has a sunset date that predates the contract expiration — what happens after sunset?
62. Which contracts contain both a non-compete and a non-disparagement clause? What parties are bound by both?
63. Find all restriction clauses in contracts where a party relationship in the KG shows a competitor relationship.
64. Which contracts have a restriction on assignment that carves out affiliate transfers — who defines "affiliate" in those agreements?
65. Find all contracts where the restriction clause explicitly references a specific patent or trademark number.

---

## Liability & Indemnification (66–75)

66. Which contracts have an uncapped indemnification obligation, and what events trigger it?
67. Find all contracts where the liability cap is expressed as a multiple of fees paid — what is the most common multiplier?
68. Which contracts have mutual indemnification vs. one-sided indemnification — does the pattern differ by jurisdiction?
69. Find all contracts where the indemnification clause covers third-party IP claims — which party bears that risk?
70. Which contracts have a consequential damages waiver that is carved out for data breach or confidentiality violations?
71. Find all contracts where the liability clause references specific dollar amounts — what are those amounts and in what currency?
72. Which contracts have both a limitation of liability clause and an indemnification clause that effectively eliminates the cap?
73. Find all contracts where IP indemnification is tied to the licensor's knowledge ("to licensor's knowledge" qualifier).
74. Which parties appear most frequently as indemnifying parties across the corpus?
75. Find all contracts where the indemnification clause survives termination — how long does it survive?

---

## Cross-Domain / Multi-Hop Graph Queries (76–88)

76. Starting from a "Party" node in the KG, find all contracts two hops away that share a governing law with a contract the party is not directly part of.
77. Which entity types cluster around contracts that also contain change-of-control clauses — do certain party types or jurisdictions co-occur?
78. Find all date entities linked to renewal events, then retrieve the contract chunks that discuss the notice requirements for those renewals.
79. Which parties in the KG are connected to contracts in three or more different clause categories simultaneously?
80. Find all "LicenseClause" nodes whose linked contracts also have a "RestrictionClause" node — are the restriction terms consistent with the license scope?
81. Starting from a Delaware jurisdiction node, traverse to all contracts, then retrieve the chunks that explain why Delaware was chosen as governing law.
82. Which parties appear in both technology license contracts and distribution agreements — how do their obligations differ across contract types?
83. Find all "TerminationClause" nodes where the linked contract also contains a "Date" entity more than 10 years in the future — what are those future dates?
84. Which contracts are connected to both a "LiabilityClause" capping liability AND a separate indemnification clause with no cap — identify the contradiction in text.
85. Find all contracts where the same party appears as both licensor and licensee (cross-licensing) — what IP is being exchanged?
86. Starting from an "IPClause" node for a specific patent, traverse to all other contracts referencing the same patent family.
87. Which "Jurisdiction" nodes in the KG have no associated date entities (no signed/effective dates) — are those contracts incomplete?
88. Find all entities connected to the "HAS_RESTRICTION" relationship, then retrieve text chunks discussing carve-outs or exceptions to those restrictions.

---

## NL-to-Cypher / Analytical (89–100)

89. How many unique governing jurisdictions are present in the corpus, and what is the average number of clause types per jurisdiction?
90. What is the distribution of contract types (license, distribution, services, manufacturing) across the knowledge graph?
91. Which clause types most frequently co-occur in the same contract — output the top-10 co-occurrence pairs?
92. Find contracts where the number of named parties exceeds the median across all contracts — what clause types are over-represented?
93. Which year has the highest density of contracts with both an IP clause and a termination-for-change-of-control clause?
94. What is the average number of restriction clauses per contract for agreements governed by California vs. New York law?
95. Find contracts where entity count in the KG exceeds 15 — are those contracts structurally more complex in the text?
96. Which relationship types in the KG (PARTY_TO, HAS_LICENSE, etc.) have the highest cross-contract reuse (same entity, different contracts)?
97. Build a subgraph of all contracts connected to a single large technology company — which clause types are most variable across those contracts?
98. Find all contracts where the IP clause grants rights in a field that conflicts with a restriction clause in the same agreement.
99. Which contracts have the highest ratio of KG entities to total chunk count — are these contracts more structured or more clause-heavy?
100. Identify contracts where the governing law entity has no path to any termination clause entity in the KG — what does this structural gap reveal about those agreements?

---

## KG Schema Reference

| Entity type | Relationship | Description |
|---|---|---|
| `Party` | `PARTY_TO` | Named party in a contract |
| `Jurisdiction` | `GOVERNED_BY_LAW` | Governing law / dispute forum |
| `Date` | `HAS_DATE` | Execution, effective, expiry dates |
| `LicenseClause` | `HAS_LICENSE` | License grant provisions |
| `TerminationClause` | `HAS_TERMINATION` | Termination triggers and procedures |
| `RestrictionClause` | `HAS_RESTRICTION` | Non-compete, non-solicit, exclusivity |
| `IPClause` | `HAS_IP_CLAUSE` | IP ownership, assignment, work-for-hire |
| `LiabilityClause` | `HAS_LIABILITY` | Caps, indemnification, warranties |
| `Clause` | `HAS_CLAUSE` | Generic / uncategorised clause |

Source: `rag/knowledge_graph/cuad_kg_builder.py` — CUAD 35+ annotation types mapped to the 9 entity types above.
