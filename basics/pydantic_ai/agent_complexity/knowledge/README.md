# Knowledge base (sample corpus)

Fake support-desk documents that the **Level 4** (agent harness) and **Level 5**
(multi-agent) examples read at runtime. This stands in for the real filesystem /
document store a support agent would investigate.

## Table of Contents

- [Contents](#contents)
- [How it is used](#how-it-is-used)
- [Note on formatting](#note-on-formatting)

## Contents

| Path | What it is |
|------|-----------|
| `customers/cust_12345.md` | Profile + transaction history for Sarah Johnson (contains the duplicate Feb-1 charge the examples resolve) |
| `policies/refund-policy.md` | Duplicate-charge, cancellation, and exception rules |
| `policies/escalation-matrix.md` | L1 / L2 / L3 escalation thresholds |
| `policies/subscription-management.md` | Plans, modifications, billing cadence |
| `templates/refund-confirmation.md` | Customer-facing refund email template |

## How it is used

Levels 4 and 5 expose sandboxed tools (`list_files`, `read_file`,
`search_files`) over this directory — see `../kb_tools.py`. The agents are **not**
told which files to open; they discover the structure, read what looks relevant,
cross-check the payment gateway, and act. File access is confined to this
directory; path-traversal attempts (`../…`) are rejected.

## Note on formatting

These files are **fixtures**, not documentation — the agents parse them as if
they were real support docs, and the tests assert on specific strings inside
them (e.g. `"Sarah Johnson"`, `"manager approval"`). They deliberately keep their
original support-doc shape rather than carrying a table of contents, so edits
here can change example and test behavior.
