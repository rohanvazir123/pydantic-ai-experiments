# Take A — FAQ

## Table of Contents
- [What is schema\_dump.sql?](#what-is-schema_dumpsql)

---

## What is schema\_dump.sql?

An aside — a set of DBeaver-style introspection queries for the `meeting_analytics`
schema created by Take A. Not part of any pipeline. Useful for browsing the live
Postgres schema:

- List all tables in `meeting_analytics`
- Describe columns (type, nullability, defaults)
- Show constraints (PKs, FKs)
- Pull the distinct topic list from `summary_topics`
