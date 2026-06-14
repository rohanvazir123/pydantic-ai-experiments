# Knowledge Graph CLI

No standalone CLI scripts here — the KG retrieval CLI lives inside the `kg/legal/retrieval/` package.

## Available CLI commands

```bash
# Interactive REPL
python -m kg.legal.retrieval.cli

# Single question
python -m kg.legal.retrieval.cli --question "Which contracts is Amazon a party to?"

# Pipe questions from stdin
echo "Find all Delaware-governed contracts" | python -m kg.legal.retrieval.cli --stdin
```
