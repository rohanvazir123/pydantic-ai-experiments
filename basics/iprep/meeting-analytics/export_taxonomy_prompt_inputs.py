"""
Export observed topics/key moments and an LLM prompt for taxonomy discovery.

This creates:
  - taxonomy_input.json: machine-readable counts from the dataset
  - taxonomy_prompt.md: prompt to paste into an LLM

Usage:
    python basics/iprep/i1/export_taxonomy_prompt_inputs.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "taxonomy_work"


PROMPT_TEMPLATE = """You are helping design an explainable taxonomy for a B2B SaaS transcript intelligence dataset.

Context:
The product analyzes meeting transcripts from support calls, external customer calls, and internal company calls.
The assignment requires:
1. Categorizing transcripts by topic or theme.
2. Generating sentiment analysis across call types.
3. Explaining trends in a way useful to product, support, sales, customer success, and engineering leaders.

Input:
I am providing observed topic tags extracted from meeting summaries, plus key moment signal types.

Task:
Create a compact taxonomy for classifying meetings.

Requirements:
- Create 6 to 10 high-level business themes.
- Themes should be useful for executive/product/support/sales/engineering analysis.
- Do not create one category per raw topic.
- Group semantically similar topics under broader themes.
- Use the counts to prioritize categories that explain the most meetings.
- Do not invent categories unsupported by the observed topic tags.
- If a category is based on low-frequency tags, mark it as lower confidence in taxonomy_notes.
- For each theme, provide theme_name, business_meaning, stakeholder_who_cares, keyword_hints, example_raw_topics, and common_risks_or_opportunities.
- Also create a call_type inference scheme with 4 to 7 call types.
- For each call type, provide call_type, meaning, keyword_hints, example_raw_topics, and ambiguity_notes.
- Prefer explainability over perfection.
- Avoid overly narrow keywords unless they strongly indicate the category.

Important:
The output will be reviewed by a human and then saved as taxonomy.json for deterministic code.
Make keyword_hints concise lowercase strings suitable for simple substring matching.

Return JSON only with this structure:

{
  "themes": [
    {
      "theme_name": "...",
      "business_meaning": "...",
      "stakeholder_who_cares": ["..."],
      "keyword_hints": ["..."],
      "example_raw_topics": ["..."],
      "common_risks_or_opportunities": ["..."]
    }
  ],
  "call_types": [
    {
      "call_type": "...",
      "meaning": "...",
      "keyword_hints": ["..."],
      "example_raw_topics": ["..."],
      "ambiguity_notes": "..."
    }
  ],
  "taxonomy_notes": ["..."],
  "suggested_review_steps": ["..."]
}

Observed topic tags with counts:
__TOPIC_COUNTS__

Observed key moment types with counts:
__KEY_MOMENT_COUNTS__
"""


def clean_topic(topic: str) -> str:
    return " ".join(topic.strip().lower().split())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_counts(dataset_dir: Path) -> dict[str, Any]:
    topic_counts: Counter[str] = Counter()
    key_moment_counts: Counter[str] = Counter()
    sentiment_counts: Counter[str] = Counter()
    meeting_count = 0

    for summary_path in sorted(dataset_dir.glob("*/summary.json")):
        meeting_count += 1
        summary = load_json(summary_path)
        topic_counts.update(clean_topic(str(topic)) for topic in summary.get("topics", []))
        sentiment = summary.get("overallSentiment")
        if sentiment:
            sentiment_counts[str(sentiment)] += 1
        for moment in summary.get("keyMoments", []):
            moment_type = moment.get("type")
            if moment_type:
                key_moment_counts[str(moment_type)] += 1

    return {
        "meeting_count": meeting_count,
        "topic_counts": [
            {"topic": topic, "count": count}
            for topic, count in topic_counts.most_common()
        ],
        "key_moment_counts": [
            {"moment_type": moment_type, "count": count}
            for moment_type, count in key_moment_counts.most_common()
        ],
        "overall_sentiment_counts": [
            {"overall_sentiment": sentiment, "count": count}
            for sentiment, count in sentiment_counts.most_common()
        ],
    }


def format_counts(rows: list[dict[str, Any]], label_key: str) -> str:
    return "\n".join(f"- {row[label_key]}: {row['count']}" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = collect_counts(dataset_dir)
    input_path = output_dir / "taxonomy_input.json"
    prompt_path = output_dir / "taxonomy_prompt.md"

    input_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    prompt = PROMPT_TEMPLATE.replace(
        "__TOPIC_COUNTS__",
        format_counts(counts["topic_counts"], "topic"),
    ).replace(
        "__KEY_MOMENT_COUNTS__",
        format_counts(counts["key_moment_counts"], "moment_type"),
    )
    prompt_path.write_text(prompt, encoding="utf-8")

    print(f"Wrote {input_path}")
    print(f"Wrote {prompt_path}")
    print("\nAfter the LLM returns taxonomy JSON, save it as:")
    print(f"  {SCRIPT_DIR / 'taxonomy.json'}")
    print("\nThen run:")
    print("  python basics\\iprep\\i1\\load_functional_schema_to_postgres.py --reset")


if __name__ == "__main__":
    main()
