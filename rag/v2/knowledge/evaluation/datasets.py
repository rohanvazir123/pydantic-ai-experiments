"""Gold dataset loader — JSONL files + PostgreSQL mirror."""

import logging
from pathlib import Path

from knowledge.evaluation.schemas import GoldSample

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class GoldDataset:
    """Load, save, and validate gold evaluation samples."""

    def __init__(self, corpus_id: str) -> None:
        self.corpus_id = corpus_id
        self._samples: list[GoldSample] = []

    def load_from_file(self) -> "GoldDataset":
        """Load samples from evaluation/data/{corpus_id}.jsonl."""
        path = DATA_DIR / f"{self.corpus_id}.jsonl"
        if not path.exists():
            logger.warning("No gold dataset found at %s", path)
            return self

        samples = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(GoldSample.model_validate_json(line))
            except Exception as exc:
                logger.warning("Skipping malformed gold sample: %s", exc)

        self._samples = samples
        logger.info("Loaded %d gold samples for corpus '%s'", len(samples), self.corpus_id)
        return self

    def load_from_python_list(self, raw: list[dict]) -> "GoldDataset":
        """Migrate from Python list format (rag/tests/retrieval/ style)."""
        self._samples = [
            GoldSample(
                corpus_id=self.corpus_id,
                query=item["query"],
                relevant_doc_sources=item.get("relevant_sources", []),
                difficulty=item.get("difficulty", "medium"),
                tags=item.get("tags", []),
            )
            for item in raw
        ]
        return self

    def save_to_file(self) -> None:
        """Write samples to evaluation/data/{corpus_id}.jsonl."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / f"{self.corpus_id}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for s in self._samples:
                f.write(s.model_dump_json() + "\n")
        logger.info("Saved %d gold samples to %s", len(self._samples), path)

    @property
    def samples(self) -> list[GoldSample]:
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)
