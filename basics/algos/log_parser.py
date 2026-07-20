"""Scan large, unstructured sensor log files for anomaly patterns in constant memory.

Lines are read one at a time via the file iterator (never loaded fully into
memory) and anomalies are yielded lazily, so both the input file and the
output stream can be arbitrarily large.
"""

import argparse
import gzip
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AnomalyRule:
    """A single detection rule applied to each log line.

    A rule works in one of two modes:

    1. Keyword mode (`min_value` and `max_value` both `None`): any regex match
       on the line counts as an anomaly, e.g. spotting the word "CRITICAL".
    2. Range mode (`min_value` and/or `max_value` set): the regex must have a
       capturing group whose text is a number (e.g. `temp=97.5` capturing
       `97.5`); the line is only an anomaly if that number falls *outside*
       the allowed `[min_value, max_value]` range. This is how sensor
       readings like out-of-bounds temperature or pressure are flagged.

    Instances are frozen/slotted since rules are created once as constants
    and never mutated, which also keeps per-instance memory overhead low.
    """

    name: str  # short identifier reported alongside each matching Anomaly
    pattern: re.Pattern[str]  # compiled once, in range mode group(1) must be numeric
    min_value: float | None = None  # inclusive lower bound; None disables the lower check
    max_value: float | None = None  # inclusive upper bound; None disables the upper check

    def is_anomaly(self, line: str) -> bool:
        """Evaluate this rule against a single line.

        Runs the compiled pattern against `line`. If there's no match, the
        line is not an anomaly for this rule. If the rule has no numeric
        bounds, a bare match is itself the anomaly signal. Otherwise the
        first capture group is parsed as a float and checked against
        `min_value`/`max_value`; the line is only flagged when that value
        falls outside the accepted range.
        """
        match = self.pattern.search(line)
        if match is None:
            return False
        if self.min_value is None and self.max_value is None:
            return True
        value = float(match.group(1))
        in_range = (self.min_value is None or value >= self.min_value) and (
            self.max_value is None or value <= self.max_value
        )
        return not in_range


@dataclass(frozen=True, slots=True)
class Anomaly:
    """One reported hit: which rule fired, on what line number, and the raw line text.

    Kept intentionally small (three primitive fields, `slots=True`) since
    `find_anomalies` may yield a large number of these from a huge log file;
    each instance should cost as little memory as possible.
    """

    rule_name: str  # AnomalyRule.name of the rule that matched
    line_number: int  # 1-based position of the offending line in the source file
    line: str  # the matching line, with trailing newline/carriage-return stripped


DEFAULT_RULES: tuple[AnomalyRule, ...] = (
    AnomalyRule("keyword_critical", re.compile(r"\b(?:CRITICAL|FATAL|FAULT)\b")),
    AnomalyRule(
        "temperature_out_of_range",
        re.compile(r"temp(?:erature)?[=:]\s*(-?\d+(?:\.\d+)?)"),
        min_value=-20.0,
        max_value=85.0,
    ),
    AnomalyRule(
        "pressure_out_of_range",
        re.compile(r"pressure[=:]\s*(-?\d+(?:\.\d+)?)"),
        min_value=0.0,
        max_value=150.0,
    ),
    AnomalyRule("server_error_code", re.compile(r"err(?:or)?_code[=:]\s*(5\d{2})")),
)


def _iter_lines(path: Path) -> Iterator[str]:
    """Open `path` and yield its lines one at a time, transparently decompressing `.gz` files.

    Picks `gzip.open` when the file extension is `.gz`, otherwise the builtin
    `open`; both are used in text mode (`"rt"`) so callers always get `str`
    lines regardless of compression. `errors="replace"` swaps any bytes that
    aren't valid UTF-8 for the U+FFFD replacement character instead of
    raising, so a single corrupted byte in a multi-gigabyte log can't abort
    the whole scan. Because this is a generator built on the file object's
    own iterator, Python only ever holds one line (plus its internal read
    buffer) in memory at a time, no matter how large the file is.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8", errors="replace") as f:
        yield from f


def find_anomalies(path: Path, rules: Iterable[AnomalyRule] = DEFAULT_RULES) -> Iterator[Anomaly]:
    """Scan `path` for anomalies, yielding each one as soon as it's found.

    For every line, strips its trailing line ending and checks it against
    every rule in `rules`, in order; a single line can match more than one
    rule, in which case it yields one `Anomaly` per matching rule. Because
    this function is itself a generator that delegates to the generator in
    `_iter_lines`, nothing about the scan is eagerly computed: no list of
    lines and no list of anomalies is ever materialized. A caller can do
    `for anomaly in find_anomalies(path): ...` and process a file far larger
    than available RAM, or stop early (as `main` does via `--limit`) without
    having paid the cost of scanning the rest of the file.
    """
    rules = tuple(rules)
    for line_number, raw_line in enumerate(_iter_lines(path), start=1):
        line = raw_line.rstrip("\r\n")
        for rule in rules:
            if rule.is_anomaly(line):
                yield Anomaly(rule.name, line_number, line)


def _parse_args() -> argparse.Namespace:
    """Build the CLI's argument parser and parse `sys.argv`.

    Exposes two arguments: the required `log_file` path (accepted as a
    `Path` so `find_anomalies` can use it directly), and an optional
    `--limit` that caps how many anomalies `main` reports before it stops
    reading the file early.
    """
    parser = argparse.ArgumentParser(description="Find anomaly patterns in large sensor log files.")
    parser.add_argument("log_file", type=Path, help="path to a .log or .log.gz sensor log file")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many anomalies")
    return parser.parse_args()


def main() -> None:
    """CLI entry point: scan the requested log file and report anomalies as they're found.

    Parses arguments, then iterates `find_anomalies` directly rather than
    collecting its results into a list first, printing each `Anomaly` as it
    arrives. `count` is tracked separately from list length precisely
    because no list is ever built. If `--limit` was given and that many
    anomalies have been printed, the loop breaks immediately, which — since
    `find_anomalies` is a generator — also stops it from reading any further
    into the file.
    """
    args = _parse_args()
    count = 0
    for anomaly in find_anomalies(args.log_file):
        print(f"[line {anomaly.line_number}] {anomaly.rule_name}: {anomaly.line}")
        count += 1
        if args.limit is not None and count >= args.limit:
            break
    print(f"Total anomalies: {count}")


if __name__ == "__main__":
    main()
