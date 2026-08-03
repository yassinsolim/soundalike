"""Build the deterministic autocomplete catalog from a row-aligned index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "api"))
from _search import write_search_catalog  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "api" / "search_catalog.json.gz",
    )
    args = parser.parse_args()
    result = write_search_catalog(args.index, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
