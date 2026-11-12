"""Run person-orientation checks against a local JPG dataset."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __package__:
    from .run_face_orientation_dataset import (
        PERSON_PROFILE,
        run_production as base_run_production,
    )
else:
    from run_face_orientation_dataset import (
        PERSON_PROFILE,
        run_production as base_run_production,
    )


async def run_person_profile(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    client_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    return await base_run_production(
        input_dir,
        output_dir,
        client_factory=client_factory,
        profile=PERSON_PROFILE,
    )


run_production = run_person_profile


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="datasets/test", help="directory containing JPGs")
    parser.add_argument(
        "--output",
        default="exports/person_orientation_eval",
        help="output directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    asyncio.run(run_production(args.input, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
