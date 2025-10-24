#!/usr/bin/env python3
"""Report Hugging Face model sizes for deployment planning."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.model_registry import list_models  # noqa: E402

_API_URL = "https://huggingface.co/api/models/{repo_id}"


@dataclass
class ModelSize:
    """Container describing an individual model payload size."""

    name: str
    repo_id: str
    bytes: int


def _fetch_model_size(repo_id: str, token: Optional[str]) -> int:
    request = Request(_API_URL.format(repo_id=repo_id))
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urlopen(request) as response:  # nosec B310 - trusted endpoint
        payload = json.loads(response.read().decode("utf-8"))
    siblings = payload.get("siblings", [])
    total = 0
    for sibling in siblings:
        size = sibling.get("size")
        if isinstance(size, int):
            total += size
    return total


def _format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def iter_model_sizes(token: Optional[str]) -> Iterable[ModelSize]:
    seen = set()
    for spec in list_models():
        if spec.name in seen:
            continue
        seen.add(spec.name)
        total_bytes = _fetch_model_size(spec.hf_repo, token)
        yield ModelSize(name=spec.name, repo_id=spec.hf_repo, bytes=total_bytes)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token",
        dest="token",
        default=None,
        help="Optional Hugging Face access token for private repositories.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    rows = list(iter_model_sizes(args.token))
    if not rows:
        print("No models available in the registry.")
        return 0
    max_name = max(len(row.name) for row in rows)
    max_repo = max(len(row.repo_id) for row in rows)
    total_bytes = sum(row.bytes for row in rows)
    header = f"{'Model':<{max_name}}  {'Repository':<{max_repo}}  Size"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.name:<{max_name}}  {row.repo_id:<{max_repo}}  {_format_size(row.bytes)}"
        )
    print("-" * len(header))
    print(f"Total{'':<{max_name + max_repo - 5}}  {_format_size(total_bytes)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
