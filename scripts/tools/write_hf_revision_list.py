#!/usr/bin/env python3
"""Write Hugging Face checkpoint revision lists ordered by training progress."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

from huggingface_hub import list_repo_refs


STEP_RE = re.compile(r"(?:^|[^0-9])step[-_\s]*([0-9]+)", re.IGNORECASE)
GLOBAL_STEP_RE = re.compile(r"(?:^|[^0-9])global_step[-_\s]*([0-9]+)", re.IGNORECASE)
OLMO2_STAGE_RE = re.compile(
    r"^stage(?P<stage>[0-9]+)(?:-ingredient(?P<ingredient>[0-9]+))?.*?step[-_\s]*(?P<step>[0-9]+)",
    re.IGNORECASE,
)


################################################################################
# Training-progress ordering
################################################################################

def token_count_key(name: str) -> tuple[int, float, str]:
    match = re.search(r"tokens?([0-9]+(?:\.[0-9]+)?)([kKmMbBtT]?)", name)
    if not match:
        return (1, float("inf"), name)
    value = float(match.group(1))
    unit = match.group(2).lower()
    multiplier = {"": 1.0, "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}[unit]
    return (0, value * multiplier, name)


def step_key(name: str) -> tuple[int, int, str]:
    match = STEP_RE.search(name)
    if not match:
        return (1, 0, name)
    return (0, int(match.group(1)), name)


def global_step_key(name: str) -> tuple[int, int, str]:
    match = GLOBAL_STEP_RE.search(name)
    if not match:
        return (1, 0, name)
    return (0, int(match.group(1)), name)


def olmo2_stage_key(name: str) -> tuple[int, int, int, str]:
    match = OLMO2_STAGE_RE.search(name)
    if not match:
        return (10**9, 10**9, 10**9, name)
    stage = int(match.group("stage"))
    ingredient = int(match.group("ingredient") or 0)
    step = int(match.group("step"))
    return (stage, ingredient, step, name)


def infer_sort_strategy(names: Iterable[str]) -> str:
    names = list(names)
    if any(OLMO2_STAGE_RE.search(name) for name in names):
        return "olmo2"
    if any(GLOBAL_STEP_RE.search(name) for name in names):
        return "global_step"
    return "token_step"


def sort_revisions(names: Iterable[str], move_main_to_end: bool, strategy: str = "auto") -> list[str]:
    names = list(names)
    if strategy == "auto":
        strategy = infer_sort_strategy(names)
    regular = [name for name in names if name != "main" or not move_main_to_end]
    if strategy == "olmo2":
        ordered = sorted(regular, key=olmo2_stage_key)
    elif strategy == "global_step":
        ordered = sorted(regular, key=global_step_key)
    elif strategy == "token_step":
        ordered = sorted(regular, key=lambda name: (token_count_key(name), step_key(name), name))
    else:
        raise ValueError(f"Unknown sort strategy: {strategy}")
    if move_main_to_end and "main" in names:
        ordered.append("main")
    return ordered


################################################################################
# Entrypoint
################################################################################

def main() -> None:
    parser = argparse.ArgumentParser("Write HF branch/revision names to a checkpoint list.")
    parser.add_argument("--model", required=True, help="HF model id.")
    parser.add_argument("--out", required=True, help="Output checkpoint list path.")
    parser.add_argument("--include-main", action="store_true", default=True)
    parser.add_argument("--exclude-main", action="store_false", dest="include_main")
    parser.add_argument("--move-main-to-end", action="store_true", default=True)
    parser.add_argument("--no-move-main-to-end", action="store_false", dest="move_main_to_end")
    parser.add_argument("--exclude-prefix", action="append", default=[])
    parser.add_argument("--include-prefix", action="append", default=[])
    parser.add_argument("--ref-kind", choices=["branches", "tags"], default="branches")
    parser.add_argument("--sort-strategy", choices=["auto", "token_step", "olmo2", "global_step"], default="auto")
    args = parser.parse_args()

    refs = list_repo_refs(args.model)
    if args.ref_kind == "branches":
        revisions = [branch.name for branch in getattr(refs, "branches", [])]
    else:
        revisions = [tag.name for tag in getattr(refs, "tags", [])]
    if not args.include_main:
        revisions = [name for name in revisions if name != "main"]
    for prefix in args.exclude_prefix:
        revisions = [name for name in revisions if not name.startswith(prefix)]
    if args.include_prefix:
        revisions = [
            name
            for name in revisions
            if any(name.startswith(prefix) for prefix in args.include_prefix)
            or (args.include_main and name == "main")
        ]

    ordered = sort_revisions(
        revisions,
        move_main_to_end=bool(args.move_main_to_end),
        strategy=args.sort_strategy,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    n_token_named = sum(1 for name in ordered if token_count_key(name)[0] == 0)
    strategy = args.sort_strategy if args.sort_strategy != "auto" else infer_sort_strategy(ordered)
    print(f"Wrote {len(ordered)} {args.ref_kind} to {out_path} ({n_token_named} token-named, sort={strategy})")


if __name__ == "__main__":
    main()
