#!/usr/bin/env python3
"""Select token-aligned training checkpoints for cross-model comparisons."""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RevisionPoint:
    name: str
    tokens: float


OLMO2_STAGE2_RE = re.compile(r"^stage2-ingredient[0-9]+-", re.IGNORECASE)


################################################################################
# Revision metadata
################################################################################

def model_safe_name(model_name: str) -> str:
    return str(model_name).strip("/").replace("/", "_")


def parse_token_count(revision: str) -> float | None:
    match = re.search(r"tokens?([0-9]+(?:\.[0-9]+)?)([kKmMbBtT]?)", revision)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    multiplier = {"": 1.0, "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}[unit]
    return value * multiplier


def parse_metadata_tokens(value: str) -> float:
    cleaned = str(value).strip().replace("_", "").replace(",", "")
    tokens = parse_token_count(f"tokens{cleaned}")
    if tokens is None:
        raise ValueError(f"Could not parse token count from metadata value {value!r}")
    return tokens


def load_token_metadata(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    sample = "\n".join(line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#"))
    if not sample:
        raise ValueError(f"Token metadata file is empty: {path}")

    first_line = sample.splitlines()[0]
    delimiter = "\t" if "\t" in first_line else ","
    rows = list(csv.reader(sample.splitlines(), delimiter=delimiter))
    header = [cell.strip().lower() for cell in rows[0]]
    revision_keys = {"revision", "checkpoint", "branch", "name"}
    token_keys = {"tokens", "token_count", "train_tokens", "training_tokens", "seen_tokens"}

    metadata: dict[str, float] = {}
    if revision_keys.intersection(header) and token_keys.intersection(header):
        rev_idx = next(idx for idx, key in enumerate(header) if key in revision_keys)
        tok_idx = next(idx for idx, key in enumerate(header) if key in token_keys)
        data_rows = rows[1:]
    else:
        rev_idx = 0
        tok_idx = 1
        data_rows = rows

    for row in data_rows:
        if len(row) <= max(rev_idx, tok_idx):
            continue
        revision = row[rev_idx].strip()
        if not revision or revision.startswith("#"):
            continue
        metadata[revision] = parse_metadata_tokens(row[tok_idx])
    if not metadata:
        raise ValueError(f"No revision/token rows found in metadata file: {path}")
    return metadata


def load_points(
    path: Path,
    *,
    token_metadata: Mapping[str, float] | None = None,
    strict_token_counts: bool = False,
    cumulative_olmo2_stage2: bool = True,
) -> list[RevisionPoint]:
    revisions = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    stage1_tokens = [
        tokens
        for revision in revisions
        if revision.startswith("stage1-")
        for tokens in [parse_token_count(revision)]
        if tokens is not None
    ]
    olmo2_stage1_budget = max(stage1_tokens) if stage1_tokens else None

    points: list[RevisionPoint] = []
    skipped: list[str] = []
    for revision in revisions:
        tokens = None
        if token_metadata is not None:
            tokens = token_metadata.get(revision)
        if tokens is None:
            tokens = parse_token_count(revision)
            if (
                tokens is not None
                and cumulative_olmo2_stage2
                and olmo2_stage1_budget is not None
                and OLMO2_STAGE2_RE.search(revision)
            ):
                tokens += olmo2_stage1_budget
        if tokens is None:
            skipped.append(revision)
            continue
        points.append(RevisionPoint(name=revision, tokens=tokens))
    if skipped and strict_token_counts:
        examples = ", ".join(skipped[:5])
        raise ValueError(
            f"{path} has {len(skipped)} revisions without token counts. "
            f"Examples: {examples}. Provide --token-metadata or disable --strict-token-counts."
        )
    points.sort(key=lambda point: (point.tokens, point.name))
    if not points:
        raise ValueError(f"No token-count revisions found in {path}. Provide --token-metadata for this model.")
    if skipped:
        print(f"Skipped {len(skipped)} revisions without token counts in {path}")
    deduped: list[RevisionPoint] = []
    seen = set()
    for point in points:
        if point.name in seen:
            continue
        deduped.append(point)
        seen.add(point.name)
    return deduped


################################################################################
# Aligned checkpoint selection
################################################################################

def linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 0:
        raise ValueError("--count must be positive.")
    if count == 1:
        return [stop]
    step = (stop - start) / float(count - 1)
    return [start + step * idx for idx in range(count)]


def nearest_unique(points: list[RevisionPoint], targets: list[float]) -> list[RevisionPoint]:
    selected: list[RevisionPoint] = []
    used_names = set()
    for target in targets:
        candidates = [point for point in points if point.name not in used_names]
        if not candidates:
            raise ValueError("Not enough unique revisions to satisfy requested token targets.")
        choice = min(candidates, key=lambda point: (abs(point.tokens - target), point.tokens))
        selected.append(choice)
        used_names.add(choice.name)
    selected.sort(key=lambda point: point.tokens)
    return selected


################################################################################
# Entrypoint
################################################################################

def main() -> None:
    parser = argparse.ArgumentParser("Sample per-model revision lists on a shared token-count grid.")
    parser.add_argument(
        "--model",
        nargs=2,
        action="append",
        metavar=("MODEL_NAME", "REVISION_FILE"),
        required=True,
        help="Model name and its full revision list. Repeat once per model.",
    )
    parser.add_argument(
        "--token-metadata",
        nargs=2,
        action="append",
        metavar=("MODEL_NAME", "TSV_OR_CSV"),
        default=[],
        help=(
            "Optional per-model revision/token metadata. File may have columns "
            "revision,tokens or be two-column revision<tab>tokens."
        ),
    )
    parser.add_argument("--strict-token-counts", action="store_true")
    parser.add_argument(
        "--cumulative-olmo2-stage2",
        action="store_true",
        default=True,
        help="Treat OLMo2 stage-2 branch token counts as additional tokens after the max stage-1 checkpoint.",
    )
    parser.add_argument(
        "--no-cumulative-olmo2-stage2",
        action="store_false",
        dest="cumulative_olmo2_stage2",
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    metadata_by_model = {
        model_name: load_token_metadata(Path(metadata_file))
        for model_name, metadata_file in args.token_metadata
    }
    model_points = {
        model_name: load_points(
            Path(revision_file),
            token_metadata=metadata_by_model.get(model_name),
            strict_token_counts=bool(args.strict_token_counts),
            cumulative_olmo2_stage2=bool(args.cumulative_olmo2_stage2),
        )
        for model_name, revision_file in args.model
    }
    unknown_metadata_models = sorted(set(metadata_by_model) - set(model_points))
    if unknown_metadata_models:
        raise ValueError(f"--token-metadata was provided for unknown models: {unknown_metadata_models}")
    overlap_start = max(points[0].tokens for points in model_points.values())
    overlap_stop = min(points[-1].tokens for points in model_points.values())
    if overlap_start > overlap_stop:
        ranges = {
            model: (points[0].tokens, points[-1].tokens)
            for model, points in model_points.items()
        }
        raise ValueError(f"Model token ranges do not overlap: {ranges}")

    targets = linspace(overlap_start, overlap_stop, int(args.count))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_by_model = {
        model_name: nearest_unique(points, targets)
        for model_name, points in model_points.items()
    }
    for model_name, selected in selected_by_model.items():
        out_path = out_dir / f"checkpoints_{model_safe_name(model_name)}.txt"
        out_path.write_text("\n".join(point.name for point in selected) + "\n", encoding="utf-8")
        print(f"Wrote {len(selected)} revisions for {model_name}: {out_path}")

    with (out_dir / "alignment.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        header = ["target_tokens"]
        for model_name in selected_by_model:
            header.extend([f"{model_safe_name(model_name)}_revision", f"{model_safe_name(model_name)}_tokens"])
        writer.writerow(header)
        for idx, target in enumerate(targets):
            row = [f"{target:.0f}"]
            for selected in selected_by_model.values():
                point = selected[idx]
                row.extend([point.name, f"{point.tokens:.0f}"])
            writer.writerow(row)
    print(f"Wrote alignment table: {out_dir / 'alignment.tsv'}")


if __name__ == "__main__":
    main()
