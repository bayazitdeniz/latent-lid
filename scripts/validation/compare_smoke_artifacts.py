#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch


DEFAULT_IGNORE_GLOBS = {
    "*.lock",
    "*.log",
}


################################################################################
# CLI and artifact discovery
################################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Compare two smoke-run artifact directories and report behavior drift."
    )
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute numeric tolerance. Defaults to exact comparison.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Relative numeric tolerance. Defaults to exact comparison.",
    )
    parser.add_argument(
        "--ignore-glob",
        action="append",
        default=[],
        help="Relative file glob to ignore. Can be passed multiple times.",
    )
    parser.add_argument(
        "--ignore-json-key",
        action="append",
        default=["exp_id"],
        help="JSON object key to ignore recursively. Defaults to exp_id.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for the machine-readable comparison summary.",
    )
    return parser.parse_args()


def should_ignore(path: Path, patterns: set[str]) -> bool:
    rel = path.as_posix()
    return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)


def artifact_files(root: Path, ignore_patterns: set[str]) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    files = {}
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root)
            if not should_ignore(rel, ignore_patterns):
                files[rel.as_posix()] = path
    return files


################################################################################
# Structured-value comparison
################################################################################

def strip_json_keys(value: Any, ignored_keys: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_json_keys(item, ignored_keys)
            for key, item in value.items()
            if key not in ignored_keys
        }
    if isinstance(value, list):
        return [strip_json_keys(item, ignored_keys) for item in value]
    return value


def compare_numbers(left: float, right: float, *, atol: float, rtol: float) -> tuple[bool, float]:
    if math.isnan(left) and math.isnan(right):
        return True, 0.0
    diff = abs(left - right)
    return diff <= atol + rtol * abs(right), diff


def compare_values(
    left: Any,
    right: Any,
    *,
    path: str,
    atol: float,
    rtol: float,
    diffs: list[dict[str, Any]],
) -> float:
    max_diff = 0.0
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            diffs.append({"path": path, "kind": "type", "left": type(left).__name__, "right": type(right).__name__})
            return max_diff
        if tuple(left.shape) != tuple(right.shape):
            diffs.append({"path": path, "kind": "shape", "left": list(left.shape), "right": list(right.shape)})
            return max_diff
        left_cpu = left.detach().cpu()
        right_cpu = right.detach().cpu()
        if left_cpu.dtype != right_cpu.dtype:
            diffs.append(
                {
                    "path": path,
                    "kind": "tensor_dtype",
                    "left": str(left_cpu.dtype),
                    "right": str(right_cpu.dtype),
                }
            )
            return max_diff
        if torch.is_floating_point(left_cpu) or torch.is_floating_point(right_cpu):
            close = torch.isclose(left_cpu, right_cpu, atol=atol, rtol=rtol, equal_nan=True)
            diff = torch.nan_to_num((left_cpu - right_cpu).abs(), nan=0.0).max().item()
            max_diff = max(max_diff, float(diff))
            if not bool(close.all()):
                diffs.append({"path": path, "kind": "tensor_values", "max_abs_diff": float(diff)})
        elif not torch.equal(left_cpu, right_cpu):
            diffs.append({"path": path, "kind": "tensor_values"})
        return max_diff

    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            diffs.append({"path": path, "kind": "type", "left": type(left).__name__, "right": type(right).__name__})
            return max_diff
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys - right_keys):
            diffs.append({"path": f"{path}.{key}", "kind": "missing_in_candidate"})
        for key in sorted(right_keys - left_keys):
            diffs.append({"path": f"{path}.{key}", "kind": "extra_in_candidate"})
        for key in sorted(left_keys & right_keys):
            max_diff = max(
                max_diff,
                compare_values(left[key], right[key], path=f"{path}.{key}", atol=atol, rtol=rtol, diffs=diffs),
            )
        return max_diff

    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            diffs.append({"path": path, "kind": "type", "left": type(left).__name__, "right": type(right).__name__})
            return max_diff
        if len(left) != len(right):
            diffs.append({"path": path, "kind": "length", "left": len(left), "right": len(right)})
            return max_diff
        for idx, (left_item, right_item) in enumerate(zip(left, right)):
            max_diff = max(
                max_diff,
                compare_values(left_item, right_item, path=f"{path}[{idx}]", atol=atol, rtol=rtol, diffs=diffs),
            )
        return max_diff

    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            diffs.append({"path": path, "kind": "type", "left": type(left).__name__, "right": type(right).__name__})
            return max_diff
        ok, diff = compare_numbers(float(left), float(right), atol=atol, rtol=rtol)
        if not ok:
            diffs.append({"path": path, "kind": "number", "left": left, "right": right, "abs_diff": diff})
        return max(max_diff, diff)

    if left != right:
        diffs.append({"path": path, "kind": "value", "left": left, "right": right})
    return max_diff


################################################################################
# Tabular and file comparison
################################################################################

def compare_dataframes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    rel_path: str,
    atol: float,
    rtol: float,
    diffs: list[dict[str, Any]],
) -> float:
    max_diff = 0.0
    if list(left.columns) != list(right.columns):
        diffs.append({"path": rel_path, "kind": "columns", "left": list(left.columns), "right": list(right.columns)})
        return max_diff
    if len(left) != len(right):
        diffs.append({"path": rel_path, "kind": "row_count", "left": len(left), "right": len(right)})
        return max_diff
    for column in left.columns:
        left_col = left[column]
        right_col = right[column]
        if pd.api.types.is_bool_dtype(left_col) or pd.api.types.is_bool_dtype(right_col):
            equal = left_col.fillna("<NA>").astype(str).equals(right_col.fillna("<NA>").astype(str))
            if not equal:
                diffs.append({"path": f"{rel_path}:{column}", "kind": "column_values"})
        elif pd.api.types.is_numeric_dtype(left_col) and pd.api.types.is_numeric_dtype(right_col):
            left_num = pd.to_numeric(left_col)
            right_num = pd.to_numeric(right_col)
            diff = (left_num - right_num).abs().max(skipna=True)
            observed_diff = 0.0 if pd.isna(diff) else float(diff)
            max_diff = max(max_diff, observed_diff)
            equal = (left_num - right_num).abs() <= atol + rtol * right_num.abs()
            equal = equal | (left_num.isna() & right_num.isna())
            if not bool(equal.all()):
                diffs.append(
                    {
                        "path": f"{rel_path}:{column}",
                        "kind": "column_values",
                        "max_abs_diff": observed_diff,
                    }
                )
        else:
            equal = left_col.fillna("<NA>").astype(str).equals(right_col.fillna("<NA>").astype(str))
            if not equal:
                diffs.append({"path": f"{rel_path}:{column}", "kind": "column_values"})
    return max_diff


def load_jsonl(path: Path) -> list[Any]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compare_file(
    rel_path: str,
    left_path: Path,
    right_path: Path,
    *,
    ignored_json_keys: set[str],
    atol: float,
    rtol: float,
    diffs: list[dict[str, Any]],
) -> float:
    suffix = left_path.suffix
    if suffix != right_path.suffix:
        diffs.append({"path": rel_path, "kind": "suffix", "left": suffix, "right": right_path.suffix})
        return 0.0
    if suffix == ".json":
        left = strip_json_keys(json.loads(left_path.read_text(encoding="utf-8")), ignored_json_keys)
        right = strip_json_keys(json.loads(right_path.read_text(encoding="utf-8")), ignored_json_keys)
        return compare_values(left, right, path=rel_path, atol=atol, rtol=rtol, diffs=diffs)
    if suffix == ".jsonl":
        left = strip_json_keys(load_jsonl(left_path), ignored_json_keys)
        right = strip_json_keys(load_jsonl(right_path), ignored_json_keys)
        return compare_values(left, right, path=rel_path, atol=atol, rtol=rtol, diffs=diffs)
    if suffix == ".pt":
        left = strip_json_keys(torch.load(left_path, map_location="cpu", weights_only=True), ignored_json_keys)
        right = strip_json_keys(torch.load(right_path, map_location="cpu", weights_only=True), ignored_json_keys)
        return compare_values(left, right, path=rel_path, atol=atol, rtol=rtol, diffs=diffs)
    if suffix == ".parquet":
        return compare_dataframes(
            pd.read_parquet(left_path),
            pd.read_parquet(right_path),
            rel_path=rel_path,
            atol=atol,
            rtol=rtol,
            diffs=diffs,
        )
    if suffix == ".csv":
        return compare_dataframes(
            pd.read_csv(left_path),
            pd.read_csv(right_path),
            rel_path=rel_path,
            atol=atol,
            rtol=rtol,
            diffs=diffs,
        )
    if left_path.read_bytes() != right_path.read_bytes():
        diffs.append({"path": rel_path, "kind": "bytes"})
    return 0.0


################################################################################
# Entrypoint
################################################################################

def main() -> None:
    args = parse_args()
    ignore_patterns = set(DEFAULT_IGNORE_GLOBS) | set(args.ignore_glob)
    ignored_json_keys = set(args.ignore_json_key)
    baseline_files = artifact_files(args.baseline_dir, ignore_patterns)
    candidate_files = artifact_files(args.candidate_dir, ignore_patterns)
    diffs: list[dict[str, Any]] = []
    max_abs_diff = 0.0

    for rel_path in sorted(set(baseline_files) - set(candidate_files)):
        diffs.append({"path": rel_path, "kind": "missing_in_candidate"})
    for rel_path in sorted(set(candidate_files) - set(baseline_files)):
        diffs.append({"path": rel_path, "kind": "extra_in_candidate"})
    for rel_path in sorted(set(baseline_files) & set(candidate_files)):
        max_abs_diff = max(
            max_abs_diff,
            compare_file(
                rel_path,
                baseline_files[rel_path],
                candidate_files[rel_path],
                ignored_json_keys=ignored_json_keys,
                atol=float(args.atol),
                rtol=float(args.rtol),
                diffs=diffs,
            ),
        )

    summary = {
        "status": "PASS" if not diffs else "FAIL",
        "baseline_dir": str(args.baseline_dir),
        "candidate_dir": str(args.candidate_dir),
        "n_baseline_files": len(baseline_files),
        "n_candidate_files": len(candidate_files),
        "n_diffs": len(diffs),
        "max_abs_diff": max_abs_diff,
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "diffs": diffs[:200],
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if diffs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
