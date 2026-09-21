#!/usr/bin/env python3
"""Build the downloadable runtime and build-time data archives.

Every archive member is rooted at ``data/``. Users should therefore extract an
archive from the repository root rather than from inside the existing data
directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("dist")
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
COPY_CHUNK_BYTES = 8 * 1024 * 1024
ALREADY_COMPRESSED_SUFFIXES = {".bz2", ".gz", ".parquet", ".tgz", ".xz", ".zip", ".zst"}
IGNORED_DIRECTORY_NAMES = {".git", "__pycache__"}
IGNORED_FILE_NAMES = {".DS_Store"}

WENDLER_UPSTREAM = {
    "name": "epfl-dlab/llm-latent-language",
    "url": "https://github.com/epfl-dlab/llm-latent-language.git",
    "commit": "d0e4f292f204d69bf9bc51cbe35e73e4f66e896f",
    "included_path": "data/wendler_2024_data/llm-latent-language",
    "source_files_included": True,
    "git_metadata_included": False,
}


@dataclass(frozen=True)
class ArchiveSpec:
    kind: str
    filename: str
    description: str
    files: tuple[str, ...]
    directories: tuple[str, ...]
    excluded: tuple[str, ...]


RUNTIME_SPEC = ArchiveSpec(
    kind="runtime",
    filename="llid-data-runtime.zip",
    description="Data consumed by LLID fitting and evaluation.",
    files=(
        "data/pud_holdout/pud_prompts_train.jsonl",
        "data/pud_holdout/pud_prompts_test.jsonl",
        "data/pud_holdout/pud_split_meta.json",
        "data/ud_holdout/ud_prompts_train.jsonl",
        "data/ud_holdout/ud_prompts_test.jsonl",
        "data/ud_holdout/ud_split_meta.json",
        "data/fineweb_lens_27lang_55m/dataset_dict.json",
        "data/fineweb_lens_27lang_55m/stats.json",
        "data/include_10lang_3domain_cap30/prompts_question_only.jsonl",
        "data/include_10lang_3domain_cap30/prompts_minimal_mcq.jsonl",
        "data/include_10lang_3domain_cap30/stats.json",
        "data/include_10lang_3domain_all/prompts_question_only.jsonl",
        "data/include_10lang_3domain_all/prompts_minimal_mcq.jsonl",
        "data/include_10lang_3domain_all/stats.json",
    ),
    directories=(
        "data/pud_holdout/pud_langs_train",
        "data/ud_holdout/ud_langs_train",
        "data/fineweb_lens_27lang_55m/train",
        "data/fineweb_lens_27lang_55m/validation",
        "data/fineweb_lens_27lang_55m/test",
        "data/wendler_2024_data/common69",
    ),
    excluded=(
        "data/fineweb_lens_27lang_55m/window_cache/",
        "data/fineweb_lens_27lang_55m_lang_cache/",
        "data/pud_holdout/tokenized_pud_prompts_*.jsonl",
    ),
)


BUILDTIME_SPEC = ArchiveSpec(
    kind="buildtime",
    filename="llid-data-buildtime.zip",
    description="Optional source, intermediate, and provenance data used to build the runtime datasets.",
    files=(
        "data/include_10lang_3domain_cap30/metadata.jsonl",
        "data/include_10lang_3domain_cap30/metadata.parquet",
        "data/include_10lang_3domain_all/metadata.jsonl",
        "data/include_10lang_3domain_all/metadata.parquet",
    ),
    directories=(
        "data/ud_cache",
        "data/ud_holdout/rejected",
        "data/wendler_2024_data/processed",
        "data/wendler_2024_data/extended",
        "data/wendler_2024_data/merged",
        "data/wendler_2024_data/llm-latent-language",
    ),
    excluded=(
        "data/fineweb_lens_27lang_55m/window_cache/",
        "data/fineweb_lens_27lang_55m_lang_cache/",
        "data/wendler_2024_data/llm-latent-language/.git/",
    ),
)

ARCHIVE_SPECS = {spec.kind: spec for spec in (RUNTIME_SPEC, BUILDTIME_SPEC)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        choices=["runtime", "buildtime", "all"],
        default="all",
        help="Archive to build. Defaults to both archives.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory relative to the repository root. Defaults to dist/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing archive with the same name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print archive contents without writing ZIP files.",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="Print every selected file in addition to the summary.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Read every compressed member after building and fail on the first CRC error.",
    )
    return parser.parse_args()


def _display_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _relative_file(repo_root: Path, relative_path: str) -> Path:
    path = repo_root / relative_path
    if path.is_symlink():
        raise ValueError(f"Archive inputs may not be symbolic links: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Required archive input is missing: {relative_path}")
    return path


def _files_under(repo_root: Path, relative_dir: str) -> list[Path]:
    root = repo_root / relative_dir
    if root.is_symlink():
        raise ValueError(f"Archive inputs may not be symbolic links: {relative_dir}")
    if not root.is_dir():
        raise FileNotFoundError(f"Required archive directory is missing: {relative_dir}")

    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(repo_root)
        if any(part in IGNORED_DIRECTORY_NAMES for part in relative.parts):
            continue
        if path.name in IGNORED_FILE_NAMES or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"Archive inputs may not be symbolic links: {relative.as_posix()}")
        if path.is_file():
            files.append(path)
    return files


def collect_archive_files(repo_root: Path, spec: ArchiveSpec) -> list[Path]:
    """Resolve and validate the files selected by an archive specification."""
    selected = {_relative_file(repo_root, path) for path in spec.files}
    for directory in spec.directories:
        selected.update(_files_under(repo_root, directory))
    return sorted(selected, key=lambda path: path.relative_to(repo_root).as_posix())


def validate_archive_provenance(repo_root: Path, spec: ArchiveSpec) -> None:
    """Ensure included upstream source files match the revision in the manifest."""
    if spec.kind != "buildtime":
        return
    upstream_relative = Path(WENDLER_UPSTREAM["included_path"])
    upstream_root = repo_root / upstream_relative
    try:
        commit = subprocess.run(
            ["git", "-C", upstream_relative.as_posix(), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", upstream_relative.as_posix(), "status", "--short", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to inspect the Wendler upstream checkout at {upstream_root}") from exc
    if commit != WENDLER_UPSTREAM["commit"]:
        raise RuntimeError(
            "Wendler upstream checkout does not match the manifest: "
            f"expected {WENDLER_UPSTREAM['commit']}, found {commit}"
        )
    if status:
        raise RuntimeError(
            "Wendler upstream checkout has uncommitted files; commit or remove them before archiving:\n"
            f"{status}"
        )


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def runtime_dataset_summary(repo_root: Path) -> dict:
    fineweb = _read_json(repo_root / "data/fineweb_lens_27lang_55m/stats.json")
    pud = _read_json(repo_root / "data/pud_holdout/pud_split_meta.json")
    ud = _read_json(repo_root / "data/ud_holdout/ud_split_meta.json")
    include_cap = _read_json(repo_root / "data/include_10lang_3domain_cap30/stats.json")
    include_all = _read_json(repo_root / "data/include_10lang_3domain_all/stats.json")
    wendler = _read_json(repo_root / "data/wendler_2024_data/common69/summary.json")

    fineweb_splits = {
        name: {
            "rows": details.get("examples"),
            "reference_tokens": details.get("tokens"),
        }
        for name, details in fineweb.get("final", {}).items()
    }
    return {
        "fineweb_27lang": {
            "languages": len(fineweb.get("langs", [])),
            "sequence_length": fineweb.get("seq_len"),
            "splits": fineweb_splits,
        },
        "pud": {
            "languages": len(pud.get("langs", [])),
            "train_prompts": _jsonl_rows(repo_root / "data/pud_holdout/pud_prompts_train.jsonl"),
            "test_prompts": _jsonl_rows(repo_root / "data/pud_holdout/pud_prompts_test.jsonl"),
        },
        "ud_extension": {
            "languages": len(ud.get("langs", [])),
            "train_prompts": _jsonl_rows(repo_root / "data/ud_holdout/ud_prompts_train.jsonl"),
            "test_prompts": _jsonl_rows(repo_root / "data/ud_holdout/ud_prompts_test.jsonl"),
        },
        "include_cap30": {
            "languages": len(include_cap.get("langs", [])),
            "prompts": include_cap.get("n_selected_rows"),
        },
        "include_all": {
            "languages": len(include_all.get("langs", [])),
            "prompts": include_all.get("n_selected_rows"),
        },
        "wendler_common69": {
            "languages": wendler.get("language_count"),
            "concepts": wendler.get("common_concept_count"),
            "rows": wendler.get("rows"),
            "tokenizer_specific_models": len(wendler.get("start_token_artifacts", {})),
        },
    }


def _zip_info(archive_path: str, compression: int) -> ZipInfo:
    info = ZipInfo(archive_path, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = (0o100644 & 0xFFFF) << 16
    return info


def _compression_for(path: Path) -> int:
    return ZIP_STORED if path.suffix.lower() in ALREADY_COMPRESSED_SUFFIXES else ZIP_DEFLATED


def _write_file(zip_file: ZipFile, source: Path, archive_path: str) -> tuple[int, str]:
    source_stat = source.stat()
    digest = hashlib.sha256()
    info = _zip_info(archive_path, _compression_for(source))
    with source.open("rb") as input_file, zip_file.open(info, mode="w", force_zip64=True) as output_file:
        while chunk := input_file.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
            output_file.write(chunk)
    final_stat = source.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise RuntimeError(f"Archive input changed while it was being read: {source}")
    return source_stat.st_size, digest.hexdigest()


def _manifest_path(spec: ArchiveSpec) -> str:
    return f"data/{spec.filename.removesuffix('.zip')}-manifest.json"


def _manifest(spec: ArchiveSpec, file_records: Sequence[dict], repo_root: Path) -> dict:
    manifest = {
        "schema_version": 1,
        "archive": spec.filename,
        "archive_kind": spec.kind,
        "description": spec.description,
        "extract_at": "repository root",
        "archive_paths_begin_with": "data/",
        "created_by": "scripts/data/build_data_archives.py",
        "file_count": len(file_records),
        "uncompressed_bytes": sum(int(record["bytes"]) for record in file_records),
        "excluded": list(spec.excluded),
        "files": list(file_records),
        "redistribution_notice": (
            "This archive contains material derived from multiple upstream datasets. "
            "Consult the repository README, citations, and upstream license terms before redistribution."
        ),
    }
    if spec.kind == "runtime":
        manifest["datasets"] = runtime_dataset_summary(repo_root)
    elif spec.kind == "buildtime":
        manifest["upstream_code"] = [WENDLER_UPSTREAM]
    return manifest


def build_archive(
    repo_root: Path,
    output_dir: Path,
    spec: ArchiveSpec,
    *,
    force: bool,
    verify: bool,
) -> Path:
    validate_archive_provenance(repo_root, spec)
    files = collect_archive_files(repo_root, spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / spec.filename
    partial_path = output_dir / f".{spec.filename}.partial"

    if output_path.exists() and not force:
        raise FileExistsError(f"Archive already exists (use --force to replace it): {output_path}")
    if partial_path.exists():
        partial_path.unlink()

    total_bytes = sum(path.stat().st_size for path in files)
    print(f"Building {spec.filename}: {len(files)} files, {_display_bytes(total_bytes)} uncompressed")
    records: list[dict] = []
    try:
        with ZipFile(partial_path, mode="w", compression=ZIP_DEFLATED, allowZip64=True) as zip_file:
            for index, source in enumerate(files, start=1):
                archive_path = source.relative_to(repo_root).as_posix()
                size = source.stat().st_size
                print(f"  [{index}/{len(files)}] {_display_bytes(size):>10}  {archive_path}")
                written_bytes, sha256 = _write_file(zip_file, source, archive_path)
                records.append({"path": archive_path, "bytes": written_bytes, "sha256": sha256})

            manifest_bytes = (
                json.dumps(_manifest(spec, records, repo_root), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            zip_file.writestr(_zip_info(_manifest_path(spec), ZIP_DEFLATED), manifest_bytes)

        if output_path.exists():
            output_path.unlink()
        os.replace(partial_path, output_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise

    if verify:
        verify_archive(output_path, spec)

    print(f"Wrote {output_path} ({_display_bytes(output_path.stat().st_size)})")
    return output_path


def verify_archive(archive_path: Path, spec: ArchiveSpec) -> None:
    """Verify archive membership, CRCs, and manifest SHA-256 checksums."""
    print(f"Verifying {archive_path} ...")
    with ZipFile(archive_path, mode="r") as zip_file:
        manifest_name = _manifest_path(spec)
        manifest = json.loads(zip_file.read(manifest_name))
        records = manifest.get("files")
        if not isinstance(records, list):
            raise RuntimeError(f"Malformed file list in {manifest_name}")
        expected_names = {manifest_name, *(str(record["path"]) for record in records)}
        actual_names = set(zip_file.namelist())
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            unexpected = sorted(actual_names - expected_names)
            raise RuntimeError(f"Archive membership mismatch: missing={missing}, unexpected={unexpected}")

        for record in records:
            member = str(record["path"])
            digest = hashlib.sha256()
            size = 0
            with zip_file.open(member, mode="r") as input_file:
                while chunk := input_file.read(COPY_CHUNK_BYTES):
                    digest.update(chunk)
                    size += len(chunk)
            if size != int(record["bytes"]):
                raise RuntimeError(f"Size verification failed for {member}: expected {record['bytes']}, found {size}")
            if digest.hexdigest() != str(record["sha256"]):
                raise RuntimeError(f"SHA-256 verification failed for {member}")


def _selected_specs(selection: str) -> Iterable[ArchiveSpec]:
    if selection == "all":
        return (RUNTIME_SPEC, BUILDTIME_SPEC)
    return (ARCHIVE_SPECS[selection],)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir

    for spec in _selected_specs(args.archive):
        if args.dry_run:
            validate_archive_provenance(REPO_ROOT, spec)
            files = collect_archive_files(REPO_ROOT, spec)
            total_bytes = sum(path.stat().st_size for path in files)
            print(f"{spec.filename}: {len(files)} files, {_display_bytes(total_bytes)} uncompressed")
            print("  excluded:")
            for excluded in spec.excluded:
                print(f"    - {excluded}")
            if args.list_files:
                print("  included:")
                for path in files:
                    print(f"    - {path.relative_to(REPO_ROOT).as_posix()}")
            continue
        build_archive(REPO_ROOT, output_dir, spec, force=args.force, verify=args.verify)


if __name__ == "__main__":
    main()
