"""Download and read the Universal Dependencies sources used by data builders."""

from __future__ import annotations

import socket
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm


TREEBANK_MAP = {
    "ar": "Arabic-PUD",
    "cs": "Czech-PUD",
    "de": "German-PUD",
    "en": "English-PUD",
    "fi": "Finnish-PUD",
    "fr": "French-PUD",
    "gl": "Galician-PUD",
    "hi": "Hindi-PUD",
    "is": "Icelandic-PUD",
    "id": "Indonesian-PUD",
    "it": "Italian-PUD",
    "ja": "Japanese-PUD",
    "ko": "Korean-PUD",
    "pl": "Polish-PUD",
    "pt": "Portuguese-PUD",
    "ru": "Russian-PUD",
    "es": "Spanish-PUD",
    "sv": "Swedish-PUD",
    "th": "Thai-PUD",
    "tr": "Turkish-PUD",
    "zh": "Chinese-PUD",
}

PUD_LANGS = tuple(TREEBANK_MAP)

UD_EXTRA_TREEBANKS = {
    "uk": {"treebank": "Ukrainian-IU", "file_prefix": "uk_iu"},
    "bg": {"treebank": "Bulgarian-BTB", "file_prefix": "bg_btb"},
    "sr": {"treebank": "Serbian-SET", "file_prefix": "sr_set"},
    "ur": {"treebank": "Urdu-UDTB", "file_prefix": "ur_udtb"},
    "fa": {"treebank": "Persian-Seraji", "file_prefix": "fa_seraji"},
    "mr": {"treebank": "Marathi-UFAL", "file_prefix": "mr_ufal"},
}

UD_EXTRA_LANGS = tuple(UD_EXTRA_TREEBANKS)


################################################################################
# Download cache
################################################################################

def default_ud_cache_path(ud_revision: str) -> Path:
    return Path("data/ud_cache") / f"ud-treebanks-v{ud_revision}.tgz"


def download_with_progress(url: str, dest: Path, timeout: int) -> None:
    socket.setdefaulttimeout(timeout)
    dest.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=0, unit="B", unit_scale=True, desc=f"Downloading {dest.name}")

    def _hook(count: int, block: int, total: int) -> None:
        if total > 0 and pbar.total == 0:
            pbar.total = total
        downloaded = count * block
        pbar.update(downloaded - pbar.n)

    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp_dest, reporthook=_hook)
        tmp_dest.replace(dest)
    finally:
        pbar.close()
        if tmp_dest.exists():
            tmp_dest.unlink(missing_ok=True)


def ensure_ud_tarball(
    *,
    ud_revision: str,
    ud_url: str,
    download_timeout: int,
    cache_path: str | Path | None = None,
) -> Path:
    tar_path = Path(cache_path) if cache_path is not None else default_ud_cache_path(ud_revision)
    if tar_path.exists():
        print(f"Using cached UD tarball: {tar_path}")
        return tar_path
    download_with_progress(ud_url, tar_path, download_timeout)
    print(f"Cached UD tarball: {tar_path}")
    return tar_path


################################################################################
# CoNLL-U records
################################################################################

def read_conllu_sentence_records(fp) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    text: str | None = None
    surface_tokens: list[str] = []
    active_mwt_range: tuple[int, int] | None = None
    for raw in fp:
        line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        line = line.rstrip("\n")
        if line.startswith("# text = "):
            text = line[len("# text = ") :].strip()
            surface_tokens = []
            active_mwt_range = None
            continue
        if not line:
            if text is not None:
                records.append({"text": text, "surface_tokens": list(surface_tokens)})
            text = None
            surface_tokens = []
            active_mwt_range = None
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        tok_id = cols[0]
        form = cols[1]
        if "-" in tok_id:
            start_str, end_str = tok_id.split("-", 1)
            active_mwt_range = (int(start_str), int(end_str))
            surface_tokens.append(form)
            continue
        if "." in tok_id:
            continue
        int_id = int(tok_id)
        if active_mwt_range is not None:
            start_id, end_id = active_mwt_range
            if start_id <= int_id <= end_id:
                if int_id == end_id:
                    active_mwt_range = None
                continue
            active_mwt_range = None
        surface_tokens.append(form)
    if text is not None:
        records.append({"text": text, "surface_tokens": list(surface_tokens)})
    if not records:
        raise ValueError("No sentence records found in conllu.")
    return records


def count_conllu_texts(fp) -> int:
    count = 0
    for raw in fp:
        line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        if line.startswith("# text = "):
            count += 1
    return count


################################################################################
# UD archive lookup
################################################################################

def _ud_member_suffix(treebank: str, file_prefix: str, split: str) -> str:
    return f"UD_{treebank}/{file_prefix}-ud-{split}.conllu"


def _find_ud_member(tar: tarfile.TarFile, treebank: str, file_prefix: str, split: str) -> tarfile.TarInfo:
    target_suffix = _ud_member_suffix(treebank, file_prefix, split)
    member = next((m for m in tar.getmembers() if m.name.endswith(target_suffix)), None)
    if member is None:
        raise FileNotFoundError(f"Missing conllu: {target_suffix}")
    return member


def sentence_records_from_ud_treebank(
    tar_path: Path,
    *,
    treebank: str,
    file_prefix: str,
    splits: list[str] | tuple[str, ...],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for split in splits:
            member = _find_ud_member(tar, treebank, file_prefix, split)
            with tar.extractfile(member) as fp:
                if fp is None:
                    raise FileNotFoundError(f"Failed to extract {member.name}")
                for record in read_conllu_sentence_records(fp):
                    record["ud_split"] = split
                    records.append(record)
    if not records:
        raise ValueError(f"No sentence records found for UD_{treebank}.")
    return records


def sentence_records_from_ud_tar(lang: str, tar_path: Path) -> list[dict[str, object]]:
    if lang not in TREEBANK_MAP:
        raise ValueError(f"No UD treebank mapping for lang '{lang}'.")
    treebank = TREEBANK_MAP[lang]
    with tarfile.open(tar_path, "r:gz") as tar:
        target_suffix = f"UD_{treebank}/{lang}_pud-ud-test.conllu"
        member = next((m for m in tar.getmembers() if m.name.endswith(target_suffix)), None)
        if member is None:
            raise FileNotFoundError(f"Missing conllu for {lang}: {target_suffix}")
        with tar.extractfile(member) as fp:
            if fp is None:
                raise FileNotFoundError(f"Failed to extract {member.name}")
            return read_conllu_sentence_records(fp)


def count_from_ud_tar(lang: str, tar_path: Path) -> int:
    if lang not in TREEBANK_MAP:
        raise ValueError(f"No UD treebank mapping for lang '{lang}'.")
    treebank = TREEBANK_MAP[lang]
    with tarfile.open(tar_path, "r:gz") as tar:
        target_suffix = f"UD_{treebank}/{lang}_pud-ud-test.conllu"
        member = next((m for m in tar.getmembers() if m.name.endswith(target_suffix)), None)
        if member is None:
            raise FileNotFoundError(f"Missing conllu for {lang}: {target_suffix}")
        with tar.extractfile(member) as fp:
            if fp is None:
                raise FileNotFoundError(f"Failed to extract {member.name}")
            return count_conllu_texts(fp)
