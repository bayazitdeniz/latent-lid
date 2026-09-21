#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from wendler_extend_utils import PROVIDER_DIR, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or resume chunked DeepL translation requests.")
    parser.add_argument("--requests-dir", type=Path, default=PROVIDER_DIR / "deepl_requests")
    parser.add_argument("--responses-dir", type=Path, default=PROVIDER_DIR / "deepl_responses")
    parser.add_argument("--auth-key", default=os.environ.get("DEEPL_AUTH_KEY", ""))
    parser.add_argument("--api-url", default=os.environ.get("DEEPL_API_URL", "https://api-free.deepl.com/v2/translate"))
    parser.add_argument(
        "--skip-usage-check",
        action="store_true",
        help="Do not query DeepL /v2/usage before running requests.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--estimate-usage", action="store_true")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Maximum retries for transient DeepL errors such as HTTP 429.",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=5.0,
        help="Initial retry delay in seconds for transient DeepL errors.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.2,
        help="Delay in seconds between DeepL calls.",
    )
    return parser.parse_args()


def retry_after_seconds(exc: HTTPError) -> float | None:
    value = exc.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def deepl_request(
    url: str,
    auth_key: str,
    data: bytes | None = None,
    *,
    max_retries: int = 8,
    retry_base_delay: float = 5.0,
) -> dict[str, object]:
    request = Request(
        url,
        data=data,
        headers={
            "Authorization": f"DeepL-Auth-Key {auth_key}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST" if data is not None else "GET",
    )
    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                delay = retry_after_seconds(exc)
                if delay is None:
                    delay = min(retry_base_delay * (2**attempt), 120.0)
                print(
                    f"DeepL HTTP {exc.code}; retrying in {delay:.1f}s "
                    f"({attempt + 1}/{max_retries})."
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"DeepL HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            if attempt < max_retries:
                delay = min(retry_base_delay * (2**attempt), 120.0)
                print(
                    f"DeepL request failed: {exc}; retrying in {delay:.1f}s "
                    f"({attempt + 1}/{max_retries})."
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"DeepL request failed: {exc}") from exc
    raise RuntimeError("DeepL request retry loop exhausted unexpectedly")


def usage_url(api_url: str) -> str:
    return api_url.rsplit("/", 1)[0] + "/usage"


def get_usage(api_url: str, auth_key: str, max_retries: int, retry_base_delay: float) -> dict[str, object]:
    return deepl_request(
        usage_url(api_url),
        auth_key,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )


def format_int(value: object) -> str:
    return f"{int(value):,}"


def print_usage_summary(usage: dict[str, object], pending_chars: int) -> None:
    character_count = int(usage.get("character_count", 0))
    character_limit = int(usage.get("character_limit", 0))
    remaining = character_limit - character_count if character_limit else 0
    projected_remaining = remaining - pending_chars if character_limit else 0
    print(
        "DeepL usage: "
        f"{format_int(character_count)} / {format_int(character_limit)} characters used; "
        f"{format_int(max(remaining, 0))} remaining."
    )
    print(
        "This run: "
        f"{format_int(pending_chars)} pending characters; "
        f"projected remaining after run: {format_int(projected_remaining)}."
    )
    if projected_remaining < 0:
        raise RuntimeError(
            "Pending requests exceed the reported DeepL character limit by "
            f"{format_int(abs(projected_remaining))} characters."
        )


def call_deepl_texts(
    api_url: str,
    auth_key: str,
    *,
    source_lang: str,
    target_lang: str,
    texts: list[str],
    context: str = "",
    max_retries: int = 8,
    retry_base_delay: float = 5.0,
) -> dict[str, object]:
    fields: list[tuple[str, str]] = [
        ("source_lang", source_lang),
        ("target_lang", target_lang),
    ]
    fields.extend(("text", text) for text in texts)
    if context:
        fields.append(("context", context))
    data = urlencode(fields).encode("utf-8")
    return deepl_request(
        api_url,
        auth_key,
        data,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )


def call_deepl(
    api_url: str,
    auth_key: str,
    payload: dict[str, object],
    *,
    max_retries: int = 8,
    retry_base_delay: float = 5.0,
) -> dict[str, object]:
    return call_deepl_texts(
        api_url,
        auth_key,
        source_lang=str(payload["source_lang"]),
        target_lang=str(payload["target_lang"]),
        texts=[str(text) for text in payload["texts"]],
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )


def call_deepl_context_chunk(
    api_url: str,
    auth_key: str,
    payload: dict[str, object],
    *,
    max_retries: int = 8,
    retry_base_delay: float = 5.0,
    request_delay: float = 0.2,
) -> dict[str, object]:
    items = payload.get("items", [])
    if not isinstance(items, list) or not items:
        return call_deepl(
            api_url,
            auth_key,
            payload,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
        )

    source_lang = str(payload["source_lang"])
    target_lang = str(payload["target_lang"])
    translations_by_index: dict[int, dict[str, object]] = {}
    cloze_indices: list[int] = []
    cloze_texts: list[str] = []

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise RuntimeError(f"Malformed request item at index {idx}")
        text = str(item.get("text", ""))
        context = str(item.get("context", ""))
        if context:
            response = call_deepl_texts(
                api_url,
                auth_key,
                source_lang=source_lang,
                target_lang=target_lang,
                texts=[text],
                context=context,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
            )
            if request_delay:
                time.sleep(request_delay)
            translations = response.get("translations", [])
            if not isinstance(translations, list) or len(translations) != 1:
                raise RuntimeError(
                    f"DeepL returned {len(translations) if isinstance(translations, list) else 'invalid'} "
                    f"translations for context item {idx}"
                )
            translations_by_index[idx] = translations[0]
        else:
            cloze_indices.append(idx)
            cloze_texts.append(text)

    if cloze_texts:
        response = call_deepl_texts(
            api_url,
            auth_key,
            source_lang=source_lang,
            target_lang=target_lang,
            texts=cloze_texts,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
        )
        if request_delay:
            time.sleep(request_delay)
        translations = response.get("translations", [])
        if not isinstance(translations, list) or len(translations) != len(cloze_texts):
            raise RuntimeError(
                f"DeepL returned {len(translations) if isinstance(translations, list) else 'invalid'} "
                f"translations for {len(cloze_texts)} cloze items"
            )
        for idx, translation in zip(cloze_indices, translations):
            translations_by_index[idx] = translation

    return {"translations": [translations_by_index[idx] for idx in range(len(items))]}


def request_paths(requests_dir: Path) -> list[Path]:
    return [path for path in sorted(requests_dir.glob("*.json")) if path.name != "manifest.json"]


def response_matches_request(response_path: Path, payload: dict[str, object]) -> bool:
    if not response_path.exists():
        return False
    try:
        response_data = read_json(response_path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        response_data.get("chunk_id") == payload.get("chunk_id")
        and response_data.get("request_version", "") == payload.get("request_version", "")
    )


def main() -> None:
    args = parse_args()
    paths = request_paths(args.requests_dir)
    request_payloads = {path: read_json(path) for path in paths}
    pending_paths = []
    for path, payload in request_payloads.items():
        if args.force or not response_matches_request(args.responses_dir / path.name, payload):
            pending_paths.append(path)
    pending_chars = sum(int(read_json(path).get("character_count", 0)) for path in pending_paths)
    if args.dry_run or args.estimate_usage:
        existing = len(paths) - len(pending_paths)
        print(
            f"Request chunks: {len(paths)}; reusable responses: {existing}; "
            f"pending chunks: {len(pending_paths)}; pending characters: {pending_chars}"
        )
        if args.auth_key and not args.skip_usage_check:
            print_usage_summary(
                get_usage(args.api_url, args.auth_key, args.max_retries, args.retry_base_delay),
                pending_chars,
            )
        return
    if not args.auth_key:
        raise ValueError("DeepL auth key is required via --auth-key or DEEPL_AUTH_KEY")
    if not args.skip_usage_check:
        print_usage_summary(
            get_usage(args.api_url, args.auth_key, args.max_retries, args.retry_base_delay),
            pending_chars,
        )

    args.responses_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    for path in paths:
        out_path = args.responses_dir / path.name
        payload = request_payloads[path]
        if not args.force and response_matches_request(out_path, payload):
            skipped += 1
            continue
        response = call_deepl_context_chunk(
            args.api_url,
            args.auth_key,
            payload,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
            request_delay=args.request_delay,
        )
        write_json(
            out_path,
            {
                "chunk_id": payload["chunk_id"],
                "request": payload,
                "response": response,
                "provider": "deepl",
                "request_version": payload.get("request_version", ""),
            },
        )
        completed += 1
        print(f"Wrote {out_path}")
    print(f"Completed {completed} chunks; skipped {skipped}.")


if __name__ == "__main__":
    main()
