import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from scripts.data.build_data_archives import (
    BUILDTIME_SPEC,
    RUNTIME_SPEC,
    build_archive,
    collect_archive_files,
)


class BuildDataArchivesTest(unittest.TestCase):
    def _write(self, root: Path, relative: str, content: str = "test\n") -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _make_repo(self, root: Path) -> None:
        json_object = "{}\n"
        jsonl = '{"id": "one"}\n'
        runtime_files = {
            "data/pud_holdout/pud_prompts_train.jsonl": jsonl,
            "data/pud_holdout/pud_prompts_test.jsonl": jsonl,
            "data/pud_holdout/pud_split_meta.json": '{"langs": ["en"]}\n',
            "data/pud_holdout/pud_langs_train/en/train.csv": "text\nhello\n",
            "data/ud_holdout/ud_prompts_train.jsonl": jsonl,
            "data/ud_holdout/ud_prompts_test.jsonl": jsonl,
            "data/ud_holdout/ud_split_meta.json": '{"langs": ["uk"]}\n',
            "data/ud_holdout/ud_langs_train/uk/train.csv": "text\nhello\n",
            "data/fineweb_lens_27lang_55m/dataset_dict.json": json_object,
            "data/fineweb_lens_27lang_55m/stats.json": (
                '{"langs": ["en"], "seq_len": 2048, '
                '"final": {"train": {"examples": 1, "tokens": 2048}}}\n'
            ),
            "data/fineweb_lens_27lang_55m/train/data.arrow": "arrow",
            "data/fineweb_lens_27lang_55m/validation/data.arrow": "arrow",
            "data/fineweb_lens_27lang_55m/test/data.arrow": "arrow",
            "data/include_10lang_3domain_cap30/prompts_question_only.jsonl": jsonl,
            "data/include_10lang_3domain_cap30/prompts_minimal_mcq.jsonl": jsonl,
            "data/include_10lang_3domain_cap30/stats.json": '{"langs": ["en"], "n_selected_rows": 1}\n',
            "data/include_10lang_3domain_all/prompts_question_only.jsonl": jsonl,
            "data/include_10lang_3domain_all/prompts_minimal_mcq.jsonl": jsonl,
            "data/include_10lang_3domain_all/stats.json": '{"langs": ["en"], "n_selected_rows": 1}\n',
            "data/wendler_2024_data/common69/summary.json": (
                '{"language_count": 1, "common_concept_count": 1, "rows": {"copy": 1}, '
                '"start_token_artifacts": {"model": []}}\n'
            ),
            "data/wendler_2024_data/common69/copy.csv": "prompt\nhello\n",
        }
        buildtime_files = {
            "data/include_10lang_3domain_cap30/metadata.jsonl": jsonl,
            "data/include_10lang_3domain_cap30/metadata.parquet": "parquet",
            "data/include_10lang_3domain_all/metadata.jsonl": jsonl,
            "data/include_10lang_3domain_all/metadata.parquet": "parquet",
            "data/ud_cache/ud-treebanks-v2.17.tgz": "tgz",
            "data/ud_holdout/rejected/uk.jsonl": jsonl,
            "data/wendler_2024_data/processed/source.csv": "source",
            "data/wendler_2024_data/extended/source.csv": "source",
            "data/wendler_2024_data/merged/source.csv": "source",
            "data/wendler_2024_data/llm-latent-language/README.md": "upstream",
            "data/wendler_2024_data/llm-latent-language/.git/config": "must not be archived",
        }
        excluded_files = {
            "data/fineweb_lens_27lang_55m/window_cache/model/cache.arrow": "cache",
            "data/fineweb_lens_27lang_55m_lang_cache/train/en/cache.arrow": "cache",
        }
        for relative, content in {**runtime_files, **buildtime_files, **excluded_files}.items():
            self._write(root, relative, content)

    def test_archive_selections_are_disjoint_and_exclude_caches(self):
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            root = Path(tmp)
            self._make_repo(root)
            runtime = {path.relative_to(root).as_posix() for path in collect_archive_files(root, RUNTIME_SPEC)}
            buildtime = {path.relative_to(root).as_posix() for path in collect_archive_files(root, BUILDTIME_SPEC)}

            self.assertFalse(runtime & buildtime)
            self.assertIn("data/wendler_2024_data/common69/copy.csv", runtime)
            self.assertIn("data/ud_cache/ud-treebanks-v2.17.tgz", buildtime)
            self.assertNotIn("data/fineweb_lens_27lang_55m/window_cache/model/cache.arrow", runtime)
            self.assertNotIn("data/fineweb_lens_27lang_55m_lang_cache/train/en/cache.arrow", buildtime)
            self.assertNotIn("data/wendler_2024_data/llm-latent-language/.git/config", buildtime)

    def test_runtime_archive_contains_data_root_and_manifest(self):
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            root = Path(tmp)
            self._make_repo(root)
            output = root / "dist"
            archive = build_archive(root, output, RUNTIME_SPEC, force=False, verify=True)

            with ZipFile(archive) as zip_file:
                names = set(zip_file.namelist())
                self.assertTrue(all(name.startswith("data/") for name in names))
                manifest_name = "data/llid-data-runtime-manifest.json"
                self.assertIn(manifest_name, names)
                manifest = json.loads(zip_file.read(manifest_name))

            self.assertEqual(manifest["archive_kind"], "runtime")
            self.assertEqual(manifest["extract_at"], "repository root")
            self.assertEqual(manifest["datasets"]["pud"]["train_prompts"], 1)
            self.assertEqual(len(manifest["files"]), manifest["file_count"])


if __name__ == "__main__":
    unittest.main()
