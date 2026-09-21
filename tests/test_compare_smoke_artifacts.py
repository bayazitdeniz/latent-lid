import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch


SCRIPT_PATH = Path("scripts/validation/compare_smoke_artifacts.py")
SPEC = importlib.util.spec_from_file_location("compare_smoke_artifacts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
compare_smoke_artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_smoke_artifacts)


class CompareSmokeArtifactsTests(unittest.TestCase):
    def test_cli_defaults_to_exact_numeric_comparison(self) -> None:
        with mock.patch.object(sys, "argv", [str(SCRIPT_PATH), "baseline", "candidate"]):
            args = compare_smoke_artifacts.parse_args()

        self.assertEqual(args.atol, 0.0)
        self.assertEqual(args.rtol, 0.0)

    def test_exact_tensor_comparison_rejects_formerly_tolerated_difference(self) -> None:
        diffs = []

        max_diff = compare_smoke_artifacts.compare_values(
            torch.tensor([1.0]),
            torch.tensor([1.0 + 5e-6]),
            path="values.pt",
            atol=0.0,
            rtol=0.0,
            diffs=diffs,
        )

        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["kind"], "tensor_values")
        self.assertGreater(max_diff, 0.0)

    def test_exact_tensor_comparison_rejects_dtype_change(self) -> None:
        diffs = []

        max_diff = compare_smoke_artifacts.compare_values(
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float64),
            path="values.pt",
            atol=0.0,
            rtol=0.0,
            diffs=diffs,
        )

        self.assertEqual(max_diff, 0.0)
        self.assertEqual(diffs, [
            {
                "path": "values.pt",
                "kind": "tensor_dtype",
                "left": "torch.float32",
                "right": "torch.float64",
            }
        ])

    def test_max_diff_includes_tolerated_tensor_difference(self) -> None:
        diffs = []

        max_diff = compare_smoke_artifacts.compare_values(
            torch.tensor([1.0]),
            torch.tensor([1.0 + 5e-6]),
            path="values.pt",
            atol=1e-5,
            rtol=0.0,
            diffs=diffs,
        )

        self.assertEqual(diffs, [])
        self.assertGreater(max_diff, 0.0)

    def test_max_diff_includes_tolerated_dataframe_difference(self) -> None:
        diffs = []

        max_diff = compare_smoke_artifacts.compare_dataframes(
            pd.DataFrame({"value": [1.0]}),
            pd.DataFrame({"value": [1.0 + 5e-6]}),
            rel_path="metrics.parquet",
            atol=1e-5,
            rtol=0.0,
            diffs=diffs,
        )

        self.assertEqual(diffs, [])
        self.assertGreater(max_diff, 0.0)


if __name__ == "__main__":
    unittest.main()
