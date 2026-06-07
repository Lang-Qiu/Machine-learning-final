from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "protocol_c_matrix.py"
    spec = importlib.util.spec_from_file_location("protocol_c_matrix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ProtocolCMatrixTests(unittest.TestCase):
    def test_subset_interventions_report_drop_and_keep_deltas(self):
        mod = load_module()
        concepts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 1, 1])
        gens = np.array(["G", "G", "G", "G"])
        names = ["jpeg_quant", "freq_radial", "texture_geometry"]
        weights = np.array([4.0, 0.0, 0.0], dtype=np.float32)
        bias = -2.0

        rows = mod.compute_subset_interventions(
            concepts,
            labels,
            gens,
            weights,
            bias,
            names,
            seed="unit",
            source="unit_dump",
            subset_sizes=(1, 2),
        )

        target = [
            r
            for r in rows
            if r["family"] == "drop"
            and r["channels"] == "jpeg_quant"
            and r["generator"] == "G"
        ][0]
        self.assertEqual(target["baseline_acc"], 1.0)
        self.assertEqual(target["acc"], 0.5)
        self.assertEqual(target["delta_pp"], -50.0)

        pair = [
            r
            for r in rows
            if r["family"] == "keep_only"
            and r["channels"] == "jpeg_quant+freq_radial"
            and r["generator"] == "G"
        ][0]
        self.assertEqual(pair["subset_size"], 2)
        self.assertEqual(pair["acc"], 1.0)
        self.assertEqual(pair["delta_pp"], 0.0)


    def test_reliance_summary_computes_compression_share(self):
        mod = load_module()
        rows = [
            {"seed": "s1", "family": "drop", "subset_size": 1, "channels": "jpeg_quant", "delta_pp": -20.0},
            {"seed": "s1", "family": "drop", "subset_size": 1, "channels": "freq_radial", "delta_pp": -10.0},
            {"seed": "s1", "family": "drop", "subset_size": 1, "channels": "hf_noise", "delta_pp": -5.0},
        ]

        summary = mod.compute_reliance_summary(rows)

        self.assertEqual(summary["n_rows"], 3)
        self.assertEqual(summary["single_drop_rows"], 3)
        self.assertEqual(summary["compression_share_single_drop"], round(30.0 / 35.0, 6))
        self.assertEqual(summary["reliance_concentration_single_drop"], round(20.0 / 35.0, 6))


if __name__ == "__main__":
    unittest.main()
