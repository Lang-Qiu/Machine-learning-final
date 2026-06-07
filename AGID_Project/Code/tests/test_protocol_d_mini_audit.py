from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "protocol_d_mini_audit.py"
    spec = importlib.util.spec_from_file_location("protocol_d_mini_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ProtocolDMiniAuditTests(unittest.TestCase):
    def test_closed_form_zero_out_reports_channel_reliance(self):
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
        weights = np.array([4.0, 0.0, 0.0], dtype=np.float32)
        names = ["jpeg_quant", "freq_radial", "texture_geometry"]

        rows = mod.closed_form_zero_out(
            concepts,
            labels,
            weights,
            -2.0,
            names,
            variant="unit",
            generator="G",
        )

        jpeg = [r for r in rows if r["channel"] == "jpeg_quant"][0]
        self.assertEqual(jpeg["baseline_acc"], 1.0)
        self.assertEqual(jpeg["acc"], 0.5)
        self.assertEqual(jpeg["delta_pp"], -50.0)
        self.assertEqual(jpeg["drop_pp"], 50.0)

    def test_summary_tracks_compression_axis_share(self):
        mod = load_module()
        rows = [
            {"variant": "full", "channel": "jpeg_quant", "drop_pp": 40.0},
            {"variant": "full", "channel": "freq_radial", "drop_pp": 10.0},
            {"variant": "full", "channel": "hf_noise", "drop_pp": 5.0},
            {"variant": "no_pair", "channel": "jpeg_quant", "drop_pp": 4.0},
            {"variant": "no_pair", "channel": "freq_radial", "drop_pp": 1.0},
        ]

        summary = mod.summarize_reliance(rows)

        self.assertEqual(summary["full"]["top_channel"], "jpeg_quant")
        self.assertEqual(summary["full"]["compression_drop_pp"], 50.0)
        self.assertEqual(summary["full"]["compression_share"], round(50.0 / 55.0, 6))
        self.assertEqual(summary["no_pair"]["compression_share"], 1.0)


if __name__ == "__main__":
    unittest.main()
