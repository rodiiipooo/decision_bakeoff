"""Smoke tests: schema load, decider score shapes, harness exit path."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decision_bakeoff.deciders import get_architecture  # noqa: E402
from decision_bakeoff.deciders.members import (  # noqa: E402
    CosineDecider,
    MockCLMDecider,
    RouterBaseline,
)
from decision_bakeoff.metrics import compute_all  # noqa: E402
from decision_bakeoff.schema import Example, load_data_dir  # noqa: E402


class TestSchema(unittest.TestCase):
    def test_load_samples(self):
        examples = load_data_dir(ROOT / "data" / "samples")
        self.assertGreaterEqual(len(examples), 100)
        types = {e.problem_type for e in examples}
        self.assertEqual(
            types,
            {
                "route",
                "retrieve_rerank",
                "bon_verify",
                "triage",
                "guided_questionnaire",
            },
        )
        gq = [e for e in examples if e.problem_type == "guided_questionnaire"]
        self.assertTrue(any(e.questions for e in gq))


class TestDeciders(unittest.TestCase):
    def setUp(self):
        self.ex = Example.from_dict(
            {
                "id": "t1",
                "problem_type": "route",
                "state": {"query": "billing refund please"},
                "candidates": [
                    {"id": "billing", "text": "department:billing"},
                    {"id": "support", "text": "department:support"},
                    {"id": "sales", "text": "department:sales"},
                ],
                "gold": "billing",
                "baseline_choice": "billing",
                "meta": {},
                "split": "shadow",
            }
        )

    def test_members_score_keys(self):
        for cls in (RouterBaseline, CosineDecider, MockCLMDecider):
            m = cls()
            if isinstance(m, RouterBaseline):
                m.bind_example(self.ex)
            probs = m.score(self.ex.state, self.ex.candidates)
            self.assertEqual(set(probs), {"billing", "support", "sales"})
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)

    def test_architectures_fit_score(self):
        calib = [self.ex]
        for arch_id in ("A0", "A1", "A2", "A3", "A4", "A5"):
            arch = get_architecture(arch_id)
            arch.fit(calib)
            arch.bind_example(self.ex)
            probs = arch.score(self.ex.state, self.ex.candidates)
            self.assertEqual(set(probs), {"billing", "support", "sales"})
            s = sum(probs.values())
            self.assertGreater(s, 0.0)


class TestMetrics(unittest.TestCase):
    def test_compute_all(self):
        preds = [{"a": 0.7, "b": 0.3}, {"a": 0.2, "b": 0.8}]
        golds = ["a", "b"]
        cids = [["a", "b"], ["a", "b"]]
        m = compute_all(preds, golds, cids)
        self.assertEqual(m["top1"], 1.0)
        self.assertIn("ndcg@3", m)
        self.assertIn("ece_stub", m)


class TestCLI(unittest.TestCase):
    def test_run_exits_zero(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "decision_bakeoff",
                "run",
                "--data",
                str(ROOT / "data" / "samples"),
                "--arch",
                "A0,A1",
                "--results",
                str(ROOT / "results"),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
        # Latest results dir should have metrics.json
        results = sorted((ROOT / "results").glob("*/metrics.json"))
        self.assertTrue(results)
        data = json.loads(results[-1].read_text())
        self.assertEqual(len(data["results"]), 2)


if __name__ == "__main__":
    unittest.main()
