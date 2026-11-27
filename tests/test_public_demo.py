"""Offline checks for the public synthetic Colab demo: no optional pip dependencies."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from demo.public_demo import (
    FEATURES, METRICS, PROFILES, SEED, SmallRidge, build_examples,
    chronological_split, create_demo, forecast, make_synthetic_games,
)


class PublicDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.state = create_demo(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_synthetic_data_is_reproducible(self):
        self.assertEqual(
            make_synthetic_games(SEED), make_synthetic_games(SEED)
        )
        self.assertEqual(len(self.state["games"]), len(PROFILES) * 70)

    def test_database_is_separate_and_has_only_synthetic_table(self):
        filename = Path(self.state["database_path"])
        self.assertEqual(filename.name, "synthetic_nba_demo.sqlite")
        with sqlite3.connect(filename) as conn:
            names = [v[0] for v in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )]
            count = conn.execute(
                "SELECT count(*) FROM simulated_boxscores"
            ).fetchone()[0]
        self.assertEqual(names, ["simulated_boxscores"])
        self.assertEqual(count, len(self.state["games"]))

    def test_no_lookahead_and_chronological_splits(self):
        example = next(e for e in build_examples(
            self.state["games"], "PTS"
        ) if e["player"] == PROFILES[0][0] and e["game"] == 11)
        first_ten = [g for g in self.state["games"]
                     if g["player"] == PROFILES[0][0] and g["game"] <= 10]
        self.assertAlmostEqual(
            example["x"][0],
            sum(g["PTS"] for g in first_ten[-5:]) / 5,
        )
        self.assertEqual(len(example["x"]), len(FEATURES))
        for stat in METRICS:
            train, val, test = chronological_split(
                build_examples(self.state["games"], stat)
            )
            self.assertLess(max(r["game"] for r in train),
                            min(r["game"] for r in val))
            self.assertLess(max(r["game"] for r in val),
                            min(r["game"] for r in test))

    def test_model_accuracy_is_finite_and_residuals_are_separate(self):
        for stat in METRICS:
            model = self.state["models"][stat]
            self.assertGreater(model["test_mae"], 0)
            self.assertGreater(model["baseline_mae"], 0)
            self.assertEqual(len(model["validation_errors"]),
                             model["val_n"])
            self.assertEqual(model["train_n"], 336)
            self.assertEqual(model["val_n"], 72)
            self.assertEqual(model["test_n"], 72)

    def test_forecasts_are_bounded_and_probabilities_monotonic(self):
        results = [
            forecast(self.state["games"], self.state["models"],
                     PROFILES[0][0], "PTS", t)
            for t in (5, 20, 40)
        ]
        self.assertTrue(all(r["forecast"] >= 0 for r in results))
        self.assertEqual(
            [r["forecast"] for r in results],
            [results[0]["forecast"]] * 3,
        )
        p = [r["over_probability"] for r in results]
        self.assertGreaterEqual(p[0], p[1])
        self.assertGreaterEqual(p[1], p[2])
        self.assertTrue(all(0 <= value <= 1 for value in p))

    def test_forecast_rejects_unknown_or_invalid_inputs(self):
        with self.assertRaises(ValueError):
            forecast(self.state["games"], self.state["models"],
                     "Not a real player", "PTS", 20)
        with self.assertRaises(ValueError):
            forecast(self.state["games"], self.state["models"],
                     PROFILES[0][0], "PTS", 20, defense=4.0)
        with self.assertRaises(ValueError):
            build_examples(self.state["games"], "PRA")


if __name__ == "__main__":
    unittest.main()
