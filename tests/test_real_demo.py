"""Offline test doubles for real ESPN parser and chronological ML pipeline.

The tests mock provider payloads. The PUBLIC DEMO never displays simulated stats.
"""
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from demo.real_espn import (
    ATHLETES, SEASONS, parse_gamelog, espn_gamelog_url, write_real_sqlite,
)
from demo.real_models import (
    WARMUP, chronological_split, real_forecast,
    supervised_examples, train_state,
)


class RealESPNTests(unittest.TestCase):
    def test_espn_url_rejects_unknown_ids(self):
        url = espn_gamelog_url(ATHLETES["Nikola Jokic"], 2027)
        self.assertIn("/3112335/gamelog?season=2027", url)
        with self.assertRaises(ValueError):
            espn_gamelog_url("not-real", 2027)
        with self.assertRaises(ValueError):
            espn_gamelog_url(ATHLETES["Nikola Jokic"], 2001)

    def test_parse_real_provider_schema_with_stale_and_playoff_records(self):
        # Test double uses the shape of ESPN's actual 2026-27 gamelog feed.
        game_id = "401909316"
        payload = {
            "labels": [
                "MIN", "FG", "FG%", "3PT", "3P%", "FT", "FT%",
                "REB", "AST", "BLK", "STL", "PF", "TO", "PTS",
            ],
            "events": {
                game_id: {
                    "gameDate": "2026-11-25T03:00:00.000+00:00",
                    "team": {"abbreviation": "DEN"},
                    "opponent": {"displayName": "Dallas Mavericks"},
                    "atVs": "vs",
                },
            },
            "seasonTypes": [
                {
                    "displayName": "2026-27 Regular Season",
                    "categories": [{
                        "type": "event",
                        "events": [{
                            "eventId": game_id,
                            "stats": [
                                "33", "6-14", "42.9", "1-4", "25",
                                "5-5", "100", "11", "5", "0",
                                "1", "2", "5", "18",
                            ],
                        }],
                    }],
                },
                {
                    "displayName": "2026-27 Postseason",
                    "categories": [{
                        "type": "event",
                        "events": [{"eventId": game_id, "stats": ["0"] * 14}],
                    }],
                },
            ],
        }
        games = parse_gamelog(
            payload, ATHLETES["Nikola Jokic"], "Nikola Jokic", 2027
        )
        self.assertEqual(len(games), 1)
        self.assertEqual(games[0]["PTS"], 18.0)
        self.assertEqual(games[0]["REB"], 11.0)
        self.assertEqual(games[0]["PRA"], 34.0)
        self.assertEqual(games[0]["3PM"], 1.0)
        self.assertEqual(games[0]["date"], "2026-11-25")
        self.assertIn(game_id, games[0]["source_url"])

    def test_parser_rejects_missing_provider_columns(self):
        with self.assertRaises(ValueError):
            parse_gamelog({
                "labels": ["PTS"], "events": {}, "seasonTypes": [],
            }, ATHLETES["Nikola Jokic"], "Nikola Jokic", 2027)

    def test_sqlite_file_is_dedicated_and_not_original_database(self):
        game = {
            "athlete_id": "3112335", "event_id": "401909316",
            "player": "Nikola Jokic", "date": "2026-11-25",
            "season": "2026-27", "team": "DEN", "opponent": "Dallas Mavericks",
            "home": 1, "MIN": 33.0, "PTS": 18.0, "REB": 11.0,
            "AST": 5.0, "3PM": 1.0, "PRA": 34.0,
            "source_url": "https://www.espn.com/nba/boxscore/_/gameId/401909316",
        }
        with tempfile.TemporaryDirectory() as root:
            path = write_real_sqlite([game], root)
            self.assertEqual(path.name, "nba_public_real_espn.sqlite")
            self.assertTrue(path.is_file())
            self.assertFalse((Path(root) / "nba_database.db").exists())


class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Lightweight test doubles (never published as NBA game data).
        names = list(ATHLETES)[:6]
        cls.games = []
        start = date(2025, 10, 21)
        for idx in range(76):
            when = (start + timedelta(days=2 * idx)).isoformat()
            for j, name in enumerate(names):
                pts = 18 + j + (idx % 7)
                reb = 4 + j + (idx % 3)
                ast = 3 + j + (idx % 4)
                cls.games.append({
                    "player": name, "athlete_id": ATHLETES[name],
                    "event_id": str(401000000 + idx * 20 + j),
                    "date": when, "team": "DEN", "opponent": "Boston Celtics",
                    "home": idx % 2, "MIN": 28.0 + (idx % 6),
                    "PTS": float(pts), "REB": float(reb),
                    "AST": float(ast), "3PM": float(idx % 4),
                    "PRA": float(pts + reb + ast),
                })

    def test_future_target_never_enters_its_own_features(self):
        rows = supervised_examples(self.games, "PTS")
        first = next(r for r in rows if r["player"] == list(ATHLETES)[0])
        modified = [dict(x) for x in self.games]
        for game in modified:
            if (game["player"] == first["player"] and
                game["event_id"] == first["event_id"]):
                game["PTS"] = 9999.0
                game["PRA"] = 9999.0
        changed = supervised_examples(modified, "PTS")
        same = next(r for r in changed if
                    r["player"] == first["player"] and
                    r["event_id"] == first["event_id"])
        self.assertEqual(first["x"], same["x"])
        self.assertNotEqual(first["y"], same["y"])

    def test_global_time_split_and_model_forecast(self):
        rows = supervised_examples(self.games, "PTS")
        train, valid, test = chronological_split(rows)
        self.assertLess(max(r["date"] for r in train),
                        min(r["date"] for r in valid))
        self.assertLess(max(r["date"] for r in valid),
                        min(r["date"] for r in test))
        state = train_state({
            "games": self.games,
            "as_of_utc": "2026-11-27T00:00:00+00:00",
            "db_path": "/content/nba_public_real_espn.sqlite",
            "failures": [],
        }, advanced=False)
        for stat, model in state["models"].items():
            self.assertGreaterEqual(model["test_mae"], 0)
            self.assertGreaterEqual(model["baseline_test_mae"], 0)
            self.assertGreater(model["n_test"], 0)
        result = real_forecast(
            state, list(ATHLETES)[0], "PTS", 20, True, 2,
        )
        self.assertGreaterEqual(result["prediction"], 0)
        self.assertTrue(0 <= result["over_estimate"] <= 1)
        self.assertEqual(len(result["last10"]), 10)

    def test_invalid_statistic_and_threshold_are_rejected(self):
        state = train_state({"games": self.games}, advanced=False)
        with self.assertRaises(ValueError):
            real_forecast(state, list(ATHLETES)[0], "XYZ", 20, True, 2)
        with self.assertRaises(ValueError):
            real_forecast(state, list(ATHLETES)[0], "PTS", -1, True, 2)


if __name__ == "__main__":
    unittest.main()
