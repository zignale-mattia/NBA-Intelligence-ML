"""Self-contained, reproducible NBA-style *synthetic* forecasting demonstration.

No live NBA data, credentials, Google Drive, or original trained models are used.
The numbers and player identities are invented for portfolio demonstration only.
The chronological validation and test sets are not used for model selection.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import math
import random
import sqlite3
import statistics

METRICS = ("PTS", "REB", "AST", "3PM")
PROFILES = (
    ("Demo Guard A", 27.0, 4.5, 8.0, 3.1, 36.0),
    ("Demo Guard B", 21.0, 5.0, 6.2, 2.5, 33.0),
    ("Demo Wing A", 24.0, 7.0, 4.0, 2.7, 35.0),
    ("Demo Wing B", 16.0, 6.3, 3.1, 1.8, 31.0),
    ("Demo Forward A", 22.0, 9.0, 4.4, 1.5, 34.0),
    ("Demo Forward B", 17.0, 8.2, 2.8, 1.2, 31.0),
    ("Demo Center A", 25.0, 12.0, 5.0, 0.7, 35.0),
    ("Demo Center B", 15.0, 11.0, 2.7, 0.3, 30.0),
)
FEATURES = ("recent_5", "recent_10", "minutes_5", "opp_defense",
            "home", "rest_days", "trend")
SEED = 20261127


def make_synthetic_games(seed: int = SEED, games_per_player: int = 70) -> list[dict]:
    """Generate fabricated basketball box scores with fixed random seed."""
    if games_per_player < 40:
        raise ValueError("Need at least 40 games per player")
    rng = random.Random(seed)
    games = []
    for player, pts, reb, ast, threes, mins in PROFILES:
        for game in range(games_per_player):
            defense = round(rng.uniform(-1.0, 1.0), 3)
            home = int(rng.random() > 0.5)
            rest = rng.choice((1, 2, 2, 3, 4))
            trend = 1.5 * math.sin(game / 9.0)
            minutes = max(19.0, min(42.0, rng.gauss(mins, 2.6) + (rest - 2) * 0.2))
            exposure = minutes / mins
            score = max(0, round(rng.gauss(
                pts * exposure - 1.7 * defense + 0.7 * home + trend,
                4.6,
            )))
            boards = max(0, round(rng.gauss(
                reb * exposure + 0.5 * defense + 0.3 * home, 2.2,
            )))
            assists = max(0, round(rng.gauss(
                ast * exposure - 0.6 * defense + 0.2 * home, 2.0,
            )))
            made_three = max(0, min(score // 3, round(rng.gauss(
                threes * exposure - 0.25 * defense, 1.1,
            ))))
            games.append({
                "player": player,
                "game": game + 1,
                "date": (date(2026, 1, 1) + timedelta(days=game * 2)).isoformat(),
                "opponent_defense": defense,
                "home": home,
                "rest_days": rest,
                "MIN": round(minutes, 1),
                "PTS": score,
                "REB": boards,
                "AST": assists,
                "3PM": made_three,
                "PRA": score + boards + assists,
            })
    return games


def write_demo_database(games: list[dict], path: str | Path) -> Path:
    """Save synthetic-only data to a dedicated file, never the original NBA DB."""
    dest = Path(path).expanduser()
    if dest.name != "synthetic_nba_demo.sqlite":
        raise ValueError("Only the dedicated synthetic demo filename is permitted")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(dest) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS simulated_boxscores (
                player TEXT NOT NULL, game INTEGER NOT NULL,
                game_date TEXT NOT NULL, opponent_defense REAL NOT NULL,
                home INTEGER NOT NULL, rest_days INTEGER NOT NULL,
                minutes REAL NOT NULL, pts INTEGER NOT NULL,
                reb INTEGER NOT NULL, ast INTEGER NOT NULL,
                three_pm INTEGER NOT NULL, pra INTEGER NOT NULL,
                PRIMARY KEY (player, game)
            )
        """)
        conn.execute("DELETE FROM simulated_boxscores")
        conn.executemany("""
            INSERT INTO simulated_boxscores VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, [
            (g["player"], g["game"], g["date"], g["opponent_defense"],
             g["home"], g["rest_days"], g["MIN"], g["PTS"], g["REB"],
             g["AST"], g["3PM"], g["PRA"])
            for g in games
        ])
    return dest


def _average(numbers):
    return sum(numbers) / len(numbers)


def _features(history: list[dict], metric: str, upcoming: dict) -> list[float]:
    """Only past game values plus known/simulated pregame fields."""
    last5 = history[-5:]
    last10 = history[-10:]
    if len(last10) < 10:
        raise ValueError("Need 10 previous games")
    avg5 = _average([g[metric] for g in last5])
    avg10 = _average([g[metric] for g in last10])
    return [
        avg5, avg10,
        _average([g["MIN"] for g in last5]),
        upcoming["opponent_defense"],
        upcoming["home"],
        upcoming["rest_days"],
        avg5 - avg10,
    ]


def build_examples(games: list[dict], metric: str) -> list[dict]:
    """Form one row per match; game n uses games strictly before n."""
    if metric not in METRICS:
        raise ValueError("Unsupported metric")
    examples = []
    for player, *_ in PROFILES:
        player_games = sorted(
            (g for g in games if g["player"] == player),
            key=lambda g: g["game"],
        )
        for index in range(10, len(player_games)):
            examples.append({
                "player": player,
                "game": player_games[index]["game"],
                "x": _features(player_games[:index], metric, player_games[index]),
                "y": float(player_games[index][metric]),
                "baseline": _average(
                    [g[metric] for g in player_games[index-10:index]]
                ),
            })
    return examples


def chronological_split(examples: list[dict]) -> tuple[list, list, list]:
    """Split by per-player game number: 70% train, 15% val, 15% test."""
    train, validation, test = [], [], []
    for player, *_ in PROFILES:
        rows = sorted(
            (e for e in examples if e["player"] == player),
            key=lambda e: e["game"],
        )
        a = int(len(rows) * 0.70)
        b = int(len(rows) * 0.85)
        train.extend(rows[:a])
        validation.extend(rows[a:b])
        test.extend(rows[b:])
    assert train and validation and test
    assert max(e["game"] for e in train) < min(e["game"] for e in validation)
    assert max(e["game"] for e in validation) < min(e["game"] for e in test)
    return train, validation, test


def _linear_solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Small pivoted Gaussian elimination for a regularized normal equation."""
    size = len(b)
    matrix = [row[:] + [value] for row, value in zip(a, b)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(matrix[r][col]))
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        if abs(matrix[col][col]) < 1e-10:
            raise ArithmeticError("Ill-conditioned normal equation")
        scalar = matrix[col][col]
        for j in range(col, size + 1):
            matrix[col][j] /= scalar
        for row in range(size):
            if row == col:
                continue
            factor = matrix[row][col]
            for j in range(col, size + 1):
                matrix[row][j] -= factor * matrix[col][j]
    return [matrix[i][-1] for i in range(size)]


class SmallRidge:
    """Transparent ridge regressor implemented with Python stdlib only."""

    def __init__(self, alpha: float = 20.0):
        self.alpha = alpha

    def fit(self, x: list[list[float]], y: list[float]):
        if not x or len(x) != len(y):
            raise ValueError("Invalid training data")
        width = len(x[0])
        self.means = [_average([row[j] for row in x]) for j in range(width)]
        self.stds = [
            max(1e-8, statistics.pstdev([row[j] for row in x]))
            for j in range(width)
        ]
        scaled = [[1.0] + [
            (row[j] - self.means[j]) / self.stds[j]
            for j in range(width)
        ] for row in x]
        size = width + 1
        gram = [[sum(row[i] * row[j] for row in scaled)
                 + (self.alpha if i == j and i > 0 else 0.0)
                 for j in range(size)] for i in range(size)]
        rhs = [sum(row[i] * v for row, v in zip(scaled, y))
               for i in range(size)]
        self.coef = _linear_solve(gram, rhs)
        return self

    def predict_one(self, x: list[float]) -> float:
        if len(x) != len(self.means):
            raise ValueError("Feature dimension mismatch")
        return max(0.0, self.coef[0] + sum(
            w * ((v - mean) / std)
            for w, v, mean, std in zip(
                self.coef[1:], x, self.means, self.stds
            )
        ))


def _mae(actual: list[float], predicted: list[float]) -> float:
    return _average([abs(a - p) for a, p in zip(actual, predicted)])


def train_demo(games: list[dict]) -> dict:
    """Fit transparent ridge models; evaluate with forward chronological holdout."""
    results = {}
    for metric in METRICS:
        rows = build_examples(games, metric)
        train, val, test = chronological_split(rows)
        model = SmallRidge(alpha=20.0).fit(
            [r["x"] for r in train], [r["y"] for r in train]
        )
        val_errors = [
            r["y"] - model.predict_one(r["x"])
            for r in val
        ]
        test_pred = [model.predict_one(r["x"]) for r in test]
        results[metric] = {
            "model": model,
            "validation_errors": val_errors,
            "test_mae": _mae([r["y"] for r in test], test_pred),
            "baseline_mae": _mae(
                [r["y"] for r in test], [r["baseline"] for r in test]
            ),
            "train_n": len(train),
            "val_n": len(val),
            "test_n": len(test),
        }
    return results


def forecast(games: list[dict], models: dict, player: str,
             metric: str, threshold: float, defense: float = 0.0,
             home: bool = True, rest: int = 2) -> dict:
    """Future fabricated matchup; probability is an *uncalibrated* demo estimate."""
    if metric not in METRICS:
        raise ValueError("Unsupported metric")
    history = sorted(
        (g for g in games if g["player"] == player),
        key=lambda g: g["game"],
    )
    if not history:
        raise ValueError("Player not found")
    if not -1.0 <= defense <= 1.0 or not 1 <= rest <= 7:
        raise ValueError("Pregame inputs out of range")
    x = _features(history, metric, {
        "opponent_defense": defense,
        "home": int(home),
        "rest_days": rest,
    })
    prediction = models[metric]["model"].predict_one(x)
    errors = models[metric]["validation_errors"]
    probability = sum(prediction + e >= threshold for e in errors) / len(errors)
    return {
        "player": player,
        "metric": metric,
        "forecast": round(prediction, 2),
        "threshold": threshold,
        "over_probability": round(probability, 4),
        "samples_calibration": len(errors),
        "recent_values": [g[metric] for g in history[-10:]],
    }


def create_demo(data_dir: str | Path = "/content/NBA_Intelligence_Demo",
                seed: int = SEED) -> dict:
    """Prepare demo database and numerical model with no external network calls."""
    games = make_synthetic_games(seed)
    db_path = write_demo_database(
        games, Path(data_dir) / "synthetic_nba_demo.sqlite"
    )
    return {
        "games": games,
        "models": train_demo(games),
        "database_path": str(db_path),
        "seed": seed,
    }
