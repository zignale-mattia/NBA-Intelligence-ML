"""Real-ESPN-trained player forecasting, temporal validation, and uncertainty.

The training labels are observed NBA statistics, never fabricated. Historical
features use only games preceding the target. No odds or wagering use intended.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
import math
import statistics

from .real_espn import STATS

WARMUP = 15
FEATURE_NAMES = (
    "avg_5", "avg_10", "avg_15", "minutes_5", "minutes_10",
    "trend_5_vs_15", "home_game", "rest_days",
)


def _mean(items):
    if not items:
        raise ValueError("Empty series")
    return sum(items) / len(items)


def _features(history: list[dict], stat: str, home: bool, rest: int) -> list[float]:
    if len(history) < WARMUP:
        raise ValueError(f"Need {WARMUP} previous real games")
    values = [float(g[stat]) for g in history]
    minutes = [float(g["MIN"]) for g in history]
    last5 = _mean(values[-5:])
    last15 = _mean(values[-15:])
    return [
        last5, _mean(values[-10:]), last15,
        _mean(minutes[-5:]), _mean(minutes[-10:]),
        last5 - last15, int(bool(home)), float(max(1, min(rest, 7))),
    ]


def player_histories(games: list[dict]) -> dict:
    grouped = defaultdict(list)
    for game in games:
        grouped[game["player"]].append(game)
    return {
        name: sorted(values, key=lambda g: (g["date"], g["event_id"]))
        for name, values in grouped.items()
    }


def supervised_examples(games: list[dict], stat: str) -> list[dict]:
    if stat not in STATS:
        raise ValueError("Unexpected statistic")
    examples = []
    for player, logs in player_histories(games).items():
        for index in range(WARMUP, len(logs)):
            current = logs[index]
            history = logs[:index]
            prev_date = date.fromisoformat(history[-1]["date"])
            game_date = date.fromisoformat(current["date"])
            rest = max(1, (game_date - prev_date).days)
            examples.append({
                "player": player,
                "date": current["date"],
                "event_id": current["event_id"],
                "x": _features(history, stat, bool(current["home"]), rest),
                "y": float(current[stat]),
                "baseline": _mean([g[stat] for g in history[-10:]]),
            })
    return sorted(examples, key=lambda e: (e["date"], e["player"]))


def chronological_split(examples: list[dict]):
    """Partition UNIQUE global NBA calendar dates, not random games or players."""
    days = sorted({row["date"] for row in examples})
    if len(days) < 35:
        raise ValueError("Need at least 35 unique dates for held-out evaluation")
    cut1, cut2 = int(len(days) * 0.70), int(len(days) * 0.85)
    training_days, validation_days, test_days = (
        set(days[:cut1]), set(days[cut1:cut2]), set(days[cut2:])
    )
    train = [r for r in examples if r["date"] in training_days]
    val = [r for r in examples if r["date"] in validation_days]
    test = [r for r in examples if r["date"] in test_days]
    if min(len(train), len(val), len(test)) < 10:
        raise ValueError("Not enough rows in chronological split")
    assert max(e["date"] for e in train) < min(e["date"] for e in val)
    assert max(e["date"] for e in val) < min(e["date"] for e in test)
    return train, val, test


def _solve(gram: list[list[float]], rhs: list[float]) -> list[float]:
    """Small pivoted Gaussian elimination for Ridge fallback."""
    n = len(rhs)
    a = [row[:] + [v] for row, v in zip(gram, rhs)]
    for j in range(n):
        pivot = max(range(j, n), key=lambda r: abs(a[r][j]))
        a[j], a[pivot] = a[pivot], a[j]
        if abs(a[j][j]) < 1e-11:
            raise ArithmeticError("Singular ridge equation")
        denom = a[j][j]
        for k in range(j, n + 1):
            a[j][k] /= denom
        for i in range(n):
            if i == j:
                continue
            factor = a[i][j]
            for k in range(j, n + 1):
                a[i][k] -= factor * a[j][k]
    return [row[-1] for row in a]


class StandardLibraryRidge:
    """Data-derived baseline works even without sklearn (offline tests)."""

    def __init__(self, alpha=20.0):
        self.alpha = alpha

    def fit(self, x, y):
        if not x or len(x) != len(y):
            raise ValueError("Invalid examples")
        n_feat = len(x[0])
        self.center = [_mean([row[j] for row in x]) for j in range(n_feat)]
        self.std = [
            max(1e-7, statistics.pstdev([row[j] for row in x]))
            for j in range(n_feat)
        ]
        norm = [[1.0] + [
            (row[j] - self.center[j]) / self.std[j]
            for j in range(n_feat)
        ] for row in x]
        n = n_feat + 1
        self.beta = _solve([
            [
                sum(row[i] * row[j] for row in norm) +
                (self.alpha if i == j and i > 0 else 0.0)
                for j in range(n)
            ]
            for i in range(n)
        ], [
            sum(row[i] * v for row, v in zip(norm, y))
            for i in range(n)
        ])
        return self

    def predict(self, x):
        return [
            max(0.0, self.beta[0] + sum(
                w * (v - mu) / sd
                for w, v, mu, sd in zip(
                    self.beta[1:], row, self.center, self.std
                )
            ))
            for row in x
        ]


def _candidates(advanced: bool):
    candidates = {
        "Ridge (stdlib)": lambda: StandardLibraryRidge(alpha=20.0),
    }
    if advanced:
        try:
            from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return candidates
        candidates = {
            "Ridge": lambda: make_pipeline(
                StandardScaler(), Ridge(alpha=20.0)
            ),
            "Random Forest": lambda: RandomForestRegressor(
                n_estimators=90, max_depth=10, min_samples_leaf=5,
                n_jobs=2, random_state=42,
            ),
            "Extra Trees": lambda: ExtraTreesRegressor(
                n_estimators=90, max_depth=11, min_samples_leaf=5,
                n_jobs=2, random_state=42,
            ),
        }
    return candidates


def _predictions(model, rows: list[dict]) -> list[float]:
    return [max(0.0, float(v)) for v in model.predict(
        [r["x"] for r in rows]
    )]


def _mae(true, pred):
    return _mean([abs(float(a) - float(b)) for a, b in zip(true, pred)])


def train_real_models(games: list[dict], *, advanced: bool = True) -> dict:
    """Model selection on validation; final test set never used for selection."""
    result = {}
    available = _candidates(advanced)
    for stat in STATS:
        rows = supervised_examples(games, stat)
        train, val, test = chronological_split(rows)
        selection = []
        for name, factory in available.items():
            candidate = factory().fit(
                [r["x"] for r in train], [r["y"] for r in train],
            )
            val_preds = _predictions(candidate, val)
            loss = _mae([r["y"] for r in val], val_preds)
            selection.append((loss, name, factory, [
                r["y"] - prediction
                for r, prediction in zip(val, val_preds)
            ]))
        selection.sort(key=lambda e: e[0])
        val_loss, chosen_name, factory, val_residuals = selection[0]
        final_model = factory().fit(
            [r["x"] for r in train + val],
            [r["y"] for r in train + val],
        )
        test_preds = _predictions(final_model, test)
        result[stat] = {
            "model": final_model, "selected_model": chosen_name,
            "validation_mae": round(val_loss, 3),
            "validation_residuals": val_residuals,
            "test_mae": round(_mae(
                [r["y"] for r in test], test_preds
            ), 3),
            "baseline_test_mae": round(_mae(
                [r["y"] for r in test],
                [r["baseline"] for r in test],
            ), 3),
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "test_from": min(r["date"] for r in test),
            "test_through": max(r["date"] for r in test),
        }
    return result


def real_forecast(state: dict, player: str, stat: str, threshold: float,
                  home: bool, rest: int) -> dict:
    if stat not in STATS or player not in state["histories"]:
        raise ValueError("Unknown player or statistic")
    history = state["histories"][player]
    model_stats = state["models"][stat]
    if not math.isfinite(float(threshold)) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative")
    if int(rest) != rest or not 1 <= rest <= 7:
        raise ValueError("Rest days must be in 1..7")
    x = _features(history, stat, home, int(rest))
    prediction = max(0.0, float(model_stats["model"].predict([x])[0]))
    residuals = model_stats["validation_residuals"]
    # Empirical residual fraction; NOT a calibrated real-game probability.
    probability = _mean([
        1.0 if prediction + e >= float(threshold) else 0.0
        for e in residuals
    ])
    return {
        "player": player, "stat": stat, "prediction": round(prediction, 2),
        "threshold": float(threshold), "over_estimate": probability,
        "last10": history[-10:],
        "latest": history[-1],
        "model": model_stats["selected_model"],
        "validation_samples": len(residuals),
    }


def train_state(data: dict, advanced: bool = True) -> dict:
    """Prepare real-data forecasts without altering private original artifacts."""
    histories = player_histories(data["games"])
    eligible = {p: h for p, h in histories.items() if len(h) >= WARMUP + 12}
    if len(eligible) < 5:
        raise RuntimeError("Insufficient actual NBA player histories for modeling")
    selected = [g for g in data["games"] if g["player"] in eligible]
    return dict(
        data,
        games=selected, histories=eligible,
        models=train_real_models(selected, advanced=advanced),
    )
