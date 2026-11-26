# Source: notebook cell 88 — Historical probability evaluation
# NOTE: Colab-oriented notebook code; requires runtime/Drive configuration.
# Not verified as standalone executable.


# ============================================================
# STEP 27 — PROBABILITY BACKTESTING ENGINE
# ============================================================

from google.colab import drive
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px

drive.mount("/content/drive")

BT_ROOT = Path("/content/drive/MyDrive/NBA_Data_Science")
BT_DB = BT_ROOT / "nba_database.db"

def probability_backtest():

    with sqlite3.connect(BT_DB) as conn:

        predictions = pd.read_sql_query("""
            SELECT *
            FROM nba_probability_history_v3
        """, conn)

        games = pd.read_sql_query("""
            SELECT *
            FROM espn_games
        """, conn)

        actual = pd.read_sql_query("""
            SELECT *
            FROM espn_player_boxscores
        """, conn)

    for df in [predictions, games, actual]:
        df["game_id"] = df["game_id"].astype(str)

    for df in [predictions, actual]:
        df["player_id"] = df["player_id"].astype(str)

    games = games[
        pd.to_numeric(
            games["completed"], errors="coerce"
        ) == 1
    ].copy()

    actual = actual[
        pd.to_numeric(
            actual["did_not_play"], errors="coerce"
        ) == 0
    ].copy()

    actual = actual[
        pd.to_numeric(
            actual["minutes"], errors="coerce"
        ) > 0
    ].copy()

    games["Tipoff"] = pd.to_datetime(
        games["game_date"], utc=True, errors="coerce"
    )

    predictions["generated_at"] = pd.to_datetime(
        predictions["generated_at"],
        utc=True,
        errors="coerce"
    )

    # Classificazione stagione: senza deduzioni arbitrarie

    games["Season_Type"] = "Unknown"

    for column in [
        "season_type", "season_type_id",
        "season_type_name"
    ]:

        if column in games.columns:

            values = games[column].astype(str).str.lower()

            games.loc[
                values.isin([
                    "2", "2.0", "regular",
                    "regular season"
                ]),
                "Season_Type"
            ] = "Regular Season"

            games.loc[
                values.isin([
                    "3", "3.0", "postseason",
                    "playoffs"
                ]),
                "Season_Type"
            ] = "Postseason"

    games = games.drop_duplicates("game_id")

    predictions = predictions.merge(
        games[
            ["game_id", "Tipoff", "Season_Type"]
        ],
        on="game_id",
        how="inner"
    )

    # Vietate previsioni successive all'inizio

    predictions = predictions[
        predictions["generated_at"] <
        predictions["Tipoff"]
    ].copy()

    predictions["probability"] = pd.to_numeric(
        predictions["probability"],
        errors="coerce"
    )

    predictions["threshold"] = pd.to_numeric(
        predictions["threshold"],
        errors="coerce"
    )

    predictions = predictions[
        predictions["probability"].between(0, 1)
        & predictions["threshold"].notna()
        & (predictions["threshold"] > 0)
    ]

    # Una previsione per evento:
    # ultima disponibile prima della partita

    predictions = predictions.sort_values(
        "generated_at"
    ).drop_duplicates(
        [
            "game_id", "player_id",
            "target", "threshold"
        ],
        keep="last"
    )

    target_map = {
        "PTS": "pts",
        "REB": "reb",
        "AST": "ast",
        "3PM": "fg3m",
        "PRA": "PRA"
    }

    for col in ["pts", "reb", "ast", "fg3m"]:
        actual[col] = pd.to_numeric(
            actual[col], errors="coerce"
        )

    actual["PRA"] = (
        actual["pts"] +
        actual["reb"] +
        actual["ast"]
    )

    actual = actual.drop_duplicates(
        ["game_id", "player_id"]
    )

    actual["Played_Date"] = pd.to_datetime(
        actual["game_date"],
        utc=True,
        errors="coerce"
    )

    actual = actual.sort_values("Played_Date")

    long_actual = actual.melt(
        id_vars=["game_id", "player_id"],
        value_vars=list(target_map.values()),
        var_name="Stat",
        value_name="Actual"
    )

    reverse_map = {
        v: k for k, v in target_map.items()
    }

    long_actual["target"] = (
        long_actual["Stat"].map(reverse_map)
    )

    result = predictions.merge(
        long_actual[
            ["game_id", "player_id", "target", "Actual"]
        ],
        on=["game_id", "player_id", "target"],
        how="inner"
    )

    result = result.dropna(subset=["Actual"])

    if result.empty:
        raise RuntimeError(
            "Nessuna previsione pre-partita "
            "con risultato effettivo disponibile."
        )

    result["Success"] = (
        result["Actual"] >= result["threshold"]
    ).astype(int)

    result["Brier"] = (
        result["probability"] - result["Success"]
    ) ** 2

    result["Correct"] = (
        (result["probability"] >= 0.5).astype(int)
        == result["Success"]
    ).astype(int)

    result["Baseline_Probability"] = np.nan

    # Benchmark storico: ultime 20 partite
    # esclusivamente anteriori alla previsione.
    # Margine prudenziale di 6 ore per evitare
    # partite iniziate ma non ancora concluse.

    history = {
        pid: group
        for pid, group in actual.groupby("player_id")
    }

    grouping = [
        "game_id", "player_id",
        "target", "generated_at"
    ]

    for key, subset in result.groupby(grouping):

        gid, pid, target, generated = key

        player_history = history.get(pid)

        if player_history is None:
            continue

        previous = player_history[
            player_history["Played_Date"]
            < generated - pd.Timedelta(hours=6)
        ].tail(20)

        if target not in target_map:
            continue

        values = previous[
            target_map[target]
        ].dropna().to_numpy()

        if len(values) < 10:
            continue

        thresholds = subset["threshold"].to_numpy()

        baseline = (
            values[:, None] >= thresholds[None, :]
        ).mean(axis=0)

        result.loc[
            subset.index,
            "Baseline_Probability"
        ] = baseline

    result["Baseline_Brier"] = (
        result["Baseline_Probability"]
        - result["Success"]
    ) ** 2

    result["Probability_Band"] = pd.cut(
        result["probability"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
        labels=[
            f"{i*10}-{(i+1)*10}%"
            for i in range(10)
        ]
    )

    # Funzione di aggregazione

    def summarize(group_columns):

        report = result.groupby(
            group_columns,
            observed=True
        ).agg(
            Predictions=("Success", "size"),
            Successes=("Success", "sum"),
            Mean_Probability=("probability", "mean"),
            Hit_Rate=("Success", "mean"),
            Classification_Accuracy=("Correct", "mean"),
            Brier_Score=("Brier", "mean")
        ).reset_index()

        report["Calibration_Gap"] = (
            report["Hit_Rate"] -
            report["Mean_Probability"]
        ).abs()

        return report

    calibration = summarize(["Probability_Band"])

    players = summarize(
        ["player_id", "player_name", "target"]
    )

    targets = summarize(["target"])

    thresholds = summarize(
        ["target", "threshold"]
    )

    seasons = summarize(
        ["Season_Type", "target"]
    )

    paired = result.dropna(
        subset=["Baseline_Brier"]
    )

    baseline_report = paired.groupby(
        "target"
    ).agg(
        Comparisons=("Brier", "size"),
        ML_Brier=("Brier", "mean"),
        Baseline_Brier=("Baseline_Brier", "mean")
    ).reset_index()

    baseline_report["Improvement_pct"] = np.where(
        baseline_report["Baseline_Brier"] > 0,
        100 * (
            1 - baseline_report["ML_Brier"]
            / baseline_report["Baseline_Brier"]
        ),
        np.nan
    )

    ece = np.average(
        calibration["Calibration_Gap"],
        weights=calibration["Predictions"]
    )

    overview = pd.DataFrame([{
        "Predictions": len(result),
        "Unique_Games": result["game_id"].nunique(),
        "Player_Games": result[
            ["player_id", "game_id"]
        ].drop_duplicates().shape[0],
        "Unique_Players": result["player_id"].nunique(),
        "Brier_Score": result["Brier"].mean(),
        "ECE": ece,
        "Updated_At": pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    }])

    reports = {
        "Overview": overview,
        "Calibration": calibration,
        "Players": players,
        "Targets": targets,
        "Thresholds": thresholds,
        "Season Types": seasons,
        "Baseline": baseline_report
    }

    output = BT_ROOT / "Probability_Backtesting"
    output.mkdir(exist_ok=True)

    for name, table in reports.items():
        table.to_csv(
            output / f"{name}.csv",
            index=False
        )

    result.to_csv(
        output / "Evaluated_Predictions.csv",
        index=False
    )

    with pd.ExcelWriter(
        output / "NBA_Probability_Backtesting.xlsx"
    ) as writer:

        for name, table in reports.items():
            table.to_excel(
                writer,
                sheet_name=name,
                index=False
            )

    print("STEP 27 COMPLETATO")
    display(overview)
    display(calibration)
    display(baseline_report)

    fig = px.scatter(
        calibration,
        x="Mean_Probability",
        y="Hit_Rate",
        size="Predictions",
        hover_name="Probability_Band",
        title="Probability Calibration",
        template="plotly_dark"
    )

    fig.add_shape(
        type="line",
        x0=0, y0=0,
        x1=1, y1=1,
        line=dict(dash="dash")
    )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    fig.show()

    return result

bt_results = probability_backtest()
