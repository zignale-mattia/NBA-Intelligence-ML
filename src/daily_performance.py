# Source: notebook cell 91 — Daily performance tracking
# NOTE: Colab-oriented notebook code; requires runtime/Drive configuration.
# Not verified as standalone executable.


# ============================================================
# STEP 28 — DAILY PERFORMANCE TRACKER V2
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import ipywidgets as widgets

from IPython.display import display, clear_output

ROOT28 = Path(
    "/content/drive/MyDrive/NBA_Data_Science"
)

PERF28 = ROOT28 / "Probability_Backtesting"

SOURCE28 = PERF28 / "Evaluated_Predictions.csv"


def daily_performance():

    if not SOURCE28.exists():

        raise RuntimeError(
            "Completare prima lo STEP 27B. "
            "Archivio backtesting non disponibile."
        )

    df = pd.read_csv(
        SOURCE28,
        dtype={
            "game_id": str,
            "player_id": str
        }
    )

    if df.empty:
        raise RuntimeError(
            "Nessuna previsione valutabile."
        )

    df["Tipoff"] = pd.to_datetime(
        df["Tipoff"],
        format="mixed",
        utc=True,
        errors="coerce"
    )

    df = df.dropna(subset=["Tipoff"])

    df["NBA_Date"] = (
        df["Tipoff"]
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )

    df["Italy_Date"] = (
        df["Tipoff"]
        .dt.tz_convert("Europe/Rome")
        .dt.strftime("%Y-%m-%d")
    )

    numeric = [
        "forecast",
        "Actual",
        "threshold",
        "probability",
        "Success",
        "Correct",
        "Brier",
        "Baseline_Brier"
    ]

    for col in numeric:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df["Probability_pct"] = (
        df["probability"] * 100
    )

    df["Outcome"] = np.where(
        df["Success"] == 1,
        "RAGGIUNTA",
        "NON RAGGIUNTA"
    )

    df["Error"] = (
        df["forecast"] - df["Actual"]
    )

    df["Absolute_Error"] = df["Error"].abs()

    # Una previsione numerica per statistica

    point = (
        df.sort_values("generated_at")
        .drop_duplicates(
            ["game_id", "player_id", "target"],
            keep="last"
        )
        .copy()
    )

    # Performance giornaliera

    daily = df.groupby("NBA_Date").agg(
        Predictions=("Success", "size"),
        Games=("game_id", "nunique"),
        Players=("player_id", "nunique"),
        Expected_Success=("probability", "mean"),
        Actual_Success=("Success", "mean"),
        Accuracy=("Correct", "mean"),
        Brier_Score=("Brier", "mean")
    ).reset_index()

    daily["Calibration_Gap"] = (
        daily["Expected_Success"]
        - daily["Actual_Success"]
    ).abs()

    # Accuratezza forecast numerici

    mae = point.groupby(
        ["NBA_Date", "target"]
    ).agg(
        Samples=("Actual", "size"),
        MAE=("Absolute_Error", "mean"),
        Bias=("Error", "mean")
    ).reset_index()

    # Performance per fascia probabilistica

    df["Confidence_Band"] = pd.cut(
        df["probability"],
        bins=[
            0, .50, .60, .70,
            .80, .90, .95, 1
        ],
        labels=[
            "0-50%",
            "50-60%",
            "60-70%",
            "70-80%",
            "80-90%",
            "90-95%",
            "95-100%"
        ],
        include_lowest=True
    )

    confidence = df.groupby(
        ["NBA_Date", "Confidence_Band"],
        observed=True
    ).agg(
        Predictions=("Success", "size"),
        Expected=("probability", "mean"),
        Actual=("Success", "mean"),
        Brier_Score=("Brier", "mean")
    ).reset_index()

    confidence["Gap"] = (
        confidence["Expected"]
        - confidence["Actual"]
    ).abs()

    # Performance per giocatore

    players = df.groupby(
        ["player_id", "player_name", "target"]
    ).agg(
        Predictions=("Success", "size"),
        Games=("game_id", "nunique"),
        Expected=("probability", "mean"),
        Hit_Rate=("Success", "mean"),
        Accuracy=("Correct", "mean"),
        Brier_Score=("Brier", "mean")
    ).reset_index()

    # Confronto benchmark sulle stesse osservazioni

    paired = df.dropna(
        subset=["Baseline_Brier", "Brier"]
    )

    benchmark = paired.groupby(
        ["NBA_Date", "target"]
    ).agg(
        Samples=("Brier", "size"),
        ML_Brier=("Brier", "mean"),
        Baseline_Brier=("Baseline_Brier", "mean")
    ).reset_index()

    benchmark["Improvement_pct"] = np.where(
        benchmark["Baseline_Brier"] > 0,
        100 * (
            1 - benchmark["ML_Brier"]
            / benchmark["Baseline_Brier"]
        ),
        np.nan
    )

    # Ultima giornata valutabile

    latest_date = df["NBA_Date"].max()

    latest = df[
        df["NBA_Date"] == latest_date
    ].copy()

    latest_point = point[
        point["NBA_Date"] == latest_date
    ].copy()

    latest_point = latest_point.sort_values(
        "Absolute_Error"
    )

    # Salvataggio report

    reports = {
        "Daily_Performance": daily,
        "Daily_MAE": mae,
        "Daily_Confidence": confidence,
        "Player_Reliability": players,
        "Daily_Benchmark": benchmark,
        "Latest_Matches": latest,
        "Latest_Player_Forecasts": latest_point
    }

    for name, table in reports.items():

        temp = PERF28 / f"{name}.tmp"

        table.to_csv(temp, index=False)

        temp.replace(
            PERF28 / f"{name}.csv"
        )

    with pd.ExcelWriter(
        PERF28 / "NBA_Daily_Performance.xlsx",
        engine="openpyxl"
    ) as writer:

        for name, table in reports.items():

            table.to_excel(
                writer,
                sheet_name=name[:31],
                index=False
            )

    print("=" * 65)
    print("NBA INTELLIGENCE — DAILY PERFORMANCE")
    print("=" * 65)

    print("Ultima giornata valutata:", latest_date)

    print(
        "Partite:",
        latest["game_id"].nunique()
    )

    print(
        "Giocatori:",
        latest["player_id"].nunique()
    )

    print(
        "Soglie analizzate:",
        len(latest)
    )

    print(
        "Brier Score:",
        round(latest["Brier"].mean(), 4)
    )

    print("\nPERFORMANCE PER FASCIA")

    display(
        confidence[
            confidence["NBA_Date"] == latest_date
        ]
    )

    print("\nMIGLIORI FORECAST NUMERICI")

    columns = [
        "player_name",
        "target",
        "forecast",
        "Actual",
        "Absolute_Error"
    ]

    display(
        latest_point[columns].head(15)
    )

    print("\nPEGGIORI FORECAST NUMERICI")

    display(
        latest_point[columns]
        .sort_values(
            "Absolute_Error",
            ascending=False
        ).head(15)
    )

    print("\nSTEP 28 COMPLETATO")

    return df


daily_results = daily_performance()


# ============================================================
# PLAYER PERFORMANCE EXPLORER
# ============================================================

player_selector = widgets.Dropdown(
    options=sorted(
        daily_results["player_name"]
        .dropna().unique().tolist()
    ),
    description="Player:",
    layout=widgets.Layout(width="420px")
)

target_selector = widgets.Dropdown(
    options=["PTS", "REB", "AST", "3PM", "PRA"],
    value="PTS",
    description="Stat:"
)

date_selector = widgets.Dropdown(
    options=sorted(
        daily_results["NBA_Date"].unique(),
        reverse=True
    ),
    description="Date:"
)

output28 = widgets.Output()


def show_player_performance(change=None):

    with output28:

        clear_output(wait=True)

        selected = daily_results[
            (
                daily_results["player_name"]
                == player_selector.value
            )
            & (
                daily_results["target"]
                == target_selector.value
            )
            & (
                daily_results["NBA_Date"]
                == date_selector.value
            )
        ].copy()

        if selected.empty:

            print(
                "Nessuna previsione valutabile "
                "per questa selezione."
            )

            return

        selected = selected.sort_values(
            "threshold"
        )

        for game_id, game in selected.groupby("game_id"):

            row = game.iloc[0]

            print("=" * 55)
            print(row["player_name"])
            print("Partita:", game_id)
            print("Statistica:", row["target"])
            print("Forecast:", round(row["forecast"], 2))
            print("Risultato:", row["Actual"])

            print(
                "Errore assoluto:",
                round(row["Absolute_Error"], 2)
            )

            display(
                game[
                    [
                        "threshold",
                        "Probability_pct",
                        "Actual",
                        "Outcome"
                    ]
                ].round(2)
            )

            fig = px.bar(
                game,
                x="threshold",
                y="Probability_pct",
                color="Outcome",
                color_discrete_map={
                    "RAGGIUNTA": "#22c55e",
                    "NON RAGGIUNTA": "#ef4444"
                },
                template="plotly_dark",
                title="Forecast vs Actual — Probability Engine"
            )

            fig.update_yaxes(range=[0, 100])

            fig.show()


for control in [
    player_selector,
    target_selector,
    date_selector
]:

    control.observe(
        show_player_performance,
        names="value"
    )

display(
    widgets.VBox([
        player_selector,
        target_selector,
        date_selector,
        output28
    ])
)

show_player_performance()
