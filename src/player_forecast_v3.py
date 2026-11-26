# Source: notebook cell 44 — Player forecasts
# NOTE: Colab-oriented notebook code; requires runtime/Drive configuration.
# Not verified as standalone executable.


# ============================================================
# STEP 17 — ADVANCED FORECAST ENGINE V3
# NBA DATA SCIENCE ANALYTICS
# ============================================================

import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

from google.colab import drive
from IPython.display import display

# ============================================================
# 1. RIPRISTINO AUTOMATICO AMBIENTE
# ============================================================

mounts = [
    "/content/drive",
    "/content/gdrive_nba",
    "/content/nba_drive"
]

BASE_DIR = None

for mount in mounts:

    candidate = os.path.join(
        mount, "MyDrive", "NBA_Data_Science"
    )

    if os.path.ismount(mount):

        if os.path.isfile(
            os.path.join(candidate, "nba_database.db")
        ):
            BASE_DIR = candidate
            break

if BASE_DIR is None:

    for mount in mounts:

        if os.path.ismount(mount):
            continue

        if os.path.isdir(mount) and os.listdir(mount):
            continue

        drive.mount(mount)

        candidate = os.path.join(
            mount, "MyDrive", "NBA_Data_Science"
        )

        if os.path.isfile(
            os.path.join(candidate, "nba_database.db")
        ):
            BASE_DIR = candidate
            break

if BASE_DIR is None:
    raise FileNotFoundError(
        "Cartella NBA_Data_Science non trovata."
    )

DB_PATH = os.path.join(
    BASE_DIR, "nba_database.db"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "nba_player_models_v3.joblib"
)

CSV_PATH = os.path.join(
    BASE_DIR, "NBA_Forecast_Engine_V3.csv"
)

EXCEL_PATH = os.path.join(
    BASE_DIR, "NBA_Forecast_Engine_V3.xlsx"
)

ARCHIVE_DIR = os.path.join(
    BASE_DIR, "Forecast_Archive_V3"
)

os.makedirs(ARCHIVE_DIR, exist_ok=True)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        "Modello V3 mancante. Completa STEP 16."
    )

NOW = pd.Timestamp.now(tz="UTC")

# ============================================================
# 2. CARICAMENTO MODELLO V3
# ============================================================

package_v3 = joblib.load(MODEL_PATH)

feature_cols = package_v3["features"]
models = package_v3["models"]

MIN_HISTORY = package_v3.get(
    "minimum_history", 20
)

print("Modello:", package_v3["model_version"])
print("Feature richieste:", len(feature_cols))

# ============================================================
# 3. CARICAMENTO DATABASE
# ============================================================

with sqlite3.connect(DB_PATH) as conn:

    box = pd.read_sql_query(
        "SELECT * FROM espn_player_boxscores",
        conn
    )

    games = pd.read_sql_query(
        "SELECT * FROM espn_games",
        conn
    )

for df in [box, games]:

    df["game_id"] = df["game_id"].astype(str)

    df["game_date"] = pd.to_datetime(
        df["game_date"],
        utc=True,
        errors="coerce",
        format="mixed"
    )

for c in ["player_id", "team_id"]:
    box[c] = box[c].astype(str)

for c in ["home_id", "away_id"]:
    games[c] = games[c].astype(str)

numeric = [
    "minutes", "pts", "reb", "ast",
    "fg3m", "fgm", "fga", "fta",
    "oreb", "tov", "did_not_play"
]

for c in numeric:

    box[c] = pd.to_numeric(
        box[c], errors="coerce"
    )

for c in [
    "home_score", "away_score", "completed"
]:

    games[c] = pd.to_numeric(
        games[c], errors="coerce"
    )

games = games.drop_duplicates("game_id")

completed = games[
    (games["completed"] == 1)
    & (games["game_date"] < NOW)
].dropna(
    subset=["home_score", "away_score"]
).copy()

box = box[
    (box["minutes"] > 0)
    & (box["did_not_play"] == 0)
    & box["game_id"].isin(completed["game_id"])
    & (box["game_date"] < NOW)
].dropna(
    subset=["pts", "reb", "ast", "fg3m"]
).drop_duplicates(
    ["game_id", "player_id"]
).copy()

# ============================================================
# 4. STATISTICHE AVANZATE
# ============================================================

def safe_divide(a, b):

    return a / b.replace(0, np.nan)

box["PRA"] = (
    box["pts"] + box["reb"] + box["ast"]
)

box["TS_pct"] = 100 * safe_divide(
    box["pts"],
    2 * (box["fga"] + 0.44 * box["fta"])
)

box["eFG_pct"] = 100 * safe_divide(
    box["fgm"] + 0.5 * box["fg3m"],
    box["fga"]
)

for stat, name in [
    ("pts", "PTS"),
    ("reb", "REB"),
    ("ast", "AST"),
    ("PRA", "PRA")
]:

    box[f"{name}_per_min"] = safe_divide(
        box[stat], box["minutes"]
    )

box["Offensive_Load"] = (
    box["fga"]
    + 0.44 * box["fta"]
    + box["tov"]
)

# ============================================================
# 5. TEAM ANALYTICS
# ============================================================

team_stats = box.groupby(
    ["game_id", "team_id"],
    as_index=False
).agg(
    Team_PTS=("pts", "sum"),
    Team_FGA=("fga", lambda s: s.sum(min_count=1)),
    Team_FTA=("fta", lambda s: s.sum(min_count=1)),
    Team_OREB=("oreb", lambda s: s.sum(min_count=1)),
    Team_TOV=("tov", lambda s: s.sum(min_count=1)),
    Team_MIN=("minutes", "sum")
)

team_stats["Possessions"] = (
    team_stats["Team_FGA"]
    + 0.44 * team_stats["Team_FTA"]
    - team_stats["Team_OREB"]
    + team_stats["Team_TOV"]
)

team_stats["Possessions"] = (
    team_stats["Possessions"]
    .where(team_stats["Possessions"] > 0)
)

team_stats["Offensive_Rating"] = (
    100 * safe_divide(
        team_stats["Team_PTS"],
        team_stats["Possessions"]
    )
)

team_stats["Team_Offensive_Load"] = (
    team_stats["Team_FGA"]
    + 0.44 * team_stats["Team_FTA"]
    + team_stats["Team_TOV"]
)

context = completed[[
    "game_id", "home_id", "away_id",
    "home_score", "away_score"
]]

box = box.merge(
    context,
    on="game_id",
    how="inner"
)

box = box[
    (box["team_id"] == box["home_id"])
    | (box["team_id"] == box["away_id"])
].copy()

box["home"] = (
    box["team_id"] == box["home_id"]
).astype(int)

box["opponent_id"] = np.where(
    box["home"] == 1,
    box["away_id"],
    box["home_id"]
)

box["Team_Score"] = np.where(
    box["home"] == 1,
    box["home_score"],
    box["away_score"]
)

box["Opponent_Score"] = np.where(
    box["home"] == 1,
    box["away_score"],
    box["home_score"]
)

box = box.merge(
    team_stats,
    on=["game_id", "team_id"],
    how="left"
)

opp_stats = team_stats[[
    "game_id", "team_id", "Possessions"
]].rename(columns={
    "team_id": "opponent_id",
    "Possessions": "Opponent_Possessions"
})

box = box.merge(
    opp_stats,
    on=["game_id", "opponent_id"],
    how="left"
)

box["Defensive_Rating"] = (
    100 * safe_divide(
        box["Opponent_Score"],
        box["Opponent_Possessions"]
    )
)

box["Pace"] = safe_divide(
    48 * (
        box["Possessions"]
        + box["Opponent_Possessions"]
    ) / 2,
    box["Team_MIN"] / 5
)

box["Usage_Proxy"] = (
    100 * safe_divide(
        box["Offensive_Load"]
        * (box["Team_MIN"] / 5),
        box["minutes"]
        * box["Team_Offensive_Load"]
    )
)

box = box.sort_values(
    ["game_date", "game_id"]
)

# ============================================================
# 6. STORICO GIOCATORI E SQUADRE
# ============================================================

histories = {
    pid: group.copy()
    for pid, group in box.groupby("player_id")
}

latest_team = box.drop_duplicates(
    "player_id",
    keep="last"
)

latest_team = latest_team[
    latest_team["game_date"]
    >= NOW - pd.Timedelta(days=90)
]

team_context = box[[
    "game_id",
    "game_date",
    "team_id",
    "Team_Score",
    "Opponent_Score",
    "Offensive_Rating",
    "Defensive_Rating",
    "Pace"
]].drop_duplicates(
    ["game_id", "team_id"]
).sort_values("game_date")

team_histories = {
    tid: group.copy()
    for tid, group in team_context.groupby("team_id")
}

# ============================================================
# 7. PARTITE FUTURE
# ============================================================

future = games[
    (games["completed"] == 0)
    & (games["status"] == "STATUS_SCHEDULED")
    & (games["game_date"] > NOW)
    & (
        games["game_date"]
        <= NOW + pd.Timedelta(days=3)
    )
].sort_values("game_date")

if future.empty:

    raise RuntimeError(
        "Nessuna partita futura. Esegui STEP 12."
    )

print("Partite future:", len(future))

# ============================================================
# 8. FEATURE ENGINEERING V3
# ============================================================

rolling_stats = [
    "pts", "reb", "ast", "fg3m",
    "PRA", "minutes",
    "TS_pct", "eFG_pct",
    "PTS_per_min",
    "REB_per_min",
    "AST_per_min",
    "PRA_per_min",
    "Usage_Proxy",
    "Pace",
    "Offensive_Rating",
    "Defensive_Rating"
]

momentum_stats = [
    "pts", "reb", "ast",
    "fg3m", "PRA", "minutes"
]

team_features = [
    "Team_Score",
    "Opponent_Score",
    "Offensive_Rating",
    "Defensive_Rating",
    "Pace"
]

def scalar_divide(a, b):

    if pd.isna(b) or b == 0:
        return np.nan

    return a / b


def build_advanced_features(
    history, opponent_id, date, home
):

    history = history[
        history["game_date"] < NOW
    ].sort_values("game_date")

    if len(history) < MIN_HISTORY:
        return None

    row = {}

    for stat in rolling_stats:

        for window in [5, 10, 20]:

            values = history[stat].tail(window)

            row[f"{stat}_avg{window}"] = (
                values.mean()
                if values.count() >= window
                else np.nan
            )

        values = history[stat].tail(10)

        row[f"{stat}_std10"] = (
            values.std(ddof=1)
            if values.count() >= 10
            else np.nan
        )

        row[f"{stat}_last1"] = (
            history[stat].iloc[-1]
        )

    # Momentum e consistency

    for stat in momentum_stats:

        momentum = (
            row[f"{stat}_avg5"]
            - row[f"{stat}_avg20"]
        )

        variation = scalar_divide(
            row[f"{stat}_std10"],
            row[f"{stat}_avg10"]
        )

        row[f"{stat}_momentum"] = momentum

        row[f"{stat}_momentum_pct"] = (
            100 * scalar_divide(
                momentum,
                row[f"{stat}_avg20"]
            )
        )

        row[f"{stat}_variation"] = variation

        row[f"{stat}_consistency"] = (
            100 / (1 + variation)
        )

    # Riposo e fattore campo

    game_day = date.tz_convert(
        "America/New_York"
    ).date()

    last_day = history.iloc[-1][
        "game_date"
    ].tz_convert(
        "America/New_York"
    ).date()

    days = (game_day - last_day).days

    row["home"] = int(home)

    row["Rest_Days"] = np.clip(
        days - 1, 0, 30
    )

    row["Back_to_Back"] = int(days == 1)

    row["Games_Previous"] = len(history)

    # Performance casa / trasferta

    venue = history[
        history["home"] == home
    ].tail(10)

    for stat in ["pts", "reb", "ast", "PRA"]:

        row[f"{stat}_venue_avg10"] = (
            venue[stat].mean()
            if venue[stat].count() >= 3
            else np.nan
        )

    # Contesto avversario

    opponent = team_histories.get(opponent_id)

    for stat in team_features:

        key = f"Opp_pre_{stat}_avg10"

        if opponent is None:

            row[key] = np.nan

        else:

            values = opponent[stat].tail(10)

            row[key] = (
                values.mean()
                if values.count() >= 5
                else np.nan
            )

    return row

# ============================================================
# 9. GENERAZIONE DATASET FUTURO
# ============================================================

rows = []

for game in future.itertuples(index=False):

    matchups = [
        (
            game.home_id,
            game.home_team,
            game.away_id,
            game.away_team,
            1
        ),
        (
            game.away_id,
            game.away_team,
            game.home_id,
            game.home_team,
            0
        )
    ]

    for tid, team, oid, opponent, home in matchups:

        roster = latest_team[
            latest_team["team_id"] == tid
        ]

        for player in roster.itertuples(index=False):

            features = build_advanced_features(
                histories[player.player_id],
                oid,
                game.game_date,
                home
            )

            if features is None:
                continue

            rows.append({
                "game_id": game.game_id,
                "player_id": player.player_id,
                "date": game.game_date,
                "Giocatore": player.player_name,
                "Squadra": team,
                "Avversario": opponent,
                "Sede": "Casa" if home else "Trasferta",
                **features
            })

forecast_v3 = pd.DataFrame(rows)

if forecast_v3.empty:
    raise ValueError("Nessun giocatore idoneo.")

missing = [
    c for c in feature_cols
    if c not in forecast_v3.columns
]

if missing:

    raise ValueError(
        f"Feature incompatibili: {missing}"
    )

X = forecast_v3[feature_cols].replace(
    [np.inf, -np.inf],
    np.nan
)

print("Player-game forecast:", len(X))
print("Feature disponibili:", len(X.columns))

# ============================================================
# 10. MACHINE LEARNING ED ENSEMBLE
# ============================================================

for target, info in models.items():

    prediction = np.zeros(
        len(forecast_v3)
    )

    for name, weight in zip(
        info["members"],
        info["weights"]
    ):

        if name == "Last 10 Baseline":

            member_prediction = (
                forecast_v3[info["baseline_column"]]
                .fillna(info["fallback"])
                .to_numpy(dtype=float)
            )

        else:

            model = info["models"][name]

            member_prediction = model.predict(X)

        prediction += (
            weight * np.maximum(
                member_prediction, 0
            )
        )

    forecast_v3[f"Forecast_{target}"] = prediction

    forecast_v3[f"Model_{target}"] = info["name"]

# Manteniamo il PRA previsto dal modello V3.
# Aggiungiamo anche la somma delle previsioni individuali.

forecast_v3["PRA_Sum"] = (
    forecast_v3["Forecast_PTS"]
    + forecast_v3["Forecast_REB"]
    + forecast_v3["Forecast_AST"]
)

# ============================================================
# 11. CONTROLLO QUALITA'
# ============================================================

TARGETS = ["PTS", "REB", "AST", "3PM", "PRA"]

generated_at = pd.Timestamp.now(tz="UTC")

forecast_v3["generated_at"] = generated_at

forecast_v3 = forecast_v3[
    forecast_v3["date"] > generated_at
].copy()

forecast_v3 = forecast_v3.replace(
    [np.inf, -np.inf],
    np.nan
).dropna(
    subset=[
        f"Forecast_{t}"
        for t in TARGETS
    ]
).drop_duplicates(
    ["game_id", "player_id"]
)

if forecast_v3.empty:
    raise ValueError("Nessun forecast valido.")

forecast_v3["Data Italia"] = (
    forecast_v3["date"]
    .dt.tz_convert("Europe/Rome")
    .dt.strftime("%d/%m/%Y %H:%M")
)

forecast_v3["Model_Version"] = "3.0"

forecast_v3["Roster_Status"] = (
    "Stimato - disponibilità non verificata"
)

# ============================================================
# 12. ARCHIVIAZIONE SQLITE SEPARATA V3
# ============================================================

with sqlite3.connect(DB_PATH) as conn:

    conn.execute("""
        CREATE TABLE IF NOT EXISTS nba_forecast_history_v3 (
            game_id TEXT NOT NULL,
            player_id TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            game_date TEXT NOT NULL,
            player_name TEXT,
            team_name TEXT,
            opponent TEXT,
            forecast_pts REAL,
            forecast_reb REAL,
            forecast_ast REAL,
            forecast_3pm REAL,
            forecast_pra REAL,
            model_version TEXT,
            PRIMARY KEY (
                game_id, player_id, generated_at
            )
        )
    """)

    records = []

    for _, r in forecast_v3.iterrows():

        records.append((
            str(r["game_id"]),
            str(r["player_id"]),
            r["generated_at"].isoformat(),
            r["date"].isoformat(),
            r["Giocatore"],
            r["Squadra"],
            r["Avversario"],
            float(r["Forecast_PTS"]),
            float(r["Forecast_REB"]),
            float(r["Forecast_AST"]),
            float(r["Forecast_3PM"]),
            float(r["Forecast_PRA"]),
            "3.0"
        ))

    conn.executemany("""
        INSERT OR IGNORE INTO nba_forecast_history_v3
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)

    total_history = conn.execute("""
        SELECT COUNT(*)
        FROM nba_forecast_history_v3
    """).fetchone()[0]

# ============================================================
# 13. ESPORTAZIONE FORECAST
# ============================================================

columns = [
    "game_id",
    "player_id",
    "date",
    "generated_at",
    "Data Italia",
    "Giocatore",
    "Squadra",
    "Avversario",
    "Sede",
    "Forecast_PTS",
    "Forecast_REB",
    "Forecast_AST",
    "Forecast_3PM",
    "Forecast_PRA",
    "PRA_Sum",
    "minutes_avg10",
    "pts_avg10",
    "pts_momentum",
    "pts_consistency",
    "Usage_Proxy_avg10",
    "TS_pct_avg10",
    "Rest_Days",
    "Back_to_Back",
    "Model_Version",
    "Roster_Status"
]

columns += [
    f"Model_{t}"
    for t in TARGETS
]

export_v3 = forecast_v3[columns].copy()

export_v3 = export_v3.sort_values(
    ["date", "Forecast_PTS"],
    ascending=[True, False]
)

for c in ["date", "generated_at"]:

    export_v3[c] = export_v3[c].map(
        lambda x: x.isoformat()
    )

stamp = generated_at.strftime(
    "%Y%m%d_%H%M%S_%f"
)

archive_path = os.path.join(
    ARCHIVE_DIR,
    f"NBA_V3_{stamp}.csv"
)

export_v3.to_csv(
    archive_path,
    index=False,
    encoding="utf-8-sig"
)

export_v3.to_csv(
    CSV_PATH + ".tmp",
    index=False,
    encoding="utf-8-sig"
)

with pd.ExcelWriter(
    EXCEL_PATH + ".tmp.xlsx",
    engine="openpyxl"
) as writer:

    export_v3.to_excel(
        writer,
        sheet_name="Player Forecast",
        index=False
    )

    pd.DataFrame(
        package_v3["test"]
    ).to_excel(
        writer,
        sheet_name="Model Accuracy",
        index=False
    )

os.replace(CSV_PATH + ".tmp", CSV_PATH)

os.replace(
    EXCEL_PATH + ".tmp.xlsx",
    EXCEL_PATH
)

# ============================================================
# 14. CONFRONTO FORECAST V2 VS V3
# ============================================================

V2_PATH = os.path.join(
    BASE_DIR, "NBA_Forecast_Engine_V2.csv"
)

if os.path.isfile(V2_PATH):

    v2 = pd.read_csv(
        V2_PATH,
        dtype={
            "game_id": str,
            "player_id": str
        }
    )

    keys = ["game_id", "player_id"]

    if all(c in v2.columns for c in keys):

        common = v2.merge(
            export_v3,
            on=keys,
            suffixes=("_V2", "_V3")
        )

        for target in TARGETS:

            common[f"Delta_{target}"] = (
                common[f"Forecast_{target}_V3"]
                - common[f"Forecast_{target}_V2"]
            )

        comparison_path = os.path.join(
            BASE_DIR,
            "NBA_V2_vs_V3_Forecasts.csv"
        )

        common.to_csv(
            comparison_path,
            index=False
        )

        print(
            "Forecast confrontabili V2/V3:",
            len(common)
        )

# ============================================================
# 15. VISUALIZZAZIONE
# ============================================================

print("\nTOP 20 — POINTS FORECAST")

top_pts = export_v3.sort_values(
    "Forecast_PTS",
    ascending=False
).head(20)

display(
    top_pts[[
        "Giocatore",
        "Squadra",
        "Avversario",
        "Forecast_PTS",
        "Forecast_REB",
        "Forecast_AST",
        "Forecast_3PM",
        "Forecast_PRA"
    ]].round(2)
)

fig = px.bar(
    top_pts,
    x="Giocatore",
    y="Forecast_PTS",
    color="Squadra",
    hover_data=[
        "Avversario",
        "Data Italia",
        "Forecast_REB",
        "Forecast_AST",
        "Forecast_PRA"
    ],
    title="NBA V3 — Top Points Forecast",
    template="plotly_dark"
)

fig.update_layout(
    height=550,
    xaxis_tickangle=-45
)

fig.show()

top_pra = export_v3.sort_values(
    "Forecast_PRA",
    ascending=False
).head(20)

fig = px.bar(
    top_pra,
    x="Giocatore",
    y="Forecast_PRA",
    color="Squadra",
    title="NBA V3 — Top PRA Forecast",
    template="plotly_dark"
)

fig.show()

# ============================================================
# 16. RIEPILOGO FINALE
# ============================================================

print("\n" + "=" * 55)
print("STEP 17 COMPLETATO!")
print("=" * 55)

print("Versione: V3.0")
print("Partite future:", export_v3["game_id"].nunique())
print("Giocatori:", export_v3["player_id"].nunique())
print("Forecast generati:", len(export_v3))
print("Forecast storici V3:", total_history)
print("Feature ML:", len(feature_cols))

print("\nMODELLI UTILIZZATI")

for target, info in models.items():

    print(target, "->", info["name"])

print("\nFILE SALVATI")

print(CSV_PATH)
print(EXCEL_PATH)
print(archive_path)

print("\nPRONTO PER STEP 18!")
