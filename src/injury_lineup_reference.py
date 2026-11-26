# Source: notebook cell 47 — Injury and lineup context
# NOTE: Colab-oriented notebook code; requires runtime/Drive configuration.
# Not verified as standalone executable.


# ============================================================
# STEP 19 — INJURY & LINEUP INTELLIGENCE
# NBA DATA SCIENCE ANALYTICS
# VERSION 3.2
# ============================================================

import os
import sqlite3
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import ipywidgets as widgets

from google.colab import drive
from IPython.display import display, clear_output
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

    if os.path.isfile(
        os.path.join(candidate, "nba_database.db")
    ):
        BASE_DIR = candidate
        break

if BASE_DIR is None:

    for mount in mounts:

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

NOW = pd.Timestamp.now(tz="UTC")

DB_PATH = os.path.join(
    BASE_DIR, "nba_database.db"
)

FORECAST_PATH = os.path.join(
    BASE_DIR, "NBA_Forecast_Engine_V3.csv"
)

PROBABILITY_PATH = os.path.join(
    BASE_DIR, "NBA_Probability_Engine_V3.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "NBA_Context_Aware_Forecast.xlsx"
)

CSV_PATH = os.path.join(
    BASE_DIR, "NBA_Context_Aware_Forecast.csv"
)

INJURY_PATH = os.path.join(
    BASE_DIR, "NBA_Injury_Report.csv"
)

ARCHIVE_DIR = os.path.join(
    BASE_DIR, "Injury_Archive"
)

os.makedirs(ARCHIVE_DIR, exist_ok=True)

print("Cartella progetto:", BASE_DIR)

# ============================================================
# 2. CONFIGURAZIONE
# ============================================================

TARGETS = {
    "PTS": "pts",
    "REB": "reb",
    "AST": "ast",
    "3PM": "fg3m",
    "PRA": "PRA"
}

MIN_IMPACT_SAMPLE = 5

HISTORY_DAYS = 365

ESPN_INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/nba/injuries"
)

session = requests.Session()

retry = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[
        429, 500, 502, 503, 504
    ],
    allowed_methods=["GET"]
)

session.mount(
    "https://",
    HTTPAdapter(max_retries=retry)
)

session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
})

# ============================================================
# 3. CARICAMENTO FORECAST
# ============================================================

if not os.path.isfile(FORECAST_PATH):

    raise FileNotFoundError(
        "Completa prima STEP 17."
    )

forecast = pd.read_csv(
    FORECAST_PATH,
    dtype={
        "game_id": str,
        "player_id": str
    }
)

forecast["date"] = pd.to_datetime(
    forecast["date"],
    utc=True,
    errors="coerce",
    format="mixed"
)

forecast = forecast[
    forecast["date"] > NOW
].drop_duplicates(
    ["game_id", "player_id"]
).copy()

if forecast.empty:

    raise ValueError(
        "Forecast scaduti. Riesegui STEP 17."
    )

print("Forecast futuri:", len(forecast))

# ============================================================
# 4. DOWNLOAD INJURY REPORT ESPN
# ============================================================

def normalize_status(value):

    value = str(value).strip().lower()

    mapping = {
        "out": "Out",
        "doubtful": "Doubtful",
        "questionable": "Questionable",
        "day-to-day": "Questionable",
        "day to day": "Questionable",
        "probable": "Probable",
        "available": "Available",
        "active": "Available",
        "game time decision": "Questionable",
        "gtd": "Questionable"
    }

    return mapping.get(value, "Unknown")


injury_rows = []

injury_feed_ok = False

print("\nDOWNLOAD INJURY REPORT ESPN...")

try:

    response = session.get(
        ESPN_INJURIES_URL,
        timeout=30
    )

    response.raise_for_status()

    payload = response.json()

    groups = payload.get("injuries")

    if not isinstance(groups, list):

        raise ValueError(
            "Struttura ESPN non riconosciuta."
        )

    for group in groups:

        team = group.get("displayName", "")

        team_info = group.get("team", {})

        if isinstance(team_info, dict):

            team = team_info.get(
                "displayName", team
            )

        items = group.get("injuries", [])

        if not isinstance(items, list):
            continue

        for injury in items:

            athlete = injury.get("athlete", {})

            if not isinstance(athlete, dict):
                continue

            player_id = athlete.get("id")

            if player_id is None:
                continue

            details = injury.get("details") or {}

            if not isinstance(details, dict):
                details = {}

            raw_status = injury.get("status", "")

            if isinstance(raw_status, dict):

                raw_status = raw_status.get(
                    "name",
                    raw_status.get("description", "")
                )

            status = normalize_status(
                raw_status
            )

            injury_rows.append({
                "player_id": str(player_id),
                "Giocatore": athlete.get(
                    "displayName",
                    athlete.get("fullName", "")
                ),
                "Squadra_ESPN": team,
                "Injury_Status": status,
                "Raw_Status": raw_status,
                "Injury_Type": details.get(
                    "type", ""
                ),
                "Injury_Location": details.get(
                    "location", ""
                ),
                "Return_Date": details.get(
                    "returnDate", ""
                ),
                "Comment": injury.get(
                    "longComment",
                    injury.get("shortComment", "")
                ),
                "Report_Date": injury.get(
                    "date", ""
                ),
                "Fetched_At": NOW.isoformat(),
                "Source": ESPN_INJURIES_URL
            })

    injury_feed_ok = True

except Exception as e:

    print("ATTENZIONE: injury feed non disponibile.")
    print(type(e).__name__, str(e))

injury_columns = [
    "player_id",
    "Giocatore",
    "Squadra_ESPN",
    "Injury_Status",
    "Raw_Status",
    "Injury_Type",
    "Injury_Location",
    "Return_Date",
    "Comment",
    "Report_Date",
    "Fetched_At",
    "Source"
]

injuries = pd.DataFrame(
    injury_rows,
    columns=injury_columns
)

if not injuries.empty:

    injuries["_report_time"] = pd.to_datetime(
        injuries["Report_Date"],
        utc=True,
        errors="coerce",
        format="mixed"
    )

    injuries = injuries.sort_values(
        "_report_time",
        na_position="first"
    ).drop_duplicates(
        "player_id",
        keep="last"
    ).drop(
        columns="_report_time"
    )

print("Feed disponibile:", injury_feed_ok)
print("Infortuni identificati:", len(injuries))

# ============================================================
# 5. CARICAMENTO DATABASE STORICO
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

for data in [box, games]:

    data["game_id"] = (
        data["game_id"].astype(str)
    )

    data["game_date"] = pd.to_datetime(
        data["game_date"],
        utc=True,
        errors="coerce",
        format="mixed"
    )

for col in ["player_id", "team_id"]:

    box[col] = box[col].astype(str)

numeric_columns = [
    "minutes",
    "pts",
    "reb",
    "ast",
    "fg3m",
    "starter",
    "did_not_play"
]

for col in numeric_columns:

    box[col] = pd.to_numeric(
        box[col],
        errors="coerce"
    )

games["completed"] = pd.to_numeric(
    games["completed"],
    errors="coerce"
)

completed_ids = set(
    games.loc[
        (games["completed"] == 1)
        & (games["game_date"] < NOW),
        "game_id"
    ]
)

box = box[
    box["game_id"].isin(completed_ids)
    & (box["game_date"] < NOW)
    & (
        box["game_date"]
        >= NOW - pd.Timedelta(days=HISTORY_DAYS)
    )
].drop_duplicates(
    ["game_id", "player_id"]
).sort_values(
    "game_date"
).copy()

box["PRA"] = (
    box["pts"] + box["reb"] + box["ast"]
)

played = box[
    (box["minutes"] > 0)
    & (box["did_not_play"] == 0)
].copy()

if played.empty:

    raise ValueError(
        "Nessuna presenza storica valida."
    )

print("\nPresenze storiche:", len(played))

# ============================================================
# 6. ANALISI ROTAZIONI E MINUTI
# ============================================================

rotation_rows = []

for player_id, group in played.groupby(
    "player_id"
):

    group = group.sort_values(
        "game_date"
    )

    last = group.iloc[-1]

    # Utilizziamo la squadra piu recente
    # per evitare di mescolare rotazioni
    # precedenti a un eventuale trasferimento.

    current_team = group[
        group["team_id"] == last["team_id"]
    ]

    last5 = current_team.tail(5)
    last10 = current_team.tail(10)
    last20 = current_team.tail(20)

    minutes5 = last5["minutes"].mean()
    minutes10 = last10["minutes"].mean()
    minutes20 = last20["minutes"].mean()

    starter_rate = last10["starter"].mean()

    if pd.isna(starter_rate):

        role = "Unknown"

    elif starter_rate >= 0.7:

        role = "Likely Starter"

    elif starter_rate >= 0.3:

        role = "Rotation / Starter"

    else:

        role = "Bench"

    rotation_rows.append({
        "player_id": player_id,
        "team_id": last["team_id"],
        "Last_Team": last["team_name"],
        "Last_Appearance": (
            last["game_date"].isoformat()
        ),
        "Minutes_Last5": minutes5,
        "Minutes_Last10": minutes10,
        "Minutes_Last20": minutes20,
        "Minutes_Trend": minutes5 - minutes20,
        "Starter_Rate_Last10": starter_rate,
        "Estimated_Role": role,
        "Rotation_Sample": len(last10),
        "PTS_Last10": last10["pts"].mean(),
        "REB_Last10": last10["reb"].mean(),
        "AST_Last10": last10["ast"].mean()
    })

rotations = pd.DataFrame(rotation_rows)

print("Profili rotazione:", len(rotations))

# ============================================================
# 7. INTEGRAZIONE INJURY REPORT E FORECAST
# ============================================================

injury_fields = [
    "player_id",
    "Injury_Status",
    "Raw_Status",
    "Injury_Type",
    "Injury_Location",
    "Return_Date",
    "Comment",
    "Report_Date",
    "Fetched_At"
]

context = forecast.merge(
    injuries[injury_fields],
    on="player_id",
    how="left",
    validate="many_to_one"
)

context = context.merge(
    rotations,
    on="player_id",
    how="left",
    validate="many_to_one"
)

context["Injury_Status"] = (
    context["Injury_Status"].fillna("Unknown")
)

context["Injury_Feed_OK"] = injury_feed_ok

context["Listed_In_Injury_Feed"] = (
    context["player_id"].isin(
        injuries["player_id"]
    )
)

def availability_flag(row):

    status = row["Injury_Status"]

    if status == "Out":
        return "EXCLUDED"

    if status in ["Doubtful", "Questionable"]:
        return "REVIEW_REQUIRED"

    if status == "Probable":
        return "PROBABLE"

    if status == "Available":
        return "AVAILABLE"

    if not row["Injury_Feed_OK"]:
        return "FEED_UNAVAILABLE"

    return "NOT_VERIFIED"


context["Availability_Flag"] = context.apply(
    availability_flag,
    axis=1
)

# ============================================================
# 8. ANALISI IMPATTO ASSENZE COMPAGNI
# ============================================================

# Confronto descrittivo delle prestazioni
# con compagno presente e con compagno
# esplicitamente registrato come DNP.
#
# Non assumiamo che un giocatore assente
# dal boxscore fosse necessariamente infortunato.

impact_rows = []

out_players = context[
    context["Injury_Status"] == "Out"
][[
    "game_id",
    "player_id",
    "Giocatore",
    "team_id"
]].dropna(
    subset=["team_id"]
).drop_duplicates()

for absent in out_players.itertuples(
    index=False
):

    absent_history = box[
        (box["player_id"] == absent.player_id)
        & (box["team_id"] == absent.team_id)
    ]

    games_with = set(
        absent_history.loc[
            (absent_history["minutes"] > 0)
            & (absent_history["did_not_play"] == 0),
            "game_id"
        ]
    )

    games_without = set(
        absent_history.loc[
            absent_history["did_not_play"] == 1,
            "game_id"
        ]
    )

    games_without -= games_with

    if min(
        len(games_with),
        len(games_without)
    ) < MIN_IMPACT_SAMPLE:

        continue

    teammates = context[
        (context["game_id"] == absent.game_id)
        & (context["team_id"] == absent.team_id)
        & (context["player_id"] != absent.player_id)
        & (context["Injury_Status"] != "Out")
    ]

    for player in teammates.itertuples(
        index=False
    ):

        hist = played[
            (played["player_id"] == player.player_id)
            & (played["team_id"] == absent.team_id)
        ]

        with_player = hist[
            hist["game_id"].isin(games_with)
        ]

        without_player = hist[
            hist["game_id"].isin(games_without)
        ]

        if min(
            len(with_player),
            len(without_player)
        ) < MIN_IMPACT_SAMPLE:

            continue

        for target, col in {
            **TARGETS,
            "MIN": "minutes"
        }.items():

            with_values = (
                with_player[col].dropna()
            )

            without_values = (
                without_player[col].dropna()
            )

            if min(
                len(with_values),
                len(without_values)
            ) < MIN_IMPACT_SAMPLE:

                continue

            mean_with = with_values.mean()

            mean_without = (
                without_values.mean()
            )

            delta = mean_without - mean_with

            delta_pct = (
                100 * delta / mean_with
                if mean_with > 0
                else np.nan
            )

            impact_rows.append({
                "game_id": absent.game_id,
                "player_id": player.player_id,
                "Giocatore": player.Giocatore,
                "Absent_Player_ID": absent.player_id,
                "Absent_Player": absent.Giocatore,
                "Target": target,
                "Games_With": len(with_values),
                "Games_Without": len(without_values),
                "Mean_With": mean_with,
                "Mean_Without": mean_without,
                "Historical_Delta": delta,
                "Historical_Delta_pct": delta_pct
            })

impact_columns = [
    "game_id",
    "player_id",
    "Giocatore",
    "Absent_Player_ID",
    "Absent_Player",
    "Target",
    "Games_With",
    "Games_Without",
    "Mean_With",
    "Mean_Without",
    "Historical_Delta",
    "Historical_Delta_pct"
]

impact = pd.DataFrame(
    impact_rows,
    columns=impact_columns
)

print("\nTEammate Impact Analysis")
print("Confronti disponibili:", len(impact))

# ============================================================
# 9. CONTEXT-AWARE FORECAST
# ============================================================

# Non modifichiamo arbitrariamente
# le previsioni ML in base alle assenze.
#
# Manteniamo i forecast originali e
# rimuoviamo quelli dei giocatori Out
# dalla lista delle previsioni utilizzabili.

for target in TARGETS:

    context[f"Context_{target}"] = (
        context[f"Forecast_{target}"]
    )

    context.loc[
        context["Injury_Status"] == "Out",
        f"Context_{target}"
    ] = np.nan

context["Context_Generated_At"] = (
    NOW.isoformat()
)

context["Context_Version"] = "3.2"

context["Forecast_Condition"] = (
    "Condizionato alla partecipazione"
)

context["Impact_Adjustment_Applied"] = False

context["Confirmed_Lineup"] = False

# Conteggio compagni indisponibili.

out_counts = context[
    context["Injury_Status"] == "Out"
].groupby(
    ["game_id", "Squadra"]
)["player_id"].nunique().rename(
    "Listed_Out_Teammates"
).reset_index()

context = context.merge(
    out_counts,
    on=["game_id", "Squadra"],
    how="left"
)

context["Listed_Out_Teammates"] = (
    context["Listed_Out_Teammates"]
    .fillna(0)
    .astype(int)
)

context.loc[
    context["Injury_Status"] == "Out",
    "Listed_Out_Teammates"
] -= 1

context["Listed_Out_Teammates"] = (
    context["Listed_Out_Teammates"].clip(lower=0)
)

if not injury_feed_ok:

    context["Listed_Out_Teammates"] = np.nan

# ============================================================
# 10. INTEGRAZIONE PROBABILITA STEP 18
# ============================================================

probability_context = pd.DataFrame()

if os.path.isfile(PROBABILITY_PATH):

    probabilities = pd.read_csv(
        PROBABILITY_PATH,
        dtype={
            "game_id": str,
            "player_id": str
        }
    )

    metadata = context[[
        "game_id",
        "player_id",
        "generated_at",
        "Injury_Status",
        "Availability_Flag",
        "Estimated_Role",
        "Listed_Out_Teammates"
    ]].rename(columns={
        "generated_at": "Current_Forecast_Generated_At"
    })

    probability_context = probabilities.merge(
        metadata,
        on=["game_id", "player_id"],
        how="inner",
        validate="many_to_one"
    )

    # Verifica che le probabilita derivino
    # dalla stessa versione del forecast.

    if "Forecast_Generated_At" in (
        probability_context.columns
    ):

        source_time = pd.to_datetime(
            probability_context[
                "Forecast_Generated_At"
            ],
            utc=True,
            errors="coerce",
            format="mixed"
        )

        current_time = pd.to_datetime(
            probability_context[
                "Current_Forecast_Generated_At"
            ],
            utc=True,
            errors="coerce",
            format="mixed"
        )

        probability_context = (
            probability_context[
                source_time.notna()
                & (source_time == current_time)
            ].copy()
        )

    else:

        print(
            "Probabilita senza timestamp sorgente: "
            "riesegui STEP 18 versione 3.1."
        )

        probability_context = pd.DataFrame()

    if not probability_context.empty:

        probability_context[
            "Conditional_Probability"
        ] = probability_context[
            "Probability_At_Least"
        ]

        probability_context.loc[
            probability_context["Injury_Status"] == "Out",
            "Conditional_Probability"
        ] = np.nan

        probability_context[
            "Conditional_Probability_pct"
        ] = (
            100
            * probability_context[
                "Conditional_Probability"
            ]
        )

        probability_context[
            "Participation_Probability"
        ] = np.nan

        probability_context[
            "Probability_Condition"
        ] = "Se il giocatore partecipa"

print(
    "\nProbabilita integrate:",
    len(probability_context)
)

# ============================================================
# 11. ANALISI QUINTETTI STIMATI
# ============================================================

lineup = context[
    context["Injury_Status"] != "Out"
].copy()

lineup = lineup.sort_values(
    [
        "game_id",
        "Squadra",
        "Starter_Rate_Last10",
        "Minutes_Last10"
    ],
    ascending=[
        True, True, False, False
    ]
)

lineup = lineup.groupby(
    ["game_id", "Squadra"],
    group_keys=False
).head(5).copy()

lineup["Lineup_Status"] = (
    "Proiezione storica non confermata"
)

lineup_columns = [
    "game_id",
    "date",
    "Squadra",
    "Avversario",
    "Giocatore",
    "Injury_Status",
    "Estimated_Role",
    "Starter_Rate_Last10",
    "Minutes_Last10",
    "Lineup_Status"
]

lineup = lineup[lineup_columns]

# ============================================================
# 12. CONTROLLO QUALITA'
# ============================================================

if context.duplicated(
    ["game_id", "player_id"]
).any():

    raise ValueError(
        "Forecast duplicati."
    )

out_mask = (
    context["Injury_Status"] == "Out"
)

for target in TARGETS:

    if not context.loc[
        out_mask,
        f"Context_{target}"
    ].isna().all():

        raise ValueError(
            "Giocatore Out non escluso."
        )

if not probability_context.empty:

    valid_p = probability_context[
        "Conditional_Probability"
    ].dropna()

    if not valid_p.between(0, 1).all():

        raise ValueError(
            "Probabilita non valide."
        )

FINISHED_AT = pd.Timestamp.now(tz="UTC")

context = context[
    context["date"] > FINISHED_AT
].copy()

valid_games = set(context["game_id"])

lineup = lineup[
    lineup["game_id"].isin(valid_games)
].copy()

impact = impact[
    impact["game_id"].isin(valid_games)
].copy()

if not probability_context.empty:

    probability_context = (
        probability_context[
            probability_context["game_id"]
            .isin(valid_games)
        ].copy()
    )

if context.empty:

    raise ValueError(
        "Nessun forecast futuro valido."
    )

print("\nCONTROLLO QUALITA' SUPERATO!")

# ============================================================
# 13. ARCHIVIAZIONE SQLITE
# ============================================================

with sqlite3.connect(
    DB_PATH,
    timeout=60
) as conn:

    conn.execute("""
        CREATE TABLE IF NOT EXISTS
        nba_injury_snapshots (
            player_id TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            player_name TEXT,
            team_name TEXT,
            injury_status TEXT,
            raw_status TEXT,
            injury_type TEXT,
            injury_location TEXT,
            report_date TEXT,
            source TEXT,
            PRIMARY KEY (
                player_id,
                fetched_at
            )
        )
    """)

    records = []

    for r in injuries.itertuples(
        index=False
    ):

        records.append((
            r.player_id,
            r.Fetched_At,
            r.Giocatore,
            r.Squadra_ESPN,
            r.Injury_Status,
            str(r.Raw_Status),
            str(r.Injury_Type),
            str(r.Injury_Location),
            str(r.Report_Date),
            r.Source
        ))

    conn.executemany("""
        INSERT OR IGNORE INTO
        nba_injury_snapshots
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS
        nba_injury_feed_runs (
            fetched_at TEXT PRIMARY KEY,
            feed_ok INTEGER,
            injury_count INTEGER,
            source TEXT
        )
    """)

    conn.execute("""
        INSERT OR IGNORE INTO
        nba_injury_feed_runs
        VALUES (?, ?, ?, ?)
    """, (
        NOW.isoformat(),
        int(injury_feed_ok),
        len(injuries),
        ESPN_INJURIES_URL
    ))

    conn.execute("""
        CREATE TABLE IF NOT EXISTS
        nba_availability_history (
            game_id TEXT NOT NULL,
            player_id TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            game_date TEXT,
            player_name TEXT,
            team_name TEXT,
            injury_status TEXT,
            availability_flag TEXT,
            feed_ok INTEGER,
            PRIMARY KEY (
                game_id,
                player_id,
                generated_at
            )
        )
    """)

    availability_records = []

    for r in context.itertuples(
        index=False
    ):

        availability_records.append((
            r.game_id,
            r.player_id,
            NOW.isoformat(),
            r.date.isoformat(),
            r.Giocatore,
            r.Squadra,
            r.Injury_Status,
            r.Availability_Flag,
            int(injury_feed_ok)
        ))

    conn.executemany("""
        INSERT OR IGNORE INTO
        nba_availability_history
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, availability_records)

    total_injury_history = conn.execute("""
        SELECT COUNT(*)
        FROM nba_injury_snapshots
    """).fetchone()[0]

# ============================================================
# 14. ESPORTAZIONE FILE
# ============================================================

stamp = NOW.strftime(
    "%Y%m%d_%H%M%S_%f"
)

export = context.copy()

export["date"] = export["date"].map(
    lambda x: x.isoformat()
)

lineup_export = lineup.copy()

lineup_export["date"] = (
    lineup_export["date"].map(
        lambda x: x.isoformat()
    )
)

injuries.to_csv(
    INJURY_PATH,
    index=False,
    encoding="utf-8-sig"
)

injuries.to_csv(
    os.path.join(
        ARCHIVE_DIR,
        f"NBA_Injuries_{stamp}.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

export.to_csv(
    CSV_PATH + ".tmp",
    index=False,
    encoding="utf-8-sig"
)

os.replace(
    CSV_PATH + ".tmp",
    CSV_PATH
)

export.to_csv(
    os.path.join(
        ARCHIVE_DIR,
        f"NBA_Context_{stamp}.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

impact.to_csv(
    os.path.join(
        BASE_DIR,
        "NBA_Teammate_Impact.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

lineup_export.to_csv(
    os.path.join(
        BASE_DIR,
        "NBA_Estimated_Lineups.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

probability_context.to_csv(
    os.path.join(
        BASE_DIR,
        "NBA_Context_Probabilities.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

status_report = context.groupby(
    [
        "Injury_Status",
        "Availability_Flag"
    ],
    as_index=False
).agg(
    Forecast_Count=("player_id", "size"),
    Unique_Players=("player_id", "nunique")
)

with pd.ExcelWriter(
    OUTPUT_PATH + ".tmp.xlsx",
    engine="openpyxl"
) as writer:

    sheets = {
        "Context Forecast": export,
        "Injury Report": injuries,
        "Estimated Lineups": lineup_export,
        "Rotation Analysis": rotations,
        "Teammate Impact": impact,
        "Context Probabilities": probability_context,
        "Availability Summary": status_report
    }

    for name, data in sheets.items():

        data.to_excel(
            writer,
            sheet_name=name,
            index=False
        )

        sheet = writer.sheets[name]

        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

os.replace(
    OUTPUT_PATH + ".tmp.xlsx",
    OUTPUT_PATH
)

# ============================================================
# 15. VISUALIZZAZIONE
# ============================================================

print("\nAVAILABILITY SUMMARY")

display(status_report)

fig = px.bar(
    status_report,
    x="Injury_Status",
    y="Forecast_Count",
    color="Injury_Status",
    title="NBA — Player Availability",
    template="plotly_dark",
    color_discrete_map={
        "Available": "#22c55e",
        "Probable": "#84cc16",
        "Questionable": "#eab308",
        "Doubtful": "#f97316",
        "Out": "#ef4444",
        "Unknown": "#64748b"
    }
)

fig.show()

print("\nGIOCATORI DA VERIFICARE")

review = context[
    context["Injury_Status"].isin([
        "Out",
        "Doubtful",
        "Questionable"
    ])
].sort_values(
    "Forecast_PTS",
    ascending=False
)

display(
    review[[
        "Giocatore",
        "Squadra",
        "Avversario",
        "Injury_Status",
        "Injury_Type",
        "Forecast_PTS",
        "Availability_Flag"
    ]].round(2)
)

# ============================================================
# 16. PLAYER CONTEXT EXPLORER
# ============================================================

player_options = context[[
    "player_id",
    "Giocatore"
]].drop_duplicates(
    "player_id"
).sort_values(
    "Giocatore"
)

players = [
    (
        f"{r.Giocatore} ({r.player_id})",
        r.player_id
    )
    for r in player_options.itertuples(
        index=False
    )
]

player_widget = widgets.Dropdown(
    options=players,
    description="Player:"
)

output = widgets.Output()

def show_player(change=None):

    with output:

        clear_output(wait=True)

        selected = context[
            context["player_id"]
            == player_widget.value
        ].sort_values("date")

        if selected.empty:
            return

        r = selected.iloc[0]

        print("=" * 60)
        print(r["Giocatore"].upper())
        print("=" * 60)

        print("Squadra:", r["Squadra"])
        print("Avversario:", r["Avversario"])

        print(
            "Injury Status:",
            r["Injury_Status"]
        )

        print(
            "Availability:",
            r["Availability_Flag"]
        )

        print(
            "Injury Type:",
            r["Injury_Type"]
        )

        print(
            "Ruolo stimato:",
            r["Estimated_Role"]
        )

        print(
            "Minuti ultime 10:",
            round(r["Minutes_Last10"], 2)
        )

        print(
            "Compagni Out:",
            r["Listed_Out_Teammates"]
        )

        print("\nFORECAST")

        rows = []

        for target in TARGETS:

            rows.append({
                "Target": target,
                "Forecast_ML": r[
                    f"Forecast_{target}"
                ],
                "Context_Forecast": r[
                    f"Context_{target}"
                ]
            })

        display(
            pd.DataFrame(rows).round(2)
        )

        player_impact = impact[
            (impact["player_id"] == r["player_id"])
            & (impact["game_id"] == r["game_id"])
        ]

        if not player_impact.empty:

            print("\nIMPATTO STORICO COMPAGNI ASSENTI")

            display(
                player_impact[[
                    "Absent_Player",
                    "Target",
                    "Games_With",
                    "Games_Without",
                    "Mean_With",
                    "Mean_Without",
                    "Historical_Delta"
                ]].round(2)
            )

            fig = px.bar(
                player_impact,
                x="Target",
                y="Historical_Delta",
                color="Absent_Player",
                barmode="group",
                title="Historical Teammate Impact",
                template="plotly_dark"
            )

            fig.add_hline(y=0)
            fig.show()

        else:

            print(
                "\nCampione assenze insufficiente "
                "o nessun compagno Out identificato."
            )

        if not probability_context.empty:

            selected_probs = probability_context[
                (
                    probability_context["player_id"]
                    == r["player_id"]
                )
                & (
                    probability_context["game_id"]
                    == r["game_id"]
                )
            ]

            if not selected_probs.empty:

                print("\nPROBABILITA CONDIZIONATE")

                display(
                    selected_probs[[
                        "Target",
                        "Threshold",
                        "Conditional_Probability_pct"
                    ]].round(2)
                )

player_widget.observe(
    show_player,
    names="value"
)

display(
    player_widget,
    output
)

show_player()

# ============================================================
# 17. RIEPILOGO FINALE
# ============================================================

print("\n" + "=" * 60)
print("STEP 19 COMPLETATO!")
print("INJURY & LINEUP INTELLIGENCE V3.2")
print("=" * 60)

print("Feed ESPN disponibile:", injury_feed_ok)

print(
    "Partite future:",
    context["game_id"].nunique()
)

print(
    "Giocatori:",
    context["player_id"].nunique()
)

print(
    "Forecast esclusi per Out:",
    int(
        (context["Injury_Status"] == "Out").sum()
    )
)

print(
    "Forecast da verificare:",
    int(
        context["Injury_Status"].isin([
            "Questionable", "Doubtful"
        ]).sum()
    )
)

print(
    "Disponibilita sconosciuta:",
    int(
        (context["Injury_Status"] == "Unknown").sum()
    )
)

print(
    "Quintetti stimati - righe:",
    len(lineup)
)

print(
    "Analisi impatto:",
    len(impact)
)

print(
    "Injury snapshot storici:",
    total_injury_history
)

print("\nFILE SALVATI")

print(OUTPUT_PATH)
print(CSV_PATH)
print(INJURY_PATH)

if not injury_feed_ok:

    print(
        "\nATTENZIONE: modulo injury degradato. "
        "Verificare la disponibilita del feed ESPN."
    )

print("\nPRONTO PER STEP 20!")
