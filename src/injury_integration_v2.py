# Source: notebook cell 84 — Injury context merging
# NOTE: Colab-oriented notebook code; requires runtime/Drive configuration.
# Not verified as standalone executable.


# ============================================================
# STEP 26B — INJURY INTEGRATION & DASHBOARD RESTART
# ============================================================

from google.colab import drive
from pathlib import Path
from datetime import datetime
import pandas as pd
import shutil

drive.mount("/content/drive")

ROOT = Path("/content/drive/MyDrive/NBA_Data_Science")

BACKUP = ROOT / "Backup_Injury_V2" / datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

BACKUP.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. CARICAMENTO NUOVI DATI
# ============================================================

keys = ["game_id", "player_id"]

availability = pd.read_csv(
    ROOT / "NBA_Availability_V2.csv",
    dtype=str
)

injuries = pd.read_csv(
    ROOT / "NBA_Injury_Sources_V2.csv",
    dtype=str
)

availability = availability.drop_duplicates(
    keys, keep="last"
)

columns = [
    "Injury_Status",
    "Availability_Flag",
    "Injury_Source",
    "Availability_Note"
]

def backup_file(path):
    if path.exists():
        shutil.copy2(path, BACKUP / path.name)

def save_csv(df, path):
    backup_file(path)
    temp = path.with_suffix(".tmp")
    df.to_csv(temp, index=False)
    temp.replace(path)

def integrate_availability(df):

    df = df.drop(
        columns=columns,
        errors="ignore"
    )

    df = df.merge(
        availability[keys + columns],
        on=keys,
        how="left",
        validate="many_to_one"
    )

    df["Injury_Status"] = (
        df["Injury_Status"].fillna("Unknown")
    )

    df["Availability_Flag"] = (
        df["Availability_Flag"].fillna("UNVERIFIED")
    )

    df["Roster_Status"] = df["Availability_Flag"]

    return df

# ============================================================
# 2. AGGIORNAMENTO INJURY REPORT
# ============================================================

save_csv(
    injuries,
    ROOT / "NBA_Injury_Report.csv"
)

# ============================================================
# 3. AGGIORNAMENTO FORECAST CONTESTUALE
# ============================================================

raw = pd.read_csv(
    ROOT / "NBA_Forecast_Engine_V3.csv",
    dtype={k: str for k in keys}
)

context_path = ROOT / "NBA_Context_Aware_Forecast.csv"

context = raw.copy()

if context_path.exists():

    previous = pd.read_csv(
        context_path,
        dtype={k: str for k in keys}
    )

    join_keys = keys + ["generated_at"]

    if all(k in previous.columns for k in join_keys):

        extra = [
            c for c in previous.columns
            if c not in raw.columns
            and c not in columns
        ]

        context = raw.merge(
            previous[join_keys + extra].drop_duplicates(
                join_keys, keep="last"
            ),
            on=join_keys,
            how="left"
        )

context = integrate_availability(context)

save_csv(context, context_path)

# ============================================================
# 4. AGGIORNAMENTO PROBABILITA
# ============================================================

prob = pd.read_csv(
    ROOT / "NBA_Probability_Engine_V3.csv",
    dtype={k: str for k in keys}
)

prob = integrate_availability(prob)

prob = prob[
    prob["Injury_Status"] != "Out"
].copy()

save_csv(
    prob,
    ROOT / "NBA_Context_Probabilities.csv"
)

# ============================================================
# 5. AGGIORNAMENTO FORMAZIONI
# ============================================================

lineup_path = ROOT / "NBA_Estimated_Lineups.csv"

if lineup_path.exists():

    lineup = pd.read_csv(
        lineup_path,
        dtype={k: str for k in keys}
    )

    if all(k in lineup.columns for k in keys):

        lineup = integrate_availability(lineup)

        lineup = lineup[
            lineup["Injury_Status"] != "Out"
        ].copy()

        save_csv(lineup, lineup_path)

# ============================================================
# 6. CONTROLLO FINALE
# ============================================================

print("=" * 60)
print("INJURY INTELLIGENCE V2 INTEGRATO")
print("=" * 60)

print("\nStati giocatori:")

display(
    context["Injury_Status"].value_counts()
)

print("\nGiocatori OUT:")

display(
    context.loc[
        context["Injury_Status"] == "Out",
        ["Giocatore", "Squadra", "Injury_Source"]
    ]
)

print("\nCONTROLLO LAMELO BALL:")

display(
    context.loc[
        context["Giocatore"].str.contains(
            "LaMelo Ball",
            case=False,
            na=False
        ),
        [
            "Giocatore",
            "Injury_Status",
            "Availability_Flag",
            "Injury_Source",
            "Forecast_PTS"
        ]
    ]
)

print("\nBackup:", BACKUP)
print("\nSTEP 26B COMPLETATO")
