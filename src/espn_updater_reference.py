# Source: notebook cell 29 — ESPN data pipeline
# NOTE: Colab-oriented notebook code; requires runtime/Drive configuration.
# Not verified as standalone executable.


# ============================================================
# NBA DATA SCIENCE ANALYTICS
# STEP 12 — AUTOMATED DATA PIPELINE
# ESPN -> SQLITE -> PLAYER BOXSCORES
# ============================================================

import os
import time
import sqlite3
import requests
import pandas as pd

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from tqdm.auto import tqdm
from google.colab import drive

# ============================================================
# 1. CONFIGURAZIONE
# ============================================================

DB_PATH = (
    "/content/drive/MyDrive/"
    "NBA_Data_Science/nba_database.db"
)

GIORNI_PASSATI = 7
GIORNI_FUTURI = 3

PAUSA_RICHIESTE = 0.25

BASE = (
    "https://site.api.espn.com/apis/"
    "site/v2/sports/basketball/nba"
)

if not os.path.isfile(DB_PATH):
    drive.mount("/content/drive")

if not os.path.isfile(DB_PATH):
    raise FileNotFoundError(
        "Database NBA non trovato su Google Drive."
    )

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
})

conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA busy_timeout = 10000")

# ============================================================
# 2. CONTROLLO SCHEMA SQLITE
# ============================================================

def colonne_tabella(nome):
    return {
        row[1]
        for row in conn.execute(
            f"PRAGMA table_info({nome})"
        ).fetchall()
    }

def aggiungi_colonna(tabella, nome, tipo):
    if nome not in colonne_tabella(tabella):
        conn.execute(
            f"ALTER TABLE {tabella} "
            f"ADD COLUMN {nome} {tipo}"
        )

conn.execute("""
CREATE TABLE IF NOT EXISTS espn_games (
    game_id TEXT PRIMARY KEY,
    season TEXT,
    game_date TEXT,
    home_team TEXT,
    away_team TEXT,
    home_score REAL,
    away_score REAL,
    completed INTEGER,
    status TEXT
)
""")

for nome, tipo in {
    "home_id": "TEXT",
    "away_id": "TEXT",
    "season": "TEXT",
    "game_date": "TEXT",
    "home_team": "TEXT",
    "away_team": "TEXT",
    "home_score": "REAL",
    "away_score": "REAL",
    "completed": "INTEGER",
    "status": "TEXT"
}.items():
    aggiungi_colonna("espn_games", nome, tipo)

conn.execute("""
CREATE TABLE IF NOT EXISTS espn_player_boxscores (
    game_id TEXT NOT NULL,
    season TEXT,
    game_date TEXT,
    team_id TEXT NOT NULL,
    team_name TEXT,
    player_id TEXT NOT NULL,
    player_name TEXT,
    starter INTEGER,
    did_not_play INTEGER,
    minutes REAL,
    pts REAL,
    reb REAL,
    ast REAL,
    stl REAL,
    blk REAL,
    tov REAL,
    oreb REAL,
    dreb REAL,
    pf REAL,
    fgm REAL,
    fga REAL,
    fg3m REAL,
    fg3a REAL,
    ftm REAL,
    fta REAL,
    plus_minus REAL,
    PRIMARY KEY (game_id, team_id, player_id)
)
""")

conn.execute("""
CREATE TABLE IF NOT EXISTS espn_boxscore_log (
    game_id TEXT PRIMARY KEY,
    downloaded_at TEXT,
    players_found INTEGER
)
""")

conn.commit()

print("Schema database verificato.")

# ============================================================
# 3. FUNZIONI DI SUPPORTO
# ============================================================

def fetch_json(url, params=None, retries=3):

    for tentativo in range(retries):
        try:
            r = session.get(
                url,
                params=params,
                timeout=30
            )
            r.raise_for_status()
            return r.json()

        except requests.RequestException as e:
            print(
                f"Richiesta fallita "
                f"({tentativo + 1}/{retries}): {e}"
            )
            time.sleep(2 * (tentativo + 1))

    return None


def numero(valore):

    try:
        if valore is None:
            return None

        s = str(valore).strip()

        if s in ["", "--", "DNP", "None"]:
            return None

        return float(s.replace("+", ""))

    except (ValueError, TypeError):
        return None


def minuti(valore):

    if valore is None:
        return None

    s = str(valore).strip()

    if ":" in s:
        try:
            mm, ss = s.split(":", 1)
            return float(mm) + float(ss) / 60
        except ValueError:
            return None

    return numero(s)


def frazione(valore):

    if valore is None:
        return None, None

    s = str(valore)

    if "-" not in s:
        return None, None

    try:
        a, b = s.split("-", 1)
        return float(a), float(b)
    except ValueError:
        return None, None


def stagione_nba(data_iso):

    data = pd.to_datetime(
        data_iso,
        utc=True
    ).tz_convert("America/New_York")

    anno = data.year

    if data.month < 7:
        anno -= 1

    return f"{anno}-{str(anno + 1)[-2:]}"


# ============================================================
# 4. AGGIORNAMENTO CALENDARIO
# ============================================================

oggi = datetime.now(
    ZoneInfo("America/New_York")
).date()

giorni = [
    oggi + timedelta(days=i)
    for i in range(
        -GIORNI_PASSATI,
        GIORNI_FUTURI + 1
    )
]

eventi = {}

print("\nAggiornamento calendario ESPN...")

for giorno in tqdm(giorni):

    dati = fetch_json(
        BASE + "/scoreboard",
        params={
            "dates": giorno.strftime("%Y%m%d"),
            "limit": 100
        }
    )

    if dati is None:
        continue

    for evento in dati.get("events", []):
        eventi[str(evento["id"])] = evento

    time.sleep(PAUSA_RICHIESTE)

print("Eventi ESPN individuati:", len(eventi))

# ============================================================
# 5. SALVATAGGIO PARTITE
# ============================================================

aggiornate = 0
completate = []

for evento in eventi.values():

    competizioni = evento.get(
        "competitions", []
    )

    if not competizioni:
        continue

    competitors = competizioni[0].get(
        "competitors", []
    )

    home = next(
        (
            x for x in competitors
            if x.get("homeAway") == "home"
        ),
        None
    )

    away = next(
        (
            x for x in competitors
            if x.get("homeAway") == "away"
        ),
        None
    )

    if home is None or away is None:
        continue

    game_id = str(evento["id"])
    data = evento.get("date")

    if not data:
        continue

    status_obj = evento.get(
        "status", {}
    ).get("type", {})

    status = status_obj.get(
        "name", "UNKNOWN"
    )

    completed = int(
        status_obj.get("completed", False)
    )

    home_score = numero(
        home.get("score")
    )

    away_score = numero(
        away.get("score")
    )

    season = stagione_nba(data)

    # Non sovrascrive una gara conclusa con
    # una versione successiva incompleta.
    precedente = conn.execute(
        """
        SELECT completed
        FROM espn_games
        WHERE game_id = ?
        """,
        (game_id,)
    ).fetchone()

    if precedente and precedente[0] == 1:
        if not completed:
            continue

    conn.execute("""
        INSERT INTO espn_games (
            game_id, season, game_date,
            home_team, away_team,
            home_id, away_id,
            home_score, away_score,
            completed, status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(game_id) DO UPDATE SET
            season = excluded.season,
            game_date = excluded.game_date,
            home_team = excluded.home_team,
            away_team = excluded.away_team,
            home_id = excluded.home_id,
            away_id = excluded.away_id,
            home_score = excluded.home_score,
            away_score = excluded.away_score,
            completed = excluded.completed,
            status = excluded.status
    """, (
        game_id,
        season,
        data,
        home["team"].get("displayName"),
        away["team"].get("displayName"),
        str(home["team"]["id"]),
        str(away["team"]["id"]),
        home_score,
        away_score,
        completed,
        status
    ))

    aggiornate += 1

    if completed:
        completate.append(game_id)

conn.commit()

print("Partite inserite/aggiornate:", aggiornate)
print("Partite concluse nel periodo:", len(completate))

# ============================================================
# 6. DOWNLOAD INCREMENTALE BOX SCORE
# ============================================================

gia_scaricate = {
    row[0]
    for row in conn.execute("""
        SELECT game_id
        FROM espn_boxscore_log
        WHERE players_found > 0
    """)
}

da_scaricare = [
    gid for gid in completate
    if gid not in gia_scaricate
]

print("\nNuovi box score da scaricare:", len(da_scaricare))

STAT_MAP = {
    "MIN": "minutes",
    "PTS": "pts",
    "REB": "reb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "TO": "tov",
    "TOV": "tov",
    "OREB": "oreb",
    "DREB": "dreb",
    "PF": "pf",
    "+/-": "plus_minus",
    "FG": "fg",
    "3PT": "fg3",
    "FT": "ft"
}

BOX_COLS = [
    "game_id", "season", "game_date",
    "team_id", "team_name",
    "player_id", "player_name",
    "starter", "did_not_play",
    "minutes", "pts", "reb", "ast",
    "stl", "blk", "tov", "oreb",
    "dreb", "pf", "fgm", "fga",
    "fg3m", "fg3a", "ftm", "fta",
    "plus_minus"
]


def estrai_boxscore(game_id):

    dati = fetch_json(
        BASE + "/summary",
        params={"event": game_id}
    )

    if not dati:
        return []

    info = conn.execute("""
        SELECT season, game_date
        FROM espn_games
        WHERE game_id = ?
    """, (game_id,)).fetchone()

    if not info:
        return []

    season, game_date = info

    righe = []

    for squadra in dati.get(
        "boxscore", {}
    ).get("players", []):

        team = squadra.get("team", {})

        team_id = str(team.get("id", ""))
        team_name = team.get(
            "displayName", ""
        )

        for gruppo in squadra.get(
            "statistics", []
        ):

            labels = gruppo.get(
                "names", []
            )

            if not labels:
                labels = gruppo.get(
                    "labels", []
                )

            for atleta in gruppo.get(
                "athletes", []
            ):

                player = atleta.get(
                    "athlete", {}
                )

                player_id = str(
                    player.get("id", "")
                )

                if not player_id or not team_id:
                    continue

                valori = atleta.get(
                    "stats", []
                )

                stats = {
                    STAT_MAP.get(
                        str(label).upper(), label
                    ): valore
                    for label, valore
                    in zip(labels, valori)
                }

                fgm, fga = frazione(
                    stats.get("fg")
                )

                fg3m, fg3a = frazione(
                    stats.get("fg3")
                )

                ftm, fta = frazione(
                    stats.get("ft")
                )

                riga = {
                    "game_id": game_id,
                    "season": season,
                    "game_date": game_date,
                    "team_id": team_id,
                    "team_name": team_name,
                    "player_id": player_id,
                    "player_name": player.get(
                        "displayName", ""
                    ),
                    "starter": int(
                        bool(atleta.get("starter"))
                    ),
                    "did_not_play": int(
                        bool(
                            atleta.get(
                                "didNotPlay", False
                            )
                        )
                    ),
                    "minutes": minuti(
                        stats.get("minutes")
                    ),
                    "pts": numero(stats.get("pts")),
                    "reb": numero(stats.get("reb")),
                    "ast": numero(stats.get("ast")),
                    "stl": numero(stats.get("stl")),
                    "blk": numero(stats.get("blk")),
                    "tov": numero(stats.get("tov")),
                    "oreb": numero(stats.get("oreb")),
                    "dreb": numero(stats.get("dreb")),
                    "pf": numero(stats.get("pf")),
                    "fgm": fgm,
                    "fga": fga,
                    "fg3m": fg3m,
                    "fg3a": fg3a,
                    "ftm": ftm,
                    "fta": fta,
                    "plus_minus": numero(
                        stats.get("plus_minus")
                    )
                }

                righe.append(riga)

    return righe


scaricate = 0
fallite = []

for game_id in tqdm(da_scaricare):

    try:
        righe = estrai_boxscore(game_id)

        # Evitiamo di registrare come riuscito
        # un download senza statistiche valide.
        valide = [
            r for r in righe
            if r["minutes"] is not None
            and r["pts"] is not None
        ]

        if not valide:
            fallite.append(game_id)
            continue

        placeholders = ",".join(
            ["?"] * len(BOX_COLS)
        )

        sql = (
            "INSERT OR REPLACE INTO "
            "espn_player_boxscores ("
            + ",".join(BOX_COLS)
            + ") VALUES ("
            + placeholders
            + ")"
        )

        with conn:

            conn.executemany(
                sql,
                [
                    tuple(r[c] for c in BOX_COLS)
                    for r in righe
                ]
            )

            conn.execute("""
                INSERT OR REPLACE
                INTO espn_boxscore_log (
                    game_id,
                    downloaded_at,
                    players_found
                )
                VALUES (?, ?, ?)
            """, (
                game_id,
                datetime.now(
                    ZoneInfo("UTC")
                ).isoformat(),
                len(righe)
            ))

        scaricate += 1

    except Exception as errore:
        fallite.append(game_id)
        print(
            f"Errore partita {game_id}: {errore}"
        )

    time.sleep(PAUSA_RICHIESTE)

# ============================================================
# 7. CONTROLLO QUALITA'
# ============================================================

tot_games = conn.execute("""
    SELECT COUNT(*) FROM espn_games
""").fetchone()[0]

tot_completed = conn.execute("""
    SELECT COUNT(*)
    FROM espn_games
    WHERE completed = 1
""").fetchone()[0]

tot_box = conn.execute("""
    SELECT COUNT(*)
    FROM espn_player_boxscores
""").fetchone()[0]

tot_players = conn.execute("""
    SELECT COUNT(DISTINCT player_id)
    FROM espn_player_boxscores
""").fetchone()[0]

tot_logged = conn.execute("""
    SELECT COUNT(*)
    FROM espn_boxscore_log
    WHERE players_found > 0
""").fetchone()[0]

# ============================================================
# 8. SALVATAGGIO REPORT
# ============================================================

REPORT_PATH = (
    "/content/drive/MyDrive/"
    "NBA_Data_Science/"
    "NBA_Pipeline_Status.csv"
)

report = pd.DataFrame([{
    "Aggiornamento UTC": datetime.now(
        ZoneInfo("UTC")
    ).isoformat(),
    "Partite database": tot_games,
    "Partite concluse": tot_completed,
    "Player box score": tot_box,
    "Giocatori unici": tot_players,
    "Partite con box score": tot_logged,
    "Box score nuovi": scaricate,
    "Download falliti": len(fallite)
}])

report.to_csv(
    REPORT_PATH,
    index=False
)

conn.close()
session.close()

print("\n======================================")
print("STEP 12 COMPLETATO")
print("======================================")

print("Partite nel database:", tot_games)
print("Partite concluse:", tot_completed)
print("Player box score:", tot_box)
print("Giocatori unici:", tot_players)
print("Partite con box score:", tot_logged)
print("Nuovi box score:", scaricate)
print("Download falliti:", len(fallite))

if fallite:
    print("Partite da riprovare:", fallite)

print("\nDatabase:", DB_PATH)
print("Report:", REPORT_PATH)
