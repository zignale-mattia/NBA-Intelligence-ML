# Source: notebook cell 83 — Injury V2 source
# NOTE: Colab-oriented notebook code; requires runtime/Drive configuration.
# Not verified as standalone executable.


# ============================================================
# STEP 26A — NBA INJURY INTELLIGENCE V2
# ============================================================

# Colab setup: %pip install pymupdf beautifulsoup4

import requests
import pandas as pd
import fitz
import re
import unicodedata

from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ROOT26 = Path(
    "/content/drive/MyDrive/NBA_Data_Science"
)

FORECAST26 = ROOT26 / "NBA_Forecast_Engine_V3.csv"

NBA_URL = "https://www.nba.com/players/injury-report"

ESPN_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/nba/injuries"
)

session26 = requests.Session()

session26.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*"
})

now26 = datetime.now(timezone.utc)

today_et = now26.astimezone(
    ZoneInfo("America/New_York")
).date()

VALID_STATUS = [
    "Out",
    "Doubtful",
    "Questionable",
    "Probable",
    "Available"
]

# ============================================================
# 1. NORMALIZZAZIONE
# ============================================================

def normalize26(value):

    value = unicodedata.normalize(
        "NFKD", str(value)
    )

    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )

    return re.sub(
        r"[^a-z0-9]", "", value.lower()
    )


def status26(value):

    value = str(value).strip().lower()

    mapping = {
        "out": "Out",
        "doubtful": "Doubtful",
        "questionable": "Questionable",
        "probable": "Probable",
        "available": "Available",
        "day-to-day": "Questionable",
        "game time decision": "Questionable"
    }

    return mapping.get(value, "Unknown")


# ============================================================
# 2. CARICAMENTO FORECAST
# ============================================================

forecast26 = pd.read_csv(
    FORECAST26,
    dtype={
        "game_id": str,
        "player_id": str
    }
)

forecast26["date"] = pd.to_datetime(
    forecast26["date"],
    utc=True,
    errors="coerce"
)

forecast26 = forecast26[
    forecast26["date"] > pd.Timestamp(now26)
].copy()

if forecast26.empty:
    raise RuntimeError(
        "Nessun forecast futuro. Aggiornare STEP 17."
    )

forecast26 = forecast26.sort_values(
    "generated_at"
).drop_duplicates(
    ["game_id", "player_id"],
    keep="last"
)

players26 = forecast26[
    ["player_id", "Giocatore", "Squadra"]
].drop_duplicates()

print("Giocatori da verificare:", len(players26))

# ============================================================
# 3. ESPN INJURY REPORT
# ============================================================

espn_records26 = []
espn_ok26 = False

try:

    response = session26.get(
        ESPN_URL,
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

        team = group.get(
            "displayName", ""
        )

        for item in group.get("injuries", []):

            athlete = item.get("athlete", {})

            details = item.get("details") or {}

            raw_status = item.get("status", "")

            if isinstance(raw_status, dict):
                raw_status = raw_status.get(
                    "name", ""
                )

            espn_records26.append({
                "player_id": str(
                    athlete.get("id", "")
                ),
                "Giocatore": athlete.get(
                    "displayName", ""
                ),
                "Squadra_ESPN": team,
                "Injury_Status": status26(
                    raw_status
                ),
                "Raw_Status": str(raw_status),
                "Injury_Type": details.get(
                    "type", ""
                ),
                "Injury_Location": details.get(
                    "location", ""
                ),
                "Return_Date": details.get(
                    "returnDate", ""
                ),
                "Comment": item.get(
                    "longComment",
                    item.get("shortComment", "")
                ),
                "Report_Date": item.get(
                    "date", ""
                ),
                "Fetched_At": now26.isoformat(),
                "Source": "ESPN"
            })

    espn_ok26 = bool(espn_records26)

    print(
        "ESPN:",
        len(espn_records26),
        "segnalazioni"
    )

except Exception as e:
    print("ESPN non disponibile:", e)


# ============================================================
# 4. REPORT UFFICIALE NBA
# ============================================================

nba_records26 = []
nba_ok26 = False
official_pdf26 = None
report_time26 = None

try:

    page = session26.get(
        NBA_URL,
        timeout=30
    )

    page.raise_for_status()

    soup = BeautifulSoup(
        page.text,
        "html.parser"
    )

    links = []

    for a in soup.find_all("a", href=True):

        url = urljoin(
            NBA_URL, a["href"]
        )

        match = re.search(
            r"Injury-Report_"
            r"(\d{4}-\d{2}-\d{2})_"
            r"(\d{2})_(\d{2})(AM|PM)",
            url,
            re.I
        )

        if not match:
            continue

        stamp = datetime.strptime(
            f"{match[1]} "
            f"{match[2]}:{match[3]} "
            f"{match[4].upper()}",
            "%Y-%m-%d %I:%M %p"
        ).replace(
            tzinfo=ZoneInfo("America/New_York")
        )

        if (
            stamp.date() == today_et
            and stamp <= now26
            and "ak-static.cms.nba.com" in url
        ):

            links.append((stamp, url))

    if not links:
        raise RuntimeError(
            "Nessun PDF ufficiale odierno trovato."
        )

    report_time26, official_pdf26 = max(
        links,
        key=lambda x: x[0]
    )

    response = session26.get(
        official_pdf26,
        timeout=40
    )

    response.raise_for_status()

    document = fitz.open(
        stream=response.content,
        filetype="pdf"
    )

    text26 = "\n".join(
        page.get_text(sort=True)
        for page in document
    )

    document.close()

    (
        ROOT26 / "NBA_Official_Injury_Report.txt"
    ).write_text(
        text26,
        encoding="utf-8"
    )

    print("Report ufficiale NBA:")
    print(official_pdf26)

    # Riconoscimento dei nomi nel formato
    # Cognome, Nome utilizzato dal report NBA.

    for _, player in players26.iterrows():

        name = player["Giocatore"]

        parts = name.split()

        if len(parts) < 2:
            continue

        variants = []

        for split in range(1, len(parts)):

            first = " ".join(parts[:split])
            last = " ".join(parts[split:])

            variants.append(
                last + ", " + first
            )

        variants.append(name)

        for variant in variants:

            pattern = (
                re.escape(variant)
                + r"\s+"
                + r"(Out|Doubtful|Questionable|"
                  r"Probable|Available)\b"
            )

            match = re.search(
                pattern,
                text26,
                re.I
            )

            if not match:
                continue

            nba_records26.append({
                "player_id": player["player_id"],
                "Giocatore": name,
                "Squadra_ESPN": player["Squadra"],
                "Injury_Status": status26(
                    match.group(1)
                ),
                "Raw_Status": match.group(1),
                "Injury_Type": "",
                "Injury_Location": "",
                "Return_Date": "",
                "Comment": (
                    "Da verificare nel PDF ufficiale"
                ),
                "Report_Date": (
                    report_time26.isoformat()
                ),
                "Fetched_At": now26.isoformat(),
                "Source": "NBA Official"
            })

            break

    nba_ok26 = bool(nba_records26)

    print(
        "NBA Official:",
        len(nba_records26),
        "giocatori riconosciuti"
    )

except Exception as e:
    print("Report NBA non disponibile:", e)


# ============================================================
# 5. UNIFICAZIONE FONTI
# ============================================================

columns26 = [
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

injuries26 = pd.DataFrame(
    espn_records26 + nba_records26,
    columns=columns26
)

injuries26 = injuries26[
    injuries26["player_id"].astype(str) != ""
].copy()

injuries26 = injuries26[
    injuries26["Injury_Status"] != "Unknown"
]

# Non sovrascrivere i dati precedenti
# se entrambi i recuperi falliscono.

if injuries26.empty:

    raise RuntimeError(
        "Nessuna segnalazione valida recuperata. "
        "Controllare le fonti e il parser."
    )

# Conservazione di entrambe le fonti

injuries26.to_csv(
    ROOT26 / "NBA_Injury_Sources_V2.csv",
    index=False
)

# ============================================================
# 6. DISPONIBILITA PER SINGOLA PARTITA
# ============================================================

availability26 = []

for _, r in forecast26.iterrows():

    pid = str(r["player_id"])

    player_rows = injuries26[
        injuries26["player_id"].astype(str) == pid
    ]

    team26 = normalize26(r["Squadra"])

    official = player_rows[
        (player_rows["Source"] == "NBA Official")
        & (
            player_rows["Squadra_ESPN"]
            .map(normalize26) == team26
        )
    ]

    espn = player_rows[
        player_rows["Source"] == "ESPN"
    ]

    game_day = r["date"].tz_convert(
        "America/New_York"
    ).date()

    status = "Unknown"
    source = "Not verified"
    note = "Disponibilità non confermata"

    if game_day == today_et:

        candidates = (
            official
            if not official.empty
            else espn
        )

        if not candidates.empty:

            statuses = set(
                candidates["Injury_Status"]
            )

            if len(statuses) == 1:

                status = next(iter(statuses))
                source = candidates.iloc[-1]["Source"]

                note = "Segnalazione recuperata"

                if (
                    not official.empty
                    and not espn.empty
                    and set(espn["Injury_Status"])
                    != statuses
                ):

                    status = "Unknown"
                    source = "NBA / ESPN conflict"
                    note = (
                        "Fonti discordanti: "
                        "verifica manuale necessaria"
                    )

            else:
                note = "Segnalazioni discordanti"

    else:
        note = "Report non verificato per questa data"

    flag = {
        "Out": "EXCLUDE",
        "Doubtful": "HIGH_RISK",
        "Questionable": "UNCERTAIN",
        "Probable": "PROBABLE",
        "Available": "AVAILABLE",
        "Unknown": "UNVERIFIED"
    }[status]

    availability26.append({
        "game_id": r["game_id"],
        "player_id": pid,
        "date": r["date"],
        "Giocatore": r["Giocatore"],
        "Squadra": r["Squadra"],
        "Avversario": r["Avversario"],
        "Injury_Status": status,
        "Availability_Flag": flag,
        "Injury_Source": source,
        "Availability_Note": note,
        "Fetched_At": now26.isoformat()
    })

availability26 = pd.DataFrame(
    availability26
)

availability26.to_csv(
    ROOT26 / "NBA_Availability_V2.csv",
    index=False
)

# ============================================================
# 7. FORECAST OPERATIVO
# ============================================================

context26 = forecast26.drop(
    columns=[
        "Injury_Status",
        "Availability_Flag"
    ],
    errors="ignore"
).merge(
    availability26[
        [
            "game_id",
            "player_id",
            "Injury_Status",
            "Availability_Flag",
            "Injury_Source",
            "Availability_Note"
        ]
    ],
    on=["game_id", "player_id"],
    how="left",
    validate="one_to_one"
)

context26["Roster_Status"] = (
    context26["Availability_Flag"]
)

context26.to_csv(
    ROOT26 / "NBA_Forecast_With_Injuries_V2.csv",
    index=False
)

operational26 = context26[
    context26["Availability_Flag"] != "EXCLUDE"
].copy()

operational26.to_csv(
    ROOT26 / "NBA_Operational_Forecast_V2.csv",
    index=False
)

# ============================================================
# 8. REPORT
# ============================================================

print("\n" + "=" * 60)
print("NBA INJURY INTELLIGENCE V2")
print("=" * 60)

print("ESPN:", "OK" if espn_ok26 else "FAILED")
print("NBA:", "OK" if nba_ok26 else "FAILED")

print("\nStati disponibilità:")

display(
    availability26[
        "Injury_Status"
    ].value_counts()
)

print("\nGiocatori OUT:")

display(
    availability26[
        availability26["Injury_Status"] == "Out"
    ]
)

print("\nCONTROLLO LAMELO BALL")

display(
    availability26[
        availability26["Giocatore"].str.contains(
            "LaMelo Ball",
            case=False,
            na=False
        )
    ]
)

print("\nFile salvati su Google Drive:")

for file in [
    "NBA_Injury_Sources_V2.csv",
    "NBA_Availability_V2.csv",
    "NBA_Forecast_With_Injuries_V2.csv",
    "NBA_Operational_Forecast_V2.csv"
]:
    print(file)

print("\nSTEP 26A COMPLETATO")
