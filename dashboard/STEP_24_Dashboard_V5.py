
# ============================================================
# STEP 24 — PROFESSIONAL NBA DASHBOARD V5
# NBA AI PREDICTION SYSTEM
# ============================================================

import os
import sqlite3
import secrets
import numpy as np
import pandas as pd
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go

from google.colab import drive
from IPython.display import display

# ============================================================
# 1. RIPRISTINO AUTOMATICO GOOGLE DRIVE
# ============================================================

mounts = [
    "/content/drive",
    "/content/gdrive_nba",
    "/content/nba_drive"
]

# Respect an explicit independent Colab data directory first.
# Existing private Drive discovery remains the fallback for prior users.
BASE_DIR = os.environ.get("NBA_DATA_DIR")
if BASE_DIR is not None and not os.path.isfile(
    os.path.join(BASE_DIR, "nba_database.db")
):
    raise FileNotFoundError(
        "NBA_DATA_DIR does not contain nba_database.db: " + BASE_DIR
    )

if BASE_DIR is None:
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

DB_PATH = os.path.join(
    BASE_DIR, "nba_database.db"
)

print("Google Drive collegato.")
print("Cartella:", BASE_DIR)

# ============================================================
# 2. CONFIGURAZIONE FILE
# ============================================================

FILES = {

    "forecast":
        "NBA_Forecast_Engine_V3.csv",

    "context":
        "NBA_Context_Aware_Forecast.csv",

    "probabilities":
        "NBA_Probability_Engine_V3.csv",

    "context_probabilities":
        "NBA_Context_Probabilities.csv",

    "injuries":
        "NBA_Injury_Report.csv",

    "lineups":
        "NBA_Estimated_Lineups.csv",

    "impact":
        "NBA_Teammate_Impact.csv",

    "games":
        "NBA_Forecast_Prossime_Partite.csv",

    "test":
        "NBA_Advanced_Test_Predictions.csv",

    "retraining":
        "NBA_Model_Retraining_Report.xlsx",

    "probability_report":
        "NBA_Probability_Engine_V3.xlsx"
}

TARGETS = [
    "PTS", "REB", "AST", "3PM", "PRA"
]

STAT_NAMES = {

    "PTS": "Punti",
    "REB": "Rimbalzi",
    "AST": "Assist",
    "3PM": "Triple",
    "PRA": "Punti + Rimbalzi + Assist"
}

# ============================================================
# 3. FUNZIONI CARICAMENTO SICURO
# ============================================================

def load_csv(key):

    path = os.path.join(
        BASE_DIR, FILES[key]
    )

    if not os.path.isfile(path):
        return pd.DataFrame()

    try:

        return pd.read_csv(
            path,
            dtype={
                "game_id": str,
                "player_id": str
            }
        )

    except Exception as e:

        print(
            f"Errore caricamento {key}: {e}"
        )

        return pd.DataFrame()


def load_excel(key, sheet):

    path = os.path.join(
        BASE_DIR, FILES[key]
    )

    if not os.path.isfile(path):
        return pd.DataFrame()

    try:

        return pd.read_excel(
            path,
            sheet_name=sheet
        )

    except Exception as e:

        print(
            f"Errore Excel {sheet}: {e}"
        )

        return pd.DataFrame()


def query_db(sql, params=()):

    with sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True,
        timeout=30
    ) as conn:

        return pd.read_sql_query(
            sql,
            conn,
            params=params
        )


def future_only(data):

    if data.empty:
        return data

    date_col = next(
        (
            c for c in [
                "date",
                "game_date",
                "Data",
                "Date"
            ]
            if c in data.columns
        ),
        None
    )

    if date_col is None:
        return pd.DataFrame()

    dates = pd.to_datetime(
        data[date_col],
        utc=True,
        errors="coerce",
        format="mixed"
    )

    return data[
        dates > pd.Timestamp.now(tz="UTC")
    ].copy()


def empty_chart(message):

    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        showarrow=False
    )

    fig.update_layout(
        template="plotly_dark",
        height=350
    )

    return fig


def select_columns(data, columns):

    return data[
        [c for c in columns if c in data.columns]
    ].copy()


def safe_round(value):

    try:

        value = float(value)

        if np.isfinite(value):
            return f"{value:.2f}"

    except Exception:
        pass

    return "N/D"

# ============================================================
# 4. FORECAST CON CONTROLLO VERSIONE
# ============================================================

def get_forecast():

    raw = future_only(
        load_csv("forecast")
    )

    context = future_only(
        load_csv("context")
    )

    if raw.empty:
        return raw

    if context.empty:
        return raw

    required = [
        "game_id",
        "player_id",
        "generated_at"
    ]

    if not all(
        c in raw.columns and c in context.columns
        for c in required
    ):

        return raw

    metadata = [
        "game_id",
        "player_id",
        "generated_at",
        "Injury_Status",
        "Availability_Flag",
        "Estimated_Role",
        "Minutes_Last10",
        "Listed_Out_Teammates",
        "Context_Generated_At",
        "Injury_Feed_OK"
    ]

    metadata = [
        c for c in metadata
        if c in context.columns
    ]

    extra = context[metadata].copy()

    extra = extra.drop_duplicates(
        required,
        keep="last"
    )

    result = raw.merge(
        extra,
        on=required,
        how="left",
        suffixes=("", "_context")
    )

    for col in metadata:

        new_col = f"{col}_context"

        if new_col in result.columns:

            result[col] = result[new_col].combine_first(
                result[col]
            )

            result = result.drop(
                columns=new_col
            )

    return result


def get_probabilities():

    data = future_only(
        load_csv("probabilities")
    )

    forecast = get_forecast()

    required = [
        "game_id",
        "player_id",
        "generated_at"
    ]

    if data.empty or forecast.empty:
        return pd.DataFrame()

    if "Forecast_Generated_At" not in data.columns:
        return pd.DataFrame()

    if not all(
        c in forecast.columns
        for c in required
    ):
        return pd.DataFrame()

    keys = forecast[required].rename(
        columns={
            "generated_at":
                "Forecast_Generated_At"
        }
    ).drop_duplicates()

    return data.merge(
        keys,
        on=[
            "game_id",
            "player_id",
            "Forecast_Generated_At"
        ],
        how="inner"
    )

# ============================================================
# 5. OVERVIEW
# ============================================================

def overview():

    data = get_forecast()

    probabilities = get_probabilities()

    try:

        games_count = query_db("""
            SELECT COUNT(*) AS n
            FROM espn_games
            WHERE completed = 1
        """).iloc[0]["n"]

        box_count = query_db("""
            SELECT COUNT(*) AS n
            FROM espn_player_boxscores
        """).iloc[0]["n"]

    except Exception:

        games_count = "N/D"
        box_count = "N/D"

    n_games = (
        data["game_id"].nunique()
        if "game_id" in data.columns
        else 0
    )

    n_players = (
        data["player_id"].nunique()
        if "player_id" in data.columns
        else 0
    )

    generated = "N/D"

    if "generated_at" in data.columns:

        dates = pd.to_datetime(
            data["generated_at"],
            utc=True,
            errors="coerce"
        )

        if dates.notna().any():

            generated = dates.max().tz_convert(
                "Europe/Rome"
            ).strftime("%d/%m/%Y %H:%M")

    summary = f"""
## NBA INTELLIGENCE — SYSTEM OVERVIEW

| Indicatore | Valore |
|---|---:|
| Partite storiche completate | {games_count} |
| Player boxscores | {box_count} |
| Partite future | {n_games} |
| Giocatori con forecast | {n_players} |
| Forecast giocatore-partita | {len(data)} |
| Probabilità disponibili | {len(probabilities)} |
| Ultimo forecast | {generated} |
| Modello di produzione | V3 |

**Nota:** i forecast sono condizionati alla partecipazione.
"""

    if data.empty:

        summary += (
            "\n\nATTENZIONE: nessun forecast futuro. "
            "Aggiornare STEP 17."
        )

    status = []

    for key, filename in FILES.items():

        path = os.path.join(
            BASE_DIR, filename
        )

        exists = os.path.isfile(path)

        modified = ""

        if exists:

            modified = pd.Timestamp(
                os.path.getmtime(path),
                unit="s",
                tz="UTC"
            ).tz_convert(
                "Europe/Rome"
            ).strftime("%d/%m/%Y %H:%M")

        status.append({
            "Modulo": key,
            "File": filename,
            "Disponibile": exists,
            "Ultima modifica": modified
        })

    return summary, pd.DataFrame(status)

# ============================================================
# 6. PLAYER FORECAST
# ============================================================

def player_choices():

    data = get_forecast()

    if data.empty:
        return []

    required = [
        "player_id",
        "Giocatore"
    ]

    if not all(
        c in data.columns
        for c in required
    ):
        return []

    players = data[
        required
    ].drop_duplicates(
        "player_id"
    ).sort_values(
        "Giocatore"
    )

    return [

        (
            f"{r.Giocatore} ({r.player_id})",
            r.player_id
        )

        for r in players.itertuples(
            index=False
        )
    ]


def game_choices(player_id):

    data = get_forecast()

    if data.empty or player_id is None:

        return gr.update(
            choices=[],
            value=None
        )

    selected = data[
        data["player_id"] == player_id
    ].drop_duplicates(
        "game_id"
    ).sort_values("date")

    options = []

    for r in selected.itertuples(index=False):

        date = pd.to_datetime(
            r.date,
            utc=True,
            errors="coerce"
        )

        label = str(r.game_id)

        if pd.notna(date):

            label = date.tz_convert(
                "Europe/Rome"
            ).strftime("%d/%m %H:%M")

        label += (
            f" — vs {getattr(r, 'Avversario', '')}"
        )

        options.append(
            (label, r.game_id)
        )

    return gr.update(
        choices=options,
        value=options[0][1] if options else None
    )


def player_analysis(player_id, game_id, target):

    data = get_forecast()

    if data.empty or player_id is None:

        return (
            "Nessun forecast disponibile.",
            pd.DataFrame(),
            empty_chart("Nessun dato"),
            pd.DataFrame()
        )

    selected = data[
        (data["player_id"] == player_id)
        & (data["game_id"] == game_id)
    ]

    if selected.empty:

        return (
            "Seleziona una partita.",
            pd.DataFrame(),
            empty_chart("Nessun dato"),
            pd.DataFrame()
        )

    r = selected.iloc[0]

    status = r.get(
        "Injury_Status",
        "Unknown"
    )

    if pd.isna(status):
        status = "Unknown"

    info = f"""
## {r.get("Giocatore", "")}

**Squadra:** {r.get("Squadra", "")}

**Avversario:** {r.get("Avversario", "")}

**Injury Status:** {status}

**Ruolo stimato:** {r.get("Estimated_Role", "N/D")}

**Minuti ultime 10:** {safe_round(r.get("Minutes_Last10"))}

**Injury report acquisito:** {r.get("Context_Generated_At", "N/D")}
"""

    rows = []

    for stat in TARGETS:

        value = r.get(
            f"Forecast_{stat}",
            np.nan
        )

        rows.append({

            "Statistica": stat,

            "Forecast": (
                value if status != "Out"
                else np.nan
            )
        })

    forecast_table = pd.DataFrame(rows)

    if status == "Out":

        info += (
            "\n\n**Giocatore indicato Out "
            "nell'ultimo report disponibile.**"
        )

    try:

        history = query_db("""
            SELECT
                game_date,
                team_name,
                minutes,
                pts,
                reb,
                ast,
                fg3m
            FROM espn_player_boxscores
            WHERE CAST(player_id AS TEXT) = ?
            AND did_not_play = 0
            AND minutes > 0
            ORDER BY game_date DESC
            LIMIT 20
        """, (str(player_id),))

    except Exception:

        history = pd.DataFrame()

    mapping = {

        "PTS": "pts",
        "REB": "reb",
        "AST": "ast",
        "3PM": "fg3m",
        "PRA": "PRA"
    }

    if history.empty:

        fig = empty_chart(
            "Storico non disponibile"
        )

    else:

        history["PRA"] = (
            history["pts"]
            + history["reb"]
            + history["ast"]
        )

        history = history.sort_values(
            "game_date"
        )

        stat_col = mapping[target]

        fig = px.line(
            history,
            x="game_date",
            y=stat_col,
            markers=True,
            title=f"Ultime partite — {target}",
            template="plotly_dark"
        )

        value = r.get(
            f"Forecast_{target}",
            np.nan
        )

        if pd.notna(value) and status != "Out":

            fig.add_hline(
                y=float(value),
                line_dash="dash",
                line_color="#38bdf8",
                annotation_text="Forecast"
            )

        fig.update_layout(height=400)

    return (
        info,
        forecast_table.round(2),
        fig,
        history.round(2)
    )

# ============================================================
# 7. PROBABILITY ENGINE
# ============================================================

def probability_analysis(
    player_id,
    game_id,
    target
):

    data = get_probabilities()

    if data.empty:

        return (
            "Probabilità assenti o non aggiornate.",
            pd.DataFrame(),
            empty_chart("Eseguire STEP 18")
        )

    forecast = get_forecast()

    player = forecast[
        (forecast["player_id"] == player_id)
        & (forecast["game_id"] == game_id)
    ]

    if player.empty:

        return (
            "Forecast non disponibile.",
            pd.DataFrame(),
            empty_chart("Nessun dato")
        )

    status = player.iloc[0].get(
        "Injury_Status",
        "Unknown"
    )

    if status == "Out":

        return (
            "Giocatore indicato Out.",
            pd.DataFrame(),
            empty_chart("Forecast escluso")
        )

    selected = data[
        (data["player_id"] == player_id)
        & (data["game_id"] == game_id)
        & (data["Target"] == target)
    ].sort_values("Threshold")

    if selected.empty:

        return (
            "Nessuna probabilità disponibile.",
            pd.DataFrame(),
            empty_chart("Nessun dato")
        )

    selected = selected.copy()

    selected["Probability_pct"] = (
        100 * selected["Probability_At_Least"]
    )

    fig = px.bar(
        selected,
        x="Threshold",
        y="Probability_pct",
        color="Probability_pct",
        color_continuous_scale="Blues",
        title=f"Probability Distribution — {target}",
        template="plotly_dark"
    )

    fig.update_yaxes(
        range=[0, 100],
        ticksuffix="%"
    )

    table = select_columns(
        selected,
        [
            "Target",
            "Threshold",
            "Forecast",
            "Probability_pct",
            "Distribution_Source",
            "Player_Sample",
            "Reference_Sample"
        ]
    )

    info = (
        "Probabilità stimate dal modello, "
        "condizionate alla partecipazione. "
        "Non sono garanzie del risultato."
    )

    return (
        info,
        table.round(2),
        fig
    )

# ============================================================
# 8. PLAYER COMPARISON
# ============================================================

def compare_players(player_a, player_b):

    data = get_forecast()

    if data.empty:

        return (
            pd.DataFrame(),
            empty_chart("Nessun forecast")
        )

    rows = []

    for pid in [player_a, player_b]:

        selected = data[
            data["player_id"] == pid
        ].sort_values("date")

        if selected.empty:
            continue

        r = selected.iloc[0]

        status = r.get(
            "Injury_Status",
            "Unknown"
        )

        for target in TARGETS:

            value = r.get(
                f"Forecast_{target}",
                np.nan
            )

            rows.append({

                "Giocatore": r["Giocatore"],

                "Avversario": r.get(
                    "Avversario", ""
                ),

                "Data": r.get("date", ""),

                "Target": target,

                "Forecast": (
                    value if status != "Out"
                    else np.nan
                ),

                "Injury_Status": status
            })

    result = pd.DataFrame(rows)

    if result.empty:

        return (
            result,
            empty_chart("Selezionare i giocatori")
        )

    fig = px.bar(
        result,
        x="Target",
        y="Forecast",
        color="Giocatore",
        barmode="group",
        title="Player Comparison — prossima partita",
        template="plotly_dark"
    )

    return result.round(2), fig

# ============================================================
# 9. GAME PREDICTIONS
# ============================================================

def game_predictions():

    data = future_only(
        load_csv("games")
    )

    if data.empty:

        return (
            "Nessuna previsione squadra futura. "
            "Aggiornare STEP 6.",
            pd.DataFrame()
        )

    return (
        "Previsioni future delle squadre.",
        data.round(4)
    )

# ============================================================
# 10. INJURY INTELLIGENCE
# ============================================================

def injury_analysis():

    data = get_forecast()

    injuries = load_csv("injuries")

    lineups = future_only(
        load_csv("lineups")
    )

    impact = load_csv("impact")

    if (
        "game_id" in impact.columns
        and "game_id" in data.columns
    ):

        impact = impact[
            impact["game_id"].isin(
                data["game_id"]
            )
        ]

    if (
        data.empty
        or "Injury_Status" not in data.columns
    ):

        return (
            empty_chart("Injury report non disponibile"),
            injuries,
            lineups,
            impact
        )

    summary = data[
        "Injury_Status"
    ].fillna(
        "Unknown"
    ).value_counts().reset_index()

    summary.columns = [
        "Status",
        "Forecast_Count"
    ]

    fig = px.bar(
        summary,
        x="Status",
        y="Forecast_Count",
        color="Status",
        template="plotly_dark",
        title="Player Availability"
    )

    return (
        fig,
        injuries,
        lineups,
        impact.round(2)
    )

# ============================================================
# 11. MODEL PERFORMANCE
# ============================================================

def model_performance():

    history = load_csv("test")

    if history.empty:

        return (
            pd.DataFrame(),
            empty_chart("Storico test assente"),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame()
        )

    required = [
        "Actual",
        "Forecast",
        "Target"
    ]

    if not all(
        c in history.columns
        for c in required
    ):

        return (
            pd.DataFrame(),
            empty_chart("Schema test non compatibile"),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame()
        )

    for col in ["Actual", "Forecast"]:

        history[col] = pd.to_numeric(
            history[col],
            errors="coerce"
        )

    history = history.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(
        subset=required
    )

    history["AE"] = (
        history["Actual"]
        - history["Forecast"]
    ).abs()

    history["SE"] = (
        history["Actual"]
        - history["Forecast"]
    ) ** 2

    history["Bias"] = (
        history["Forecast"]
        - history["Actual"]
    )

    metrics = history.groupby(
        "Target",
        as_index=False
    ).agg(

        MAE=("AE", "mean"),

        MSE=("SE", "mean"),

        Bias=("Bias", "mean"),

        Sample=("AE", "size")
    )

    metrics["RMSE"] = np.sqrt(
        metrics.pop("MSE")
    )

    fig = px.bar(
        metrics,
        x="Target",
        y="MAE",
        color="Target",
        title="Historical Test — Mean Absolute Error",
        template="plotly_dark"
    )

    calibration = load_excel(
        "probability_report",
        "Calibration ECE"
    )

    coverage = load_excel(
        "probability_report",
        "Interval Coverage"
    )

    retraining = load_excel(
        "retraining",
        "Run Summary"
    )

    return (
        metrics.round(4),
        fig,
        calibration,
        coverage,
        retraining
    )

# ============================================================
# 12. DATABASE MONITOR
# ============================================================

def database_monitor():

    tables = query_db("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
    """)

    rows = []

    for name in tables["name"]:

        safe_name = name.replace(
            '"', '""'
        )

        count = query_db(
            f'SELECT COUNT(*) AS n FROM "{safe_name}"'
        ).iloc[0]["n"]

        rows.append({

            "Tabella": name,
            "Record": count
        })

    size = (
        os.path.getsize(DB_PATH)
        / 1024 / 1024
    )

    summary = f"""
## Database Monitor

**Dimensione database:** {size:.2f} MB

**Tabelle presenti:** {len(rows)}

**Modalità:** sola lettura
"""

    return summary, pd.DataFrame(rows)

# ============================================================
# 13. STILE PROFESSIONALE
# ============================================================

CSS = """
body, .gradio-container {background:#080e1c !important; color:#e2e8f0 !important;}
.gradio-container {max-width:1500px !important; margin:auto !important; padding:24px !important;}
#main-header {padding:32px; border-radius:20px; background:linear-gradient(120deg,#101b30,#17365c); border:1px solid #334155;}
#main-header h1 {color:#f8fafc !important; font-size:36px !important; letter-spacing:1px;}
#main-header p {color:#7dd3fc !important;}
.gradio-container .block {border-radius:14px !important;}
.gradio-container button {border-radius:10px !important;}
.gradio-container .tab-nav {flex-wrap:wrap; gap:6px;}
.gradio-container .tab-nav button {padding:12px !important;}
footer {display:none !important;}
"""


# ============================================================
# 14. CREAZIONE DASHBOARD
# ============================================================

players = player_choices()

default_player = next(
    (
        pid for name, pid in players
        if "jokic" in name.lower()
        or "jokić" in name.lower()
    ),
    players[0][1] if players else None
)

import inspect
from pathlib import Path
import json
from html import escape

def quality_view():
    root = Path(BASE_DIR) / "Production_Pipeline"
    p = root / "Quality_Control" / "Latest_Quality_Status.json"
    q = root / "Quality_Control" / "NBA_Quality_Control.csv"
    if not p.exists():
        return "Esegui STEP 22D in Colab.", pd.DataFrame()
    info = json.loads(p.read_text(encoding="utf-8"))
    return "\n".join(f"**{k}:** {v}  " for k,v in info.items()), pd.read_csv(q) if q.exists() else pd.DataFrame()

def kpi_view():
    f = get_forecast()
    p = get_probabilities()
    items = [("Partite future", f["game_id"].nunique() if "game_id" in f else 0), ("Giocatori", f["player_id"].nunique() if "player_id" in f else 0), ("Forecast", len(f)), ("Probabilità", len(p))]
    return '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px">'+''.join(f'<div style="background:#151f32;border:1px solid #334155;border-radius:14px;padding:20px;color:#94a3b8">{escape(k)}<div style="font-size:32px;color:#7dd3fc;font-weight:700">{v:,}</div></div>' for k,v in items)+'</div>'

def export_files():
    names = list(FILES.values()) + ["NBA_Probability_Summary_V3.csv", "NBA_Forecast_Engine_V3.xlsx", "NBA_Advanced_Model_Accuracy.xlsx"]
    return [str(Path(BASE_DIR)/x) for x in dict.fromkeys(names) if (Path(BASE_DIR)/x).is_file()]

# Close the previous dashboard before starting a new share link.
if "app" in globals():
    try: app.close()
    except Exception: pass

theme_v5 = gr.themes.Soft(primary_hue="sky", neutral_hue="slate").set(body_background_fill="#080e1c", body_background_fill_dark="#080e1c", body_text_color="#e2e8f0", body_text_color_dark="#e2e8f0", block_background_fill="#151f32", block_background_fill_dark="#151f32", block_label_text_color="#cbd5e1", block_label_text_color_dark="#cbd5e1", input_background_fill="#101b30", input_background_fill_dark="#101b30", input_text_color="#e2e8f0", input_text_color_dark="#e2e8f0")
block_options = {"title":"NBA Intelligence V5"}
launch_params = inspect.signature(gr.Blocks.launch).parameters
if "theme" not in launch_params: block_options["theme"] = theme_v5
if "css" not in launch_params: block_options["css"] = CSS
with gr.Blocks(**block_options) as app:

    gr.Markdown(
        """
# NBA INTELLIGENCE V5

Machine Learning · Predictive Analytics · Model Monitoring
        """,
        elem_id="main-header"
    )

    gr.Markdown(
        "Dashboard privata — NBA Data Science Analytics"
    )

    # --------------------------------------------------------
    # TAB 1 — OVERVIEW
    # --------------------------------------------------------

    with gr.Tab("Overview"):

        refresh_overview = gr.Button(
            "Aggiorna sistema",
            variant="primary"
        )

        kpi_cards = gr.HTML()
        overview_text = gr.Markdown()

        files_table = gr.Dataframe(
            interactive=False
        )

        refresh_overview.click(
            overview,
            outputs=[
                overview_text,
                files_table
            ]
        )

    # --------------------------------------------------------
    # TAB 2 — PLAYER FORECAST
    # --------------------------------------------------------

    with gr.Tab("Player Forecast"):

        refresh_players = gr.Button(
            "Ricarica giocatori"
        )

        with gr.Row():

            player = gr.Dropdown(
                choices=players,
                value=default_player,
                label="Giocatore"
            )

            game = gr.Dropdown(
                choices=[],
                label="Partita"
            )

            target = gr.Dropdown(
                choices=TARGETS,
                value="PTS",
                label="Statistica"
            )

        analyze = gr.Button(
            "Analizza giocatore",
            variant="primary"
        )

        player_info = gr.Markdown()

        player_table = gr.Dataframe(
            interactive=False
        )

        player_chart = gr.Plot()

        history_table = gr.Dataframe(
            interactive=False
        )

        player.change(
            game_choices,
            inputs=[player],
            outputs=[game]
        )

        analyze.click(
            player_analysis,
            inputs=[player, game, target],
            outputs=[
                player_info,
                player_table,
                player_chart,
                history_table
            ]
        )

    # --------------------------------------------------------
    # TAB 3 — PROBABILITIES
    # --------------------------------------------------------

    with gr.Tab("Probability Engine"):

        prob_player = gr.Dropdown(
            choices=players,
            value=default_player,
            label="Giocatore"
        )

        prob_game = gr.Dropdown(
            choices=[],
            label="Partita"
        )

        prob_target = gr.Dropdown(
            choices=TARGETS,
            value="PTS",
            label="Statistica"
        )

        prob_button = gr.Button(
            "Calcola probabilità disponibili",
            variant="primary"
        )

        prob_info = gr.Markdown()

        prob_table = gr.Dataframe(
            interactive=False
        )

        prob_chart = gr.Plot()

        prob_player.change(
            game_choices,
            inputs=[prob_player],
            outputs=[prob_game]
        )

        prob_button.click(
            probability_analysis,
            inputs=[
                prob_player,
                prob_game,
                prob_target
            ],
            outputs=[
                prob_info,
                prob_table,
                prob_chart
            ]
        )

    # --------------------------------------------------------
    # TAB 4 — PLAYER COMPARISON
    # --------------------------------------------------------

    with gr.Tab("Player Comparison"):

        with gr.Row():

            player_a = gr.Dropdown(
                choices=players,
                label="Giocatore A"
            )

            player_b = gr.Dropdown(
                choices=players,
                label="Giocatore B"
            )

        compare_button = gr.Button(
            "Confronta",
            variant="primary"
        )

        comparison_table = gr.Dataframe(
            interactive=False
        )

        comparison_chart = gr.Plot()

        compare_button.click(
            compare_players,
            inputs=[
                player_a,
                player_b
            ],
            outputs=[
                comparison_table,
                comparison_chart
            ]
        )

    # --------------------------------------------------------
    # TAB 5 — GAME PREDICTIONS
    # --------------------------------------------------------

    with gr.Tab("Game Predictions"):

        games_button = gr.Button(
            "Carica prossime partite",
            variant="primary"
        )

        games_info = gr.Markdown()

        games_table = gr.Dataframe(
            interactive=False
        )

        games_button.click(
            game_predictions,
            outputs=[
                games_info,
                games_table
            ]
        )

    # --------------------------------------------------------
    # TAB 6 — INJURY INTELLIGENCE
    # --------------------------------------------------------

    with gr.Tab("Injury Intelligence"):

        injury_button = gr.Button(
            "Carica ultimo injury report",
            variant="primary"
        )

        injury_chart = gr.Plot()

        gr.Markdown("### Injury Report")

        injury_table = gr.Dataframe(
            interactive=False
        )

        gr.Markdown(
            "### Estimated Lineups — non confermati"
        )

        lineup_table = gr.Dataframe(
            interactive=False
        )

        gr.Markdown(
            "### Historical Teammate Impact"
        )

        impact_table = gr.Dataframe(
            interactive=False
        )

        injury_button.click(
            injury_analysis,
            outputs=[
                injury_chart,
                injury_table,
                lineup_table,
                impact_table
            ]
        )

    # --------------------------------------------------------
    # TAB 7 — MODEL PERFORMANCE
    # --------------------------------------------------------

    with gr.Tab("Model Performance"):

        model_button = gr.Button(
            "Carica performance",
            variant="primary"
        )

        metrics_table = gr.Dataframe(
            interactive=False
        )

        metrics_chart = gr.Plot()

        gr.Markdown("### Probability Calibration")

        calibration_table = gr.Dataframe(
            interactive=False
        )

        gr.Markdown("### Prediction Interval Coverage")

        coverage_table = gr.Dataframe(
            interactive=False
        )

        gr.Markdown("### Retraining Status")

        retraining_table = gr.Dataframe(
            interactive=False
        )

        model_button.click(
            model_performance,
            outputs=[
                metrics_table,
                metrics_chart,
                calibration_table,
                coverage_table,
                retraining_table
            ]
        )

    # --------------------------------------------------------
    # TAB 8 — DATABASE
    # --------------------------------------------------------

    with gr.Tab("Database Monitor"):

        db_button = gr.Button(
            "Analizza database",
            variant="primary"
        )

        db_info = gr.Markdown()

        db_table = gr.Dataframe(
            interactive=False
        )

        db_button.click(
            database_monitor,
            outputs=[
                db_info,
                db_table
            ]
        )

    # --------------------------------------------------------
    # AGGIORNAMENTO SELETTORI
    # --------------------------------------------------------

    with gr.Tab("Quality Control"):
        qc_btn = gr.Button("Leggi ultimo controllo qualità", variant="primary")
        qc_info = gr.Markdown()
        qc_table = gr.Dataframe(interactive=False)
        qc_btn.click(quality_view, outputs=[qc_info,qc_table])
    with gr.Tab("Export Center"):
        export_btn = gr.Button("Carica report disponibili", variant="primary")
        export_output = gr.File(file_count="multiple", interactive=False)
        export_btn.click(export_files, outputs=export_output)
    with gr.Tab("Control Center"):
        gr.Markdown("## Aggiornamento dati\nEsegui STEP 22C e 22D oppure il pannello STEP 23 in Colab. Questa dashboard consulta i dati salvati, non avvia il training o la pipeline.")
        control_refresh = gr.Button("Rileggi stato dei dati")
        control_info = gr.Markdown()
        control_table = gr.Dataframe(interactive=False)
        control_refresh.click(overview, outputs=[control_info,control_table])
    app.load(kpi_view, outputs=kpi_cards)
    refresh_overview.click(kpi_view, outputs=kpi_cards)

    def refresh_player_lists():

        options = player_choices()

        first = (
            options[0][1]
            if options else None
        )

        return (

            gr.update(
                choices=options,
                value=first
            ),

            gr.update(
                choices=options,
                value=first
            ),

            gr.update(
                choices=options,
                value=None
            ),

            gr.update(
                choices=options,
                value=None
            )
        )

    refresh_players.click(
        refresh_player_lists,
        outputs=[
            player,
            prob_player,
            player_a,
            player_b
        ]
    ).then(
        game_choices,
        inputs=[player],
        outputs=[game]
    ).then(
        game_choices,
        inputs=[prob_player],
        outputs=[prob_game]
    )

    # --------------------------------------------------------
    # CARICAMENTO INIZIALE
    # --------------------------------------------------------

    app.load(
        overview,
        outputs=[
            overview_text,
            files_table
        ]
    )

    app.load(
        game_choices,
        inputs=[player],
        outputs=[game]
    )

    app.load(
        game_choices,
        inputs=[prob_player],
        outputs=[prob_game]
    )

# ============================================================
# 15. ACCESSO PROTETTO
# ============================================================

USERNAME = "admin"

# Use a private environment variable, or enter a password interactively.
# Never write the password to notebook output or source control.
from getpass import getpass
PASSWORD = os.environ.get("NBA_DASHBOARD_PASSWORD") or getpass(
    "Choose a dashboard password (input hidden): "
)
if not PASSWORD:
    raise ValueError("A non-empty dashboard password is required.")

print("\n" + "=" * 60)

print("NBA INTELLIGENCE V5")
print("DASHBOARD PRONTA PER L'AVVIO")

print("=" * 60)

print("Username:", USERNAME)
print("Password: [hidden — do not share credentials]")

print("\nConserva le credenziali privatamente.")

print(
    "Apri il link Gradio che apparirà qui sotto."
)

# ============================================================
# 16. AVVIO SERVER
# ============================================================

# Compatibilità tra versioni Gradio.

import inspect

launch_parameters = inspect.signature(
    app.launch
).parameters

launch_kwargs = {

    "share": True,

    "inline": False,

    "debug": False,

    "show_error": True,

    "auth": (
        USERNAME,
        PASSWORD
    )
}

if "css" in launch_parameters:

    launch_kwargs["css"] = CSS

if "theme" in launch_parameters:

    launch_kwargs["theme"] = theme_v5

launch_kwargs["allowed_paths"] = [str(Path(BASE_DIR) / x) for x in list(FILES.values()) + ["NBA_Probability_Summary_V3.csv", "NBA_Forecast_Engine_V3.xlsx", "NBA_Advanced_Model_Accuracy.xlsx"] if (Path(BASE_DIR) / x).is_file()]
app.launch(**launch_kwargs)
