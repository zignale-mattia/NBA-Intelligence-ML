"""Rebuild the public Colab notebook for *real ESPN games only*."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/NBA_Intelligence_Public_Demo.ipynb"


def cell(kind, name, source):
    output = {
        "cell_type": kind, "id": name, "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        output["execution_count"] = None
        output["outputs"] = []
    return output


cells = [
    cell("markdown", "intro", """# NBA Intelligence — REAL NBA Stats & ML Forecasts

**LIVE ESPN DATA ACQUISITION · VERIFIED NBA PLAYERS · TRAINING ON REAL GAMES**

This notebook gets the **actual 2025-26 and 2026-27 regular-season game
logs** of selected NBA players from the public ESPN athlete JSON API, creates
a **temporary isolated SQLite database** in Colab, trains and compares
**Ridge / Random Forest / Extra Trees** models chronologically, and launches
a full nine-section interactive Gradio research terminal, including league-wide
ESPN game box scores, 30-team snapshots, hundreds of real NBA athletes,
player scouting, peer comparisons, league leaders, model validation,
source-linked game histories and next-fixture forecasts.

**No synthetic/fabricated player data.** No private Google Drive, author's
database, API keys or credentials needed. **ESPN Internet access is
required**. If ESPN is unavailable, the notebook stops rather than making
up statistics.

- The league explorer retrieves up to 180 recent completed NBA games and
  thousands of actual player performances; sample sizes vary by date.
- These are real historical observations and newly trained **estimate**
  models, not guarantees of future performance.
- Fixtures are retrieved from the ESPN scoreboard. A scheduled team game is
  **not confirmation a player will play**. The public model does *not*
  incorporate injury reports, opponent defensive ratings or official lineups.
- Model validation uses globally chronological game-date splits; the final
  test period is excluded from model selection. MAE and baseline performance
  are displayed even if the new model performs worse.
- This is a **portable public real-data workflow**. The original advanced
  14-stage research notebook is preserved separately; its V3 ensemble and
  full injury-aware dashboard are **not** claimed to run here.
- Gradio's temporary share link can be accessed by anyone who has it; the
  underlying data consists solely of public ESPN data. No private files
  should be uploaded.

**How to run:** In Google Colab, select **Runtime → Run all**. If a package
upgrade requests a runtime restart, restart and select Run all again. Allow
time for actual ESPN requests and model training. The final cell prints
your temporary Gradio public URL.
"""),
    cell("code", "install", """# 1/3 — Install the real-data demo from the latest public GitHub main.
import sys
import subprocess
from pathlib import Path

REPO = Path("/content/NBA-Intelligence-ML")
REPO_URL = "https://github.com/zignale-mattia/NBA-Intelligence-ML.git"

if not REPO.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(REPO)],
        check=True,
    )
else:
    # Reuse only a valid checkout; never fetch secrets or personal Drive files.
    subprocess.run(
        ["git", "-C", str(REPO), "pull", "--ff-only"],
        check=True,
    )

requirements = REPO / "demo/requirements-colab.txt"
if not requirements.is_file():
    raise FileNotFoundError(f"Requirements missing: {requirements}")

subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "-q", "-r", str(requirements)],
    check=True,
)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

print("GitHub source ready:", REPO)
print("Dependencies installed. Restart Colab if package changes require it.")
"""),
    cell("code", "fetch-and-train", """# 2/3 — REAL played NBA games from ESPN; new temporary SQLite and trained ML.
from demo.real_espn import prepare_data, upcoming_matchups
from demo.real_models import train_state
from demo.league_explorer import build_league_explorer, persist_league_sqlite

data = prepare_data("/content/NBA_Real_Data")
state = train_state(data, advanced=True)

# Expand the recruiter view beyond 14 modeled athletes: collect actual
# completed NBA games and all participant box scores from public ESPN.
state["league"] = build_league_explorer(days=42, max_games=180)
state["league"]["db_path"] = persist_league_sqlite(
    state["league"], "/content/NBA_Real_Data"
)

try:
    state["upcoming"] = upcoming_matchups(state["games"], days=9)
except Exception as exc:
    state["upcoming"] = {}
    print("ESPN future schedule temporarily unavailable:", exc)

print("=" * 66)
print("REAL ESPN NBA GAME DATA - NO SIMULATED RECORDS")
print("Data fetched at:", state["as_of_utc"])
print("NBA players:", len(state["histories"]))
print("Real game records:", len(state["games"]))
print("Temporary SQLite:", state["db_path"])
print("Upcoming verified team fixtures:", len(state["upcoming"]))
print("REAL league games:", len(state["league"]["matches"]))
print("REAL league player boxscores:", len(state["league"]["rows"]))
print("Unique NBA athletes:", state["league"]["players"])
print("NBA teams:", state["league"]["teams"])
print("League SQLite:", state["league"]["db_path"])
if state["failures"]:
    print("ESPN request warnings:", len(state["failures"]))
    for warning in state["failures"][:3]:
        print("  ", warning)

print("\\nOUT-OF-SAMPLE TEST RESULTS (OBSERVED NBA STATISTICS)")
for stat, result in state["models"].items():
    print(
        f"{stat:>3}: {result['selected_model']:<15} "
        f"MAE {result['test_mae']:.2f} "
        f"vs last-10 baseline {result['baseline_test_mae']:.2f} "
        f"(test {result['n_test']} games)"
    )
"""),
    cell("code", "launch", """# 3/3 — Gradio user interface with REAL ESPN observations and data provenance.
from demo.real_app import make_real_app

app = make_real_app(state)
app.launch(share=True, inline=False, show_error=True)
print("The temporary dashboard URL works while this Colab session is running.")
"""),
]
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3"
        },
        "language_info": {"name": "python"},
    },
    "cells": cells,
}
TARGET.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("Saved", TARGET)
