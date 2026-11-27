"""Regenerate the checked-in Colab portfolio notebook with no execution outputs."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/NBA_Intelligence_Public_Demo.ipynb"


def markdown(cell_id: str, source: str):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(cell_id: str, source: str):
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown("welcome", """# NBA Intelligence — Interactive Colab Portfolio Demo

**⚠️ SYNTHETIC DATA ONLY. No real NBA player predictions.**

This notebook showcases a self-contained NBA-style data science pipeline:
reproducible simulated box scores → temporary SQLite → leakage-aware pregame
features → regularized linear regression → chronological holdout metrics →
threshold estimates → interactive Gradio dashboard.

- **No access** to the author's personal Google Drive, private SQLite database,
  API keys, account credentials, or pre-trained models.
- **No live ESPN/NBA requests** in this public demo. All game values are invented
  and labeled as such.
- The more advanced **real-data research notebook** remains available separately
  in the same public repository. It requires collecting data and producing its
  own intermediate artifacts before forecasting.
- The optional **Gradio share link is publicly reachable** and expires with the
  Colab runtime. Do not upload private files to this demonstration.

**Usage:** In a clean Colab runtime, click **Runtime → Run all**. The first cell
installs dependencies. If Colab requests a runtime restart because NumPy/Pandas
changed, restart the session and select Run all once again.
"""),
    code("setup", """# 1/3 — Clone the public source and install the isolated demo requirements.
import subprocess
import sys
from pathlib import Path

REPO = Path("/content/NBA-Intelligence-ML")
if not REPO.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/zignale-mattia/NBA-Intelligence-ML.git",
         str(REPO)],
        check=True,
    )

requirements = REPO / "demo/requirements-colab.txt"
if not requirements.is_file():
    raise FileNotFoundError(f"Demo requirements missing: {requirements}")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r",
     str(requirements)],
    check=True,
)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
print("Source ready:", REPO)
print("Dependencies installed. If Colab requests a restart, restart and Run all.")
"""),
    code("data-and-models", """# 2/3 — All generated records are simulated; the SQLite file stays in /content.
from demo.public_demo import create_demo, forecast

state = create_demo("/content/NBA_Intelligence_Demo")
print("SYNTHETIC-ONLY DATASET — NOT REAL NBA DATA")
print("Simulation seed:", state["seed"])
print("Invented game logs:", len(state["games"]))
print("Temporary SQLite database:", state["database_path"])
print("\\nChronological test-set mean absolute errors (synthetic only):")
for stat, metrics in state["models"].items():
    print(
        f"  {stat:>3}: Ridge MAE {metrics['test_mae']:.2f} | "
        f"last-10 baseline MAE {metrics['baseline_mae']:.2f} | "
        f"test samples {metrics['test_n']}"
    )
"""),
    code("gradio-dashboard", """# 3/3 — Launch interactive public demo; no Google Drive or personal data.
from demo.app import make_app

dashboard = make_app(state)
dashboard.launch(share=True, inline=False, show_error=True)
print("Only simulated data is exposed. Close or disconnect Colab to end the demo.")
"""),
]
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Wrote", OUTPUT)
