# Validation and testing

## Tests that run automatically

Run from the repository root:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the same command with Python 3.11. The tests cover:

- Python AST parsing for all `src/*.py`, `demo/*.py` modules and the dashboard source.
- Python AST parsing for notebook code cells, ignoring notebook-only shell commands.
- Presence of 14 curated cells, unique cell IDs and no saved execution outputs.
- Data-directory selection precedence, database presence checks and optional environment override.
- Repository Git index inspection to prevent adding raw database files or model binaries.
- Synthetic-only demo generation, independent SQLite tables, reproducibility, pregame-only feature construction and chronological partitions.
- Demo model output checks, comparison against historical baseline, probability monotonicity and input validation.
- Public Colab notebook syntax, outputs, README launch badge and absence of Drive mounting.

These are **static and local-unit checks**. Passing them does not guarantee that network calls succeed, model files are compatible, data is current, or a full dashboard launches.

The demo-only core is exercised locally without external Python packages. Its optional Gradio front end still needs an actual Google Colab smoke test.

## Tests not completed here

| Check | Status |
| --- | --- |
| Public synthetic Colab demo installation, Gradio browser and share link | Not verified |
| Full real-data Colab pipeline from clean runtime | Not verified |
| ESPN data refresh and API shape compatibility | Not verified |
| Full SQLite schema / integrity check against user's private database | Not performed |
| ML retraining and accuracy metrics from held-out dates | Not verified |
| Prediction-time leakage and calibration validation | Not independently verified |
| Injury data freshness and lineage validation | Not verified |
| Gradio dashboard browser / authentication test | Not verified |
| Scheduled update pipeline reliability | Not verified |

## Next validation milestone

Use a **copy** of the original data directory, not the only copy. Execute setup, collection, feature creation and a fresh model train in a new Colab runtime. Record sample counts, temporal splits, MAE, baseline MAE, data timestamps and forecasts. Separately test the dashboard and availability integrations. Do not publish any private datasets, raw logs or user credentials.

The current tree omits ad hoc experimental backtest scripts. The trained-model validation portions remain, since those are necessary to assess predictive quality.
