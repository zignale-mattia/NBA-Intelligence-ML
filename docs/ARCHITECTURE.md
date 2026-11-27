# Architecture and stage dependencies

The notebook is a curated collection of Colab stages. **Notebook cells are not independent applications**: each may read from the SQLite database, use files written by previous stages or rely on earlier Colab variables.

| Cell (1-based) | Notebook stage | Input / prerequisite | Principal output |
| --- | --- | --- | --- |
| 1 | Environment setup | Colab runtime; `NBA_DATA_DIR` optional | `BASE_DIR` and Python imports |
| 2 | Historical game collection | ESPN API, data directory | SQLite game table |
| 3 | Player boxscore collection | Completed games, ESPN API | SQLite player boxscore table |
| 4 | Exploratory analysis | SQLite tables | Exploratory tables and plots |
| 5 | Team ML prediction | Historical game data | Team win classifier artifact |
| 6 | Upcoming game prediction | Classifier artifact and historical data | Future team-game forecasts |
| 7 | Incremental database update | Existing SQLite database | Newly completed games and box scores |
| 8 | Advanced features | SQLite player and game tables | Feature CSV and feature list |
| 9 | Ensemble model training | Feature CSV and list | V3 trained models and held-out predictions |
| 10 | Player forecast engine | V3 model artifact and history | V3 player forecast files |
| 11 | Threshold probability engine | Held-out predictions + V3 forecasts | Estimated probabilities and interval exports |
| 12 | Injury and lineup context | V3 forecasts, probabilities, SQLite and injury feeds | Contextual forecasts, lineups and teammate impact |
| 13 | Injury report ingestion | Live injury information + V3 forecasts | Availability files |
| 14 | Contextual injury integration | Prior injury and contextual forecast files | Updated contextual outputs with backups |

Cell 14 consumes contextual outputs from cell 12 and availability files from cell 13. Run these stages in order; live feed availability and output compatibility still require runtime verification.

## Data handling

- Primary storage: SQLite database `nba_database.db`; never committed.
- Tabular intermediates: CSV and Excel; intentionally ignored by Git.
- Model artifacts: joblib; must be generated locally from trusted data.
- Main forecast: `NBA_Forecast_Engine_V3.csv`.
- Probability output: `NBA_Probability_Engine_V3.csv`.
- Dashboard: reads multiple forecast and contextual files; missing outputs will affect particular dashboard tabs.

## Configuring paths

The setup notebook cell selects `BASE_DIR` by checking `NBA_DATA_DIR`, then existing databases in supported Colab mounts. The pure-Python function `src.project_config.resolve_data_dir` implements the same selection logic for scripts/tests, without mounting or modifying any files.

The original reference scripts may still contain Colab-specific paths or assume global state. Adapt them before attempting independent execution.

## Security and verification boundaries

- Never commit database dumps, proprietary datasets, tokens or passwords.
- Gradio `share=True` creates a remote-accessible temporary address; login protection is not a substitute for secure hosting.
- Probabilities should be validated against timestamped pregame predictions, with data leakage checks.
- No end-to-end, live ESPN, or dashboard browser test was performed during this repository cleanup.

## Public recruiter demo (separate architecture)

The [Colab demo](COLAB.md) does **not** use any real basketball data. Its
flow is: seeded simulated match logs → temporary SQLite table → last-five/ten
pregame features → standard-library ridge regression → per-player temporal
train/validation/test partitions → synthetic threshold estimates → Gradio.
The code lives in `demo/public_demo.py`, `demo/app.py` and
`notebooks/NBA_Intelligence_Public_Demo.ipynb`. The original research
pipeline above remains unchanged, apart from allowing the V5 dashboard to
respect an explicit `NBA_DATA_DIR` before checking Drive mounts.
