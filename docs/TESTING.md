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
- Real ESPN gamelog schema parsing with mocked provider responses and verifiable ESPN source links.
- Strict separation of temporary real-data SQLite from the original database.
- Prior-game feature construction, leakage checks, chronological partitions, forecasting output and input validation.
- Public real-data Colab notebook syntax, outputs, README launch badge and absence of Drive mounting.

These are **static and local-unit checks**. Passing them does not guarantee that network calls succeed, model files are compatible, data is current, or a full dashboard launches.

The ESPN parser and fallback forecasting core are tested locally with standard library mocks. A real ESPN integration test retrieved real game logs and checked out-of-sample MAE; the new nine-tab Gradio browser front end was built and rendered locally in Safari using real ESPN observations. Its new Colab public share link still needs a fresh-runtime smoke test.

## Tests not completed here

| Check | Status |
| --- | --- |
| Public REAL ESPN Colab demo installation, Gradio browser and share link | Not verified |
| Full real-data Colab pipeline from clean runtime | Not verified |
| ESPN API behavior and data refresh on 2026-11-27 | Verified live, not guaranteed for later dates |
| Full SQLite schema / integrity check against user's private database | Not performed |
| Public demo ML training and held-out MAE versus baseline | Checked locally on actual ESPN athlete histories |
| Prediction-time leakage and calibration validation | Not independently verified |
| Injury data freshness and lineage validation | Not verified |
| Public dashboard local browser | HTTP 200 and Safari visual rendering checked; new Colab shared link remains untested |
| Scheduled update pipeline reliability | Not verified |

## Next validation milestone

Use a **copy** of the original data directory, not the only copy. Execute setup, collection, feature creation and a fresh model train in a new Colab runtime. Record sample counts, temporal splits, MAE, baseline MAE, data timestamps and forecasts. Separately test the dashboard and availability integrations. Do not publish any private datasets, raw logs or user credentials.

The current tree omits ad hoc experimental backtest scripts. The trained-model validation portions remain, since those are necessary to assess predictive quality.

## Extended analytics dashboard test evidence

On November 27, 2026, an expanded ESPN integration run downloaded **180
completed regular-season NBA games and 3,993 real player lines for 460
athletes and all 30 teams**, with no failed source requests. The temporary
SQLite database was reopened and row counts matched ESPN acquisition.
Gradio 5.50.0 was locally installed in an isolated Python 3.11 environment,
the new dashboard with 125 components and eight event callbacks was
constructed successfully, the local HTTP endpoint returned 200, and the
page rendered in Safari.

Additional offline tests cover regular-season filtering, actual ESPN
game-summary athlete parsing, shot attempts, team score reconstruction,
sample rankings, database isolation, and HTML escaping of untrusted strings.
This does not establish uninterrupted ESPN availability or a successful
deployment in a brand-new Google Colab runtime.
