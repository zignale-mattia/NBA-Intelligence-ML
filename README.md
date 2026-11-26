# NBA Intelligence V6

NBA player forecasting, matchup analysis and probability estimation prototype, built with Python, ESPN-derived game and box-score data, scikit-learn ensemble models, SQLite and Gradio.

> **Project status:** research prototype. Several components were developed and run in Google Colab. The end-to-end hourly update pipeline and some backtesting integrations have **not** passed a complete production validation. Dashboard sharing links are temporary and not included.

## Features
- ESPN historical NBA game and player box-score collection.
- Player forecasts for **points (PTS), rebounds (REB), assists (AST), made three-pointers (3PM), and points+rebounds+assists (PRA)**.
- Rolling statistics, home/away, opponent trends, rest days and player-form features.
- Candidate Ridge, RandomForest, ExtraTrees, XGBoost and LightGBM regression models with evaluation by chronological partitions.
- Empirical prediction intervals and threshold probabilities.
- Injury/lineup context, probabilistic backtesting and daily forecast-versus-actual reports.
- Interactive Gradio dashboard (V5 source, V6 extension developed in notebook).

## Repository layout
- `notebooks/`: sanitized Colab development notebook, with outputs removed.
- `src/`: original notebook source cells, separated by function for easier review. **These are Colab-oriented references, not guaranteed standalone CLI applications.**
- `dashboard/`: original V5 Gradio source.
- `docs/`: project notes and screenshots (add screenshots after removing private information).

## Quick start (Google Colab)
1. Open `notebooks/NBA_Data_Science_Analytics_V4_sanitized.ipynb` in Colab.
2. Install dependencies from `requirements.txt` as appropriate for the active runtime.
3. Mount your own Google Drive and configure `NBA_Data_Science` working directory. Many notebook cells contain Colab-specific paths.
4. Collect your own data or provide your personal local SQLite database. **The database and pretrained model artifacts are intentionally not distributed.**
5. Run the desired cells in dependency order. The notebook records iterative development and is not a one-click production deployment.

## ML and evaluation
Models target numerical outputs; the Probability Engine estimates threshold event probabilities (e.g. P(PTS >= 25)) from historical forecast errors. Performance should be assessed against **pregame** predictions only, with chronological splits and calibration reports. An outcome matching one prediction does not validate the overall model.

## Privacy and data licensing
This repository excludes raw NBA data, SQLite databases, model binaries, Colab-generated outputs, credentials and logs. Verify ESPN/NBA terms before redistributing any scraped or downloaded data. Do not commit `.env`, tokens, personal Drive paths containing sensitive details, or credential-bearing notebook outputs.

## Limitations
- Colab sessions and temporary Gradio public links are not always-on hosting.
- Source code contains historical Google Drive paths that must be adapted.
- Some feature pipelines rely on previous notebook state and are not independently packaged.
- Injury data is informational, not guaranteed to reflect a player's official availability.
- Probability calibration and automated refresh require further independent validation.

## Technologies
Python · pandas · NumPy · scikit-learn · SQLite · XGBoost · LightGBM · Plotly · Gradio · Google Colab

## License
No open-source license selected yet. All rights reserved unless the repository owner adds a license.
