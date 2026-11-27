# NBA Intelligence — REAL NBA Google Colab guide

## Launch with real ESPN game statistics

[![Open real NBA demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zignale-mattia/NBA-Intelligence-ML/blob/main/notebooks/NBA_Intelligence_Public_Demo.ipynb)

1. Open the badge above using Google Colab.
2. Verify the notebook title is **NBA_Intelligence_Public_Demo.ipynb**, NOT the
   separate V4 research notebook.
3. Choose **Runtime → Run all**. The installer fetches the latest GitHub
   source and installs Gradio and scikit-learn.
4. Restart the Python runtime and run the cells again if Colab prompts you
   to do so after a NumPy/Pandas upgrade. This is common on first launch.
5. The second code cell fetches actual ESPN 2025-26 and 2026-27 regular-season
   histories for 14 forecasted athletes, **plus up to 180 recently completed
   NBA games and all participating players' box-score lines**. It stores two
   local SQLite databases, retrieves scheduled upcoming fixtures and trains
   three regression-model candidates. Hundreds of ESPN requests can take time.
6. The third cell opens the redesigned **nine-section research terminal**
   (league rankings, player scouting, game center, team reports, comparison,
   data explorer, forecast studio and model audit). It starts Gradio and prints
   a temporary
   `https://<random>.gradio.live` URL. Open the URL to view the real game
   histories, source URLs, current scheduled matchups and ML predictions.
   Keep Colab open while using the dashboard.

**No fabricated game values or players are used in this demo.** The historical
statistics are actual reported played NBA games retrieved **at runtime**
from ESPN. If ESPN is blocked, times out, or changes its undocumented JSON
format, fetching can fail or yield incomplete data. The code reports errors
rather than substituting synthetic statistics. At least five players and
150 played games are required.

**What is being predicted?** PTS, REB, AST, made three-pointers (3PM), and
PTS+REB+AST (PRA) for an NBA player's *next team fixture*. ESPN schedule
information is used for the time, opponent, home/away status and rest days.
If no actual upcoming team game is retrieved, the display explicitly labels
the estimate as a hypothetical home/rest scenario. The model currently does
**not** incorporate opponent defensive quality, player injuries or lineups.
A listed team fixture does **not** establish a player's availability.

**How the model works:** The public demo builds each prediction from only
games *before* the corresponding game. Features include rolling last-5,
last-10, and last-15 averages, minutes and prior trend, home status and days
rest. It creates a *globally chronological NBA date* split (~70% train, 15%
validation, 15% final test). Ridge, Random Forest and Extra Trees candidates
are selected using validation MAE. The chosen candidate is refitted on
train+validation and compared against a last-10-games baseline on the
previously untouched test dates. If the model underperforms the baseline,
the dashboard displays that result transparently.

**Probability estimate:** The threshold estimate is an empirical fraction
of validation residuals. It is **not a formally calibrated probability**
or a guarantee about a future game. Predictive results depend on the current
data snapshot and may change when ESPN publishes additional game logs.

**Data privacy:** The featured-player SQLite database is created at
`/content/NBA_Real_Data/nba_public_real_espn.sqlite`. The public repository
does not ship the author's personal database, trained models, credentials
or output files. No Google Drive mount is used. A public Gradio share URL
exists only while Colab is running, and should not be used to upload private
files. ESPN sources and box-score links are visible in the app. Check terms
of use before republishing raw datasets or third-party imagery.

## Full original research workflow (separate)

[Open the 14-stage research notebook](https://colab.research.google.com/github/zignale-mattia/NBA-Intelligence-ML/blob/main/notebooks/NBA_Data_Science_Analytics_V4_sanitized.ipynb)

The full original notebook includes ESPN collection, advanced feature
engineering, Ridge/Random Forest/ExtraTrees/XGBoost/LightGBM ensemble
experiments, injury/lineup context and V3 probability engines. It requires
running its stages in dependency order and is **not yet verified as a
clean-room one-click workflow**. The compact, real-data Colab demo is an
independent **newly trained** project version. It does not claim to execute
the original complete V3 pipeline.

## Test scope

Offline unit tests cover parser schema, source links, model construction,
no feature use of a game's own target, temporal splitting, prediction
bounds and database isolation. Real ESPN acquisition and basic forecasting
were also smoke-tested on the owner's Mac. A fresh Colab browser/Gradio
launch should be verified separately: Google Colab runtime versions and
ESPN service behavior may change after publication.

## Expanded public research terminal (November 2026)

The new league data source independently downloads ESPN NBA scoreboards over
the previous 42 calendar days and retrieves full player box scores for up to
the latest **180 completed regular-season games**. A live test on November 27,
2026 yielded **180 games, 3,993 athlete performances, 460 distinct players
and 30 teams**. These are actual published ESPN figures, not synthetic data.
As the rolling window advances or ESPN coverage changes, counts will differ.

League-wide records are also stored in a separate disposable SQLite database
at /content/NBA_Real_Data/nba_league_real_espn.sqlite. The featured players'
historical database remains separate. The combined dashboard offers nine tabs:
Overview, Forecast Studio, Player Intelligence, Compare Players, League
Leaders, Game Center, Team Analytics, Data Explorer and ML Model Audit.

All 460 recent players can be explored statistically, but only the 14
featured players have trained next-game forecasting models. NBA sample
win percentages and scoring leaders are **not** official full-season
standings. Fixture details do not verify individual playing availability.

The full new Gradio interface was locally launched and rendered in Safari
against actual ESPN data. A first-time Google Colab runtime may need a Python
restart after dependency installation. The temporary Gradio public link and
its model results require a Colab session that is still active.
