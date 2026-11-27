# Release checklist

## Completed for this source-only portfolio release

- [x] Published code on the owner's GitHub account.
- [x] Removed notebook execution outputs and discarded experimental backtest stages from the current notebook.
- [x] Added consistent notebook data-path selection with an environment override.
- [x] Preserved the original model and forecast calculation stages rather than rewriting the algorithms.
- [x] Added offline syntax, notebook-structure and path-configuration tests.
- [x] Added a GitHub Actions validation workflow.
- [x] Documented data prerequisites and the correct included dashboard version.
- [x] Excluded raw data, exported reports, model artifacts and secrets from the current tracked files.
- [x] Documented temporary Gradio sharing and private dashboard-password setup.

## Added for the public REAL NBA Colab demo

- [x] Separate REAL ESPN data quickstart notebook with an Open in Colab badge.
- [x] Temporary SQLite of verified ESPN regular-season game logs; no personal Drive and no private data.
- [x] Ridge, Random Forest and Extra Trees candidates, chronological test split and last-ten baseline on observed NBA game logs.
- [x] Real-player Gradio demonstration with ESPN box-score verification links.
- [x] Documented temporary sharing, dependencies, public-vs-full run modes and privacy.
- [x] Extended local offline checks and real ESPN ingestion/model smoke tests.
- [ ] Run new REAL public quickstart from a completely fresh Colab Python runtime.
- [ ] Visually check real Gradio browser launch and interaction.

## Still to validate using the original Colab environment

- [ ] Execute the full pipeline from a clean runtime on a COPY of the private data directory.
- [ ] Confirm compatibility of all forecast and probability output file schemas.
- [ ] Verify prediction-time feature leakage safeguards and calibration using timestamped predictions.
- [ ] Report reproducible accuracy versus baseline with dataset and split dates.
- [ ] Verify injury/lineup reference integration and stale-data handling.
- [ ] Verify the complete V5 dashboard, authentication and launch flow.
- [ ] Capture sanitized screenshots after checking rights and removing private data.
- [ ] Freeze a tested dependency environment from Colab.
- [ ] Verify data licensing before uploading any datasets or external images.
- [ ] Decide whether to choose a software license.
- [ ] If desired, separately plan a history rewrite; old commits still contain experimental revisions.
