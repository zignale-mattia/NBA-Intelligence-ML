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
