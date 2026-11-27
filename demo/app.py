"""Gradio presentation of the synthetic-only NBA Intelligence portfolio demo."""
from __future__ import annotations

from html import escape

from .public_demo import METRICS, PROFILES, forecast


def _trend_chart(values: list[float], forecast_value: float) -> str:
    """Return a self-contained accessible SVG; no web image or CDN required."""
    values = list(values) + [forecast_value]
    min_val = min(0.0, min(values) - 2)
    max_val = max(values) + 3
    width, height = 680, 175
    def coords(i, v):
        return (round(20 + i * (width - 40) / (len(values) - 1), 1),
                round(150 - (v - min_val) / (max_val - min_val) * 128, 1))
    points = [coords(i, v) for i, v in enumerate(values)]
    history = " ".join(f"{x},{y}" for x, y in points[:-1])
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    circles = "".join(
        f'<circle cx="{x}" cy="{y}" r="3" fill="#3b82f6"/>'
        for x, y in points[:-1]
    )
    return (
        '<div role="img" aria-label="Simulated last ten games in blue; '
        'next simulated forecast in amber">'
        '<svg viewBox="0 0 680 175" style="width:100%;height:auto;'
        'background:#101827;border-radius:10px">'
        f'<polyline points="{history}" fill="none" stroke="#60a5fa" '
        'stroke-width="2.5" stroke-linecap="round"/>'
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" '
        'stroke="#f59e0b" stroke-width="2.5" stroke-dasharray="6,5"/>'
        f'{circles}'
        f'<circle cx="{x1}" cy="{y1}" r="5" fill="#f59e0b"/>'
        '<text x="20" y="169" fill="#cbd5e1" font-size="11">'
        'Past 10 simulated games → next simulated forecast</text>'
        '</svg></div>'
    )


def make_app(state: dict):
    """Construct the dashboard; importing Gradio is optional for offline tests."""
    import gradio as gr

    models = state["models"]
    games = state["games"]
    rows = [
        [stat, round(models[stat]["test_mae"], 2),
         round(models[stat]["baseline_mae"], 2),
         models[stat]["train_n"], models[stat]["val_n"],
         models[stat]["test_n"]]
        for stat in METRICS
    ]

    def evaluate(player, stat, threshold, opponent_defense, is_home, days_rest):
        result = forecast(
            games, models, player, stat, float(threshold),
            float(opponent_defense), bool(is_home), int(days_rest),
        )
        pct = 100 * result["over_probability"]
        report = (
            f"### {escape(player)} — simulated next-game forecast\n"
            f"**{escape(stat)}:** {result['forecast']:.2f}  ·  "
            f"**Probability of ≥ {result['threshold']:g}:** {pct:.1f}%\n\n"
            f"Probability is the fraction of **{result['samples_calibration']} "
            "synthetic validation residuals** that meet the threshold. "
            "**Not calibrated for real games, betting, or financial decisions.**"
        )
        return report, _trend_chart(
            result["recent_values"], result["forecast"]
        )

    first_player = PROFILES[0][0]
    initial_text, initial_chart = evaluate(first_player, "PTS", 20, 0, True, 2)

    with gr.Blocks(
        title="NBA Intelligence | Synthetic Colab Demo",
        theme=gr.themes.Soft(),
    ) as ui:
        gr.Markdown(
            "# NBA Intelligence — Interactive ML Portfolio Demo\n"
            "**SYNTHETIC DATA ONLY · NOT LIVE NBA PREDICTIONS**\n\n"
            "This independent Colab demo demonstrates a temporal forecasting "
            "workflow, a regularized regression model, threshold estimates, "
            "SQLite storage, and a Gradio dashboard. All players and results "
            "are fabricated with a fixed seed; no credentials or private "
            "Google Drive are involved."
        )
        with gr.Tab("Player forecast"):
            with gr.Row():
                player = gr.Dropdown(
                    choices=[p[0] for p in PROFILES],
                    value=first_player, label="Fabricated player",
                )
                stat = gr.Dropdown(
                    choices=list(METRICS), value="PTS", label="Statistic",
                )
                threshold = gr.Number(
                    value=20, label="Threshold (≥)", precision=1,
                )
            with gr.Row():
                defense = gr.Slider(
                    minimum=-1.0, maximum=1.0, value=0.0,
                    step=0.1, label="Synthetic opponent defense (+ = tougher)",
                )
                is_home = gr.Checkbox(value=True, label="Home game")
                rest = gr.Slider(
                    minimum=1, maximum=7, value=2, step=1,
                    label="Days of rest",
                )
            recalculate = gr.Button("Compute simulated forecast", variant="primary")
            report = gr.Markdown(initial_text)
            trend = gr.HTML(initial_chart)
            recalculate.click(
                fn=evaluate,
                inputs=[player, stat, threshold, defense, is_home, rest],
                outputs=[report, trend],
            )

        with gr.Tab("Model evaluation"):
            gr.Markdown(
                "### Chronological holdout results (synthetic records)\n"
                "For each simulated player: **70% training / 15% validation / "
                "15% testing** of rows after 10 warm-up games. "
                "The comparison is against a last-10-game average baseline. "
                "These are not measured NBA forecasting results."
            )
            gr.Dataframe(
                headers=[
                    "Metric", "Ridge test MAE", "Last-10 test MAE",
                    "Train samples", "Validation samples", "Test samples",
                ],
                value=rows,
                interactive=False,
                label="Synthetic out-of-sample evaluation",
            )
            gr.Markdown(
                "The Python-standard-library Ridge regression is intentionally "
                "small and transparent. The original research project contains "
                "more complex real-data stages (scikit-learn, XGBoost, LightGBM), "
                "which are **not executed by this quick demonstration**."
            )

        with gr.Tab("Data and privacy"):
            gr.Markdown(
                f"**Dataset:** {len(games)} invented box-score rows\n\n"
                f"**Random seed:** {state['seed']}\n\n"
                f"**SQLite path (temporary Colab runtime):** "
                f"`{state['database_path']}`\n\n"
                "No ESPN requests, NBA trademarks/logos/assets, Google Drive "
                "mounts, private databases, token values, or original ML "
                "artifacts are needed. Colab storage disappears when the runtime "
                "is reset. A temporary Gradio share URL may be publicly reachable; "
                "do not upload private files to this demo."
            )

    return ui
