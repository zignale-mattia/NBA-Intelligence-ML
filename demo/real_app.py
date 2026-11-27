"""Interactive Gradio display of actual ESPN NBA box scores and real-trained models."""
from __future__ import annotations

from datetime import date
from html import escape

from .real_espn import STATS
from .real_models import real_forecast


def _svg_history(values, predicted):
    series = list(values) + [predicted]
    upper = max(series) + 4
    lower = min(0, min(series) - 4)
    coords = [
        (round(24 + i * 60, 1),
         round(135 - (v - lower) / max(1, upper - lower) * 105, 1))
        for i, v in enumerate(series)
    ]
    path = " ".join(f"{x},{y}" for x, y in coords[:-1])
    x0, y0 = coords[-2]
    x1, y1 = coords[-1]
    points = "".join(
        f'<circle cx="{x}" cy="{y}" r="3" fill="#60a5fa"/>'
        for x, y in coords[:-1]
    )
    return (
        '<div role="img" aria-label="Last 10 actual NBA game values in blue; '
        'next game model estimate in amber">'
        '<svg viewBox="0 0 680 170" style="width:100%;max-width:900px;'
        'background:#101827;border-radius:12px">'
        f'<polyline points="{path}" fill="none" stroke="#60a5fa" '
        'stroke-width="2.5" stroke-linejoin="round"/>'
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" '
        'stroke="#f59e0b" stroke-width="2.5" stroke-dasharray="5,5"/>'
        f'{points}'
        f'<circle cx="{x1}" cy="{y1}" r="5" fill="#f59e0b"/>'
        '<text x="20" y="160" fill="#cbd5e1" font-size="12">'
        '10 actual ESPN box scores → model next-game estimate</text>'
        '</svg></div>'
    )


def make_real_app(state: dict):
    import gradio as gr

    games = state["games"]
    histories = state["histories"]
    models = state["models"]
    schedule = state.get("upcoming") or {}
    players = sorted(histories)
    latest = {p: histories[p][-1] for p in players}

    def display_forecast(player, stat, threshold, home_setting, rest_setting):
        if player not in histories or stat not in STATS:
            raise gr.Error("Select a listed NBA player and statistic")
        fixture = schedule.get(player)
        if fixture:
            home = fixture["home"]
            rest = max(1, min(7, (
                date.fromisoformat(fixture["date_utc"][:10]) -
                date.fromisoformat(latest[player]["date"])
            ).days))
        else:
            home = bool(home_setting)
            rest = int(rest_setting)
        result = real_forecast(
            state, player, stat, float(threshold), home, rest,
        )
        last = result["latest"]
        pct = 100 * result["over_estimate"]
        if fixture:
            matchup_text = (
                f"**Verified ESPN scheduled game:** "
                f"{escape(fixture['team'])} "
                f"{'vs' if fixture['home'] else '@'} "
                f"{escape(str(fixture['opponent']))} · "
                f"{fixture['date_utc']} (UTC) "
                f"— [official game listing]({fixture['url']}). "
                f"Venue and rest were derived from this ESPN fixture."
            )
        else:
            matchup_text = (
                "**No upcoming matchup confirmed from ESPN in the next nine "
                "days.** This is a **hypothetical venue/rest scenario**, "
                "not a prediction for a verified fixture."
            )
        text = (
            f"### {escape(player)} — real-data NBA forecast\n\n"
            f"{matchup_text}\n\n"
            f"**Model estimate ({escape(stat)}): {result['prediction']:.2f}** "
            f"· Estimated P({escape(stat)} ≥ {result['threshold']:g}) "
            f"**{pct:.1f}%**\n\n"
            f"Trained on actual ESPN box scores. Model selected using "
            f"historical validation: **{escape(result['model'])}**. "
            f"Threshold percentage comes from "
            f"{result['validation_samples']} held-out residuals; "
            f"**it is not a calibrated guarantee**.\n\n"
            f"Last recorded ESPN game: **{last['date']}** vs "
            f"**{escape(last['opponent'])}** — "
            f"{int(last['PTS'])} PTS, {int(last['REB'])} REB, "
            f"{int(last['AST'])} AST, {int(last['3PM'])} 3PM. "
            f"[Verify box score]({last['source_url']}).\n\n"
            "**Limitations:** opponent defensive strength, injuries and "
            "confirmed playing status are *not* included in this quick model. "
            "This is an educational forecast, not an official game outcome."
        )
        recent = list(reversed(result["last10"]))
        table = [
            [
                game["date"], game["opponent"],
                "Home" if game["home"] else "Away",
                round(game["MIN"], 1),
                int(game["PTS"]), int(game["REB"]), int(game["AST"]),
                int(game["3PM"]), int(game["PRA"]),
                game["source_url"],
            ]
            for game in recent
        ]
        return (
            text,
            _svg_history(
                [g[stat] for g in result["last10"]],
                result["prediction"],
            ),
            table,
        )

    metric_table = [
        [
            stat, models[stat]["selected_model"],
            models[stat]["validation_mae"],
            models[stat]["test_mae"],
            models[stat]["baseline_test_mae"],
            models[stat]["n_train"], models[stat]["n_val"],
            models[stat]["n_test"],
            models[stat]["test_from"], models[stat]["test_through"],
        ]
        for stat in STATS
    ]
    default_player = "Nikola Jokic" if "Nikola Jokic" in histories else players[0]
    first_report, first_svg, first_games = display_forecast(
        default_player, "PTS", 20, True, 2,
    )

    with gr.Blocks(title="NBA Intelligence | Real ESPN Data") as demo:
        gr.Markdown(
            "# NBA Intelligence — Real NBA Analytics & Forecasting\n"
            "**REAL ESPN MATCH DATA · REAL NBA PLAYERS · CHRONOLOGICAL ML "
            "EVALUATION**\n\n"
            f"Data fetched from ESPN: **{escape(state['as_of_utc'])} UTC**. "
            f"**{len(games)} verified played-game logs** from "
            f"**{len(histories)} actual NBA players**, 2025-26 & 2026-27 "
            "regular seasons. The database was created at launch, "
            "without personal Google Drive or private API keys."
        )
        if state.get("failures"):
            gr.Markdown(
                f"**Data availability notice:** "
                f"{len(state['failures'])} ESPN player-season requests "
                "failed. No simulated records were inserted; results use "
                "only successfully retrieved real records."
            )
        with gr.Tab("Next-game estimate"):
            with gr.Row():
                player = gr.Dropdown(
                    choices=players, value=default_player,
                    label="NBA player (real ESPN game logs)",
                )
                stat = gr.Dropdown(
                    choices=list(STATS), value="PTS", label="Statistic",
                )
                threshold = gr.Number(
                    value=20, minimum=0, label="Threshold (≥)",
                )
            with gr.Accordion("Scenario fallback if no verified upcoming match", open=False):
                home = gr.Checkbox(
                    value=True, label="Assumed home game (fallback only)",
                )
                rest = gr.Slider(
                    minimum=1, maximum=7, step=1, value=2,
                    label="Assumed rest days (fallback only)",
                )
            run = gr.Button("Compute forecast from REAL game history",
                            variant="primary")
            prediction = gr.Markdown(first_report)
            chart = gr.HTML(first_svg)
        with gr.Tab("Real game logs & ESPN sources"):
            gr.Markdown(
                "Every displayed row is a played NBA game retrieved from "
                "the ESPN athlete game-log feed. Box-score URLs allow "
                "independent verification. Seasons exclude preseason and "
                "playoffs."
            )
            raw_table = gr.Dataframe(
                value=first_games,
                headers=[
                    "Date", "Opponent", "Venue", "MIN",
                    "PTS", "REB", "AST", "3PM", "PRA", "ESPN box score",
                ],
                interactive=False, label="Last 10 actual games",
            )
        with gr.Tab("Out-of-sample model evaluation"):
            gr.Markdown(
                "### Actual NBA held-out test performance\n"
                "Supervised examples use ONLY earlier player games to "
                "calculate rolling features. All players share global "
                "chronological date splits: first 70% training, next 15% "
                "validation/model selection, last 15% final testing. "
                "Test MAE is compared with the simple last-10-games mean. "
                "Some models may underperform the baseline."
            )
            gr.Dataframe(
                value=metric_table,
                headers=[
                    "Stat", "Chosen model", "Validation MAE",
                    "Test MAE", "Last-10 baseline MAE", "Train",
                    "Validation", "Test", "Test from", "Test through",
                ],
                interactive=False, label="Real observed NBA results",
            )
            gr.Markdown(
                "Model candidates in Colab: Ridge, Random Forest and "
                "Extra Trees when scikit-learn is installed. Models are "
                "selected on validation error and refitted on train plus "
                "validation for the final untouched test period. "
                "These are *new public-demo models*, not the original "
                "multi-stage V3 research model binaries."
            )
        with gr.Tab("Data lineage and limitations"):
            gr.Markdown(
                f"**Primary source:** ESPN athlete regular-season game-log "
                f"JSON endpoint, requested at runtime.\n\n"
                f"**Last fetch:** {state['as_of_utc']} UTC\n\n"
                f"**Actual observed game logs:** {len(games)}\n\n"
                f"**Temporary isolated SQLite file:** "
                f"`{state['db_path']}`\n\n"
                "No fabricated athlete names, game results or box scores; "
                "no author Google Drive, password or secret token. ESPN "
                "data may be delayed or unavailable. The model is "
                "**not** injury-aware and may not know whether a player "
                "will actually play; the ESPN next-matchup listing is "
                "not a confirmation of player availability.\n\n"
                "The full original 14-stage research notebook and the "
                "advanced V3 code are preserved in the repository. "
                "This public Colab workflow is a more portable, smaller "
                "real-data implementation, not a claim that the original "
                "ensemble was retrained end to end."
            )
        run.click(
            display_forecast,
            inputs=[player, stat, threshold, home, rest],
            outputs=[prediction, chart, raw_table],
        )
    return demo
