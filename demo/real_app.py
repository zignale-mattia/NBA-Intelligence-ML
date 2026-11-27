"""NBA Intelligence Pro — portfolio dashboard with genuine ESPN NBA data.

The league explorer displays broad, independently sourced played-game boxscores.
The forecasting tab uses only the smaller athlete histories for which a
chronologically evaluated public model exists. No fabricated observations.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from html import escape
from pathlib import Path

from .analytics_views import (
    describe, grouped_threshold, player_rows, player_table, match_table,
    standings_table, kpis_html, comparison, METRICS,
)
from .league_explorer import leaderboards, team_performance
from .portfolio_charts import (
    time_series, player_comparison, rank_chart,
    distribution_chart, model_accuracy_chart, team_win_chart,
)
from .real_espn import STATS
from .real_models import real_forecast


def _merged_histories(state):
    """Prefer richer league boxscores; extend 14 featured athletes historically."""
    merged = {}
    for row in state["games"]:
        game_id = str(row.get("game_id") or row.get("event_id"))
        merged[(row["athlete_id"], game_id)] = {
            **row, "game_id": game_id,
        }
    for row in state["league"]["rows"]:
        merged[(row["athlete_id"], row["game_id"])] = row
    by_player = defaultdict(list)
    for row in merged.values():
        by_player[row["player"]].append(row)
    return {
        p: sorted(v,key=lambda x:(x["date"],x["game_id"]))
        for p,v in by_player.items()
    }


def _last_game_markup(row):
    return (
        f'<span class="data-tag">ESPN verified</span> '
        f'{escape(row["date"])} · '
        f'{escape(str(row.get("opponent", "Unknown")))} · '
        f'<a href="{escape(row["source_url"],quote=True)}" target="_blank" '
        f'rel="noopener noreferrer">Box score ↗</a>'
    )


def _section(title,subtitle=""):
    return (
        f'<div class="dash-title">{escape(title)}</div>'
        f'<div class="dash-description">{escape(subtitle)}</div>'
    )


def _player_profile(player,rows,stat="PTS",period=20,venue="All"):
    games=player_rows(rows,player,venue=venue,last_n=int(period))
    if not games:
        return (
            '<div class="note-amber">No ESPN game logs match this filter.</div>',
            time_series([],stat),
            distribution_chart([],stat,player),
            [],"No selected records",
        )
    cards=[]
    for label,metric in (
        ("AVG POINTS","PTS"),("AVG REBOUNDS","REB"),
        ("AVG ASSISTS","AST"),("AVG THREES","3PM"),
        ("AVG PRA","PRA"),
    ):
        profile=describe(games,metric)
        cards.append((
            label,
            f'{profile["avg"]:.1f}' if profile["avg"] is not None else "—",
            f'{profile["games"]} recorded games',
            "teal" if metric=="PTS" else "blue",
        ))
    markup=(
        f'<div class="section-card"><div class="section-title">'
        f'{escape(player)} · {len(games)} played games</div>'
        f'<div class="section-note">Source-linked NBA box scores from '
        f'{escape(games[0]["date"])} to {escape(games[-1]["date"])}. '
        f'Includes both recent league-wide and historical featured-player '
        f'game logs when available. All statistics are observed; availability '
        f'varies with the ESPN acquisition window.</div></div>'
        + kpis_html(cards)
    )
    headers,data=player_table(games,limit=200)
    return markup,time_series(games,stat,last_n=35),distribution_chart(games,stat,player),data,(
        f"Game log · {len(games)} records · {stat}"
    )


def _forecast_html(state,player,stat,threshold,home_setting,rest_setting):
    histories=state["histories"]
    if player not in histories or stat not in STATS:
        raise ValueError("Unknown modeled player or statistic")
    fixture=state.get("upcoming",{}).get(player)
    latest=histories[player][-1]
    if fixture:
        home=bool(fixture["home"])
        rest=max(1,min(7,(
            date.fromisoformat(fixture["date_utc"][:10])-
            date.fromisoformat(latest["date"])
        ).days))
    else:
        home=bool(home_setting)
        rest=int(rest_setting)
    forecast=real_forecast(state,player,stat,float(threshold),home,rest)
    last10=forecast["last10"]
    observed10=describe(last10,stat)
    observed20=describe(histories[player][-20:],stat)
    historical_hit=grouped_threshold(histories[player][-20:],stat,threshold)
    model=state["models"][stat]
    fixture_html=(
        f'<strong>Next ESPN team fixture</strong> · '
        f'{escape(fixture["team"])} '
        f'{"vs" if fixture["home"] else "@"} '
        f'{escape(str(fixture["opponent"]))}<br>'
        f'{escape(fixture["date_utc"])} UTC · '
        f'<a href="{escape(fixture["url"],quote=True)}" '
        f'target="_blank" rel="noopener noreferrer">Verify schedule ↗</a>'
        if fixture else
        '<strong>Hypothetical next-game scenario.</strong> No team fixture '
        'was found in the nine-day ESPN schedule window. Venue/rest inputs '
        'are assumed values, not a verified future matchup.'
    )
    score_cards=(
        '<div class="forecast-headline">'
        '<div class="forecast-hero">'
        f'<div class="forecast-label">{escape(player)} · predicted {escape(stat)}</div>'
        f'<div class="forecast-big">{forecast["prediction"]:.1f} '
        f'<small>{escape(stat)}</small></div>'
        f'<div class="forecast-details">Selected historical model: '
        f'<strong>{escape(forecast["model"])}</strong> · '
        f'Test MAE: {model["test_mae"]:.2f} · '
        f'Baseline MAE: {model["baseline_test_mae"]:.2f}</div>'
        '</div><div class="forecast-box">'
        f'<div class="forecast-label">Estimated P({escape(stat)} ≥ '
        f'{float(threshold):g})</div>'
        f'<div class="forecast-big">{forecast["over_estimate"]*100:.1f}%</div>'
        f'<div class="forecast-details">Empirical validation residual estimate '
        f'from {forecast["validation_samples"]} observations. '
        'Not a calibrated probability or guarantee.</div>'
        '</div></div>'
    )
    meta=(
        '<div class="section-card"><div class="section-title">'
        f'{fixture_html}</div>'
        '<div class="metric-stack">'
        f'<div class="metric-tag"><b>{observed10["avg"]:.1f}</b>'
        f'<span>Actual last 10 avg {stat}</span></div>'
        f'<div class="metric-tag"><b>{observed20["avg"]:.1f}</b>'
        f'<span>Actual last 20 avg {stat}</span></div>'
        f'<div class="metric-tag"><b>{historical_hit:.1f}%</b>'
        f'<span>Last 20 actual threshold hit %</span></div>'
        f'<div class="metric-tag"><b>{model["n_test"]}</b>'
        '<span>Out-of-sample test records</span></div>'
        '</div>'
        '<div class="metadata-line">'
        f'Last observed NBA game: {_last_game_markup(latest)}. '
        'These estimates exclude injury status, confirmed lineups and opponent '
        'defensive strength; a listed team fixture is not confirmation '
        'that this athlete will participate.'
        '</div></div>'
    )
    headers,data=player_table([
        {**row,"game_id":row.get("game_id",row.get("event_id",""))}
        for row in histories[player]
    ],limit=100)
    return (score_cards+meta,
            time_series(histories[player],stat,
                        f"{player} · 35 actual games + next-game ML estimate",
                        model_estimate=forecast["prediction"],last_n=35),
            data)


def _team_summary(team,league,team_rows):
    matches=[m for m in league["matches"] if m["home"]==team or m["away"]==team]
    home_games=sum(m["home"]==team for m in matches)
    wins=sum((m["home"]==team and m["home_score"]>m["away_score"]) or
             (m["away"]==team and m["away_score"]>m["home_score"])
             for m in matches)
    n=len(matches)
    points_for=sum(m["home_score"] if m["home"]==team else m["away_score"]
                   for m in matches)
    points_against=sum(m["away_score"] if m["home"]==team else m["home_score"]
                       for m in matches)
    players=[r for r in team_rows if r["team"]==team]
    cards=kpis_html([
        ("SAMPLE WINS",wins,f"{n} ESPN games","teal"),
        ("SAMPLE LOSSES",n-wins,f"{home_games} home games","blue"),
        ("WIN RATE",f"{wins/n*100:.1f}%" if n else "—","sample, not NBA standings","teal"),
        ("PTS FOR/G",f"{points_for/n:.1f}" if n else "—","sampled games","blue"),
        ("PTS AGAINST/G",f"{points_against/n:.1f}" if n else "—","sampled games","amber"),
    ])
    return (
        cards,
        match_table(matches,limit=80),
        [[p["player"],p["games"],round(p["mean"],1),round(p["high"],0),
          round(p["total"],0)]
         for p in leaderboards(players,"PTS",min_games=2)[:25]],
    )


def make_real_app(state: dict):
    """Create the full interactive recruiter dashboard from verified ESPN rows."""
    import gradio as gr

    if not state.get("league") or not state["league"].get("rows"):
        raise RuntimeError(
            "Run ESPN league-wide box-score collection before the dashboard. "
            "No simulated or empty league fallback is allowed."
        )
    league=state["league"]
    games=league["matches"]
    league_rows=league["rows"]
    all_histories=_merged_histories(state)
    all_names=sorted(all_histories)
    observed=list(row for records in all_histories.values() for row in records)
    model_names=sorted(state["histories"])
    team_names=sorted({r["team"] for r in league_rows})
    team_stats=team_performance(games)
    default_model="Nikola Jokic" if "Nikola Jokic" in model_names else model_names[0]
    default_player=default_model if default_model in all_histories else all_names[0]
    default_compare=next((p for p in ("Shai Gilgeous-Alexander","Jalen Brunson")
                         if p in all_histories and p != default_player),all_names[1])
    style=Path(__file__).with_name("portfolio_theme.css").read_text(encoding="utf-8")
    leader_initial=leaderboards(league_rows,"PTS",min_games=3)
    total_games=len(games)
    unique_players=len({r["athlete_id"] for r in league_rows})
    unique_teams=len(team_names)
    kpis=kpis_html([
        ("REAL BOXSCORES",f"{len(league_rows):,}","player performances · 42-day snapshot","teal"),
        ("NBA GAMES",f"{total_games:,}","completed official ESPN games","blue"),
        ("NBA PLAYERS",f"{unique_players:,}","distinct athletes · recent sample","teal"),
        ("NBA TEAMS",f"{unique_teams}","all teams represented","blue"),
        ("ML FORECASTS",f"{len(model_names)}","players with trained models","amber"),
    ])

    def leaderboard_outputs(stat,min_games):
        results=leaderboards(league_rows,stat,int(min_games))
        rows=[[i+1,r["player"],r["team"],r["games"],
               round(r["mean"],2),round(r["total"],0),round(r["high"],0)]
              for i,r in enumerate(results[:35])]
        return rank_chart(results,stat,limit=15),rows

    def profile_outputs(player,stat,period,venue):
        if player not in all_histories:
            raise gr.Error("Select a player from the current ESPN dataset")
        _,chart,dist,table,_label=_player_profile(
            player,all_histories[player],stat,int(period),venue
        )
        profile,_,_,_,_= _player_profile(
            player,all_histories[player],stat,int(period),venue
        )
        return profile,chart,dist,table

    def forecast_outputs(player,stat,threshold,venue,rest):
        try:
            return _forecast_html(state,player,stat,threshold,venue,rest)
        except (ValueError,TypeError,KeyError) as exc:
            raise gr.Error(str(exc))

    def compare_outputs(first,second,stat,limit):
        names=list(dict.fromkeys([first,second]))
        if len(names)<2:
            raise gr.Error("Select two different NBA players")
        data=comparison(observed,names,stat,limit=int(limit))
        rows=[[r["player"],r["games"],r["avg"],
               r["median"],r["high"],r["std"],r["last_game"]]
              for r in data]
        return player_comparison(all_histories,stat,names,last_n=int(limit)),rows

    def game_scoreboard(event_id):
        match=next((g for g in games if g["game_id"]==event_id),None)
        if not match:
            return '<div class="note-amber">Select a verified NBA game.</div>',[]
        home=escape(match["home"])
        away=escape(match["away"])
        home_score=match["home_score"]
        away_score=match["away_score"]
        box=(
            '<div class="section-card"><div class="section-title">'
            f'{away} <b>{away_score}</b> &nbsp; — &nbsp; '
            f'<b>{home_score}</b> {home}</div>'
            f'<div class="section-note">{escape(match["date"])} · '
            f'{home_score+away_score} total points · '
            f'<a href="{escape(match["source_url"],quote=True)}" target="_blank" '
            f'rel="noopener noreferrer">ESPN official box score ↗</a></div></div>'
        )
        rows=sorted(
            (r for r in league_rows if r["game_id"]==event_id),
            key=lambda r:(-r["PTS"],r["player"])
        )
        return box,[
            [r["player"],r["team"],"Yes" if r["starter"] else "No",
             r["MIN"],r["PTS"],r["REB"],r["AST"],r["3PM"],
             r["FGM"],r["FGA"],r["3PA"],r["FTM"],r["FTA"],
             r["STL"],r["BLK"],r["TOV"],r["PRA"]]
            for r in rows
        ]

    def game_choices(team):
        filtered=[m for m in games if team=="All teams"
                  or m["home"]==team or m["away"]==team]
        return [
            (f'{m["date"]} · {m["away_abbr"]} {m["away_score"]} '
             f'@ {m["home_abbr"]} {m["home_score"]}',m["game_id"])
            for m in sorted(filtered,key=lambda g:g["date"],reverse=True)
        ]

    def filter_games(team):
        matches=[m for m in games if team=="All teams"
                 or m["home"]==team or m["away"]==team]
        choices=game_choices(team)
        selected=choices[0][1] if choices else None
        box,players=game_scoreboard(selected)
        return (match_table(matches,limit=250),
                gr.update(choices=choices,value=selected),
                box,players)

    def team_outputs(team):
        return _team_summary(team,league,league_rows)

    def search_rows(name,team,stat,limit):
        search=str(name or "").lower().strip()
        selected=[
            r for r in observed
            if (not search or search in r["player"].lower()
                or search in r.get("opponent","").lower())
            and (team=="All teams" or r["team"]==team)
        ]
        headers,table=player_table(selected,limit=int(limit))
        summ=describe(selected,stat)
        headline=(
            f'<div class="section-card"><div class="section-title">'
            f'{len(selected):,} matching real ESPN records · '
            f'{len({r["player"] for r in selected})} distinct players'
            f'</div><div class="section-note">'
            f'Average {escape(stat)}: '
            f'<strong>{summ["avg"] if summ["avg"] is not None else "—"}</strong>'
            f' · high: <strong>{summ["high"] if summ["high"] is not None else "—"}</strong>'
            f' · showing up to {limit} source-linked rows.'
            '</div></div>'
        )
        return headline,table

    first_game=sorted(games,key=lambda g:(g["date"],g["game_id"]))[-1]
    first_box,first_players=game_scoreboard(first_game["game_id"])
    f_html,f_plot,f_logs=_forecast_html(
        state,default_model,"PTS",20,True,2
    )
    p_html,p_plot,p_dist,p_logs,_=_player_profile(
        default_player,all_histories[default_player],"PTS",50,"All"
    )
    comp_plot,comp_data=compare_outputs(
        default_player,default_compare,"PTS",20
    )
    initial_teams=_team_summary(team_names[0],league,league_rows)
    explorer_html,explorer_rows=search_rows(
        "", "All teams","PTS",100
    )

    with gr.Blocks(
        css=style,
        theme=gr.themes.Base(primary_hue="teal",neutral_hue="slate"),
        title="NBA Intelligence | Analytics Research Terminal",
        analytics_enabled=False,
    ) as app:
        gr.HTML(
            '<div id="portfolio-shell"><div class="nba-hero">'
            '<div class="hero-eyebrow"><i class="signal-dot"></i>'
            'RESEARCH TERMINAL · ESPN NBA DATA</div>'
            '<div class="hero-title">NBA <span>INTELLIGENCE</span><br>'
            'Analytics & Forecast Studio</div>'
            '<div class="hero-sub">An interactive basketball research portfolio '
            'connecting official ESPN box scores, league-wide scouting, '
            'player benchmarking and chronologically evaluated machine learning. '
            'Real games. Transparent methods. Source-linked results.</div>'
            '<div class="hero-line">'
            f'<span class="hero-pill"><b>{len(league_rows):,}</b> real player performances</span>'
            f'<span class="hero-pill"><b>{unique_players}</b> NBA players</span>'
            f'<span class="hero-pill"><b>{total_games}</b> completed games</span>'
            '<span class="hero-pill"><b>ESPN</b> source verified</span>'
            '</div>'
            f'<span class="hero-badge">DATA SNAPSHOT · {escape(league["as_of_utc"][:16])} UTC</span>'
            '</div></div>'
        )
        gr.HTML(kpis)
        if league.get("warnings") or state.get("failures"):
            gr.HTML(
                '<div class="note-amber">ESPN data availability: '
                f'{len(league.get("warnings",[]))} league requests and '
                f'{len(state.get("failures",[]))} featured-player requests '
                'were incomplete. Only returned, verified real game data is displayed.'
                '</div>'
            )

        with gr.Tabs():
            with gr.Tab("Overview"):
                gr.HTML(_section(
                    "The NBA, beyond a single prediction",
                    "Recent league-wide sample, real scoring leaders and completed matchups. "
                    "These are NOT full-season league standings."
                ))
                with gr.Row():
                    with gr.Column(scale=6):
                        gr.Plot(
                            value=rank_chart(leader_initial,"PTS",limit=15),
                            label="Top NBA scorers · recent sample",
                        )
                    with gr.Column(scale=5):
                        gr.Plot(
                            value=team_win_chart(team_stats),
                            label="League teams · sample record",
                        )
                gr.HTML(_section(
                    "Latest completed NBA games",
                    "Scores and results collected live from ESPN during this notebook launch."
                ))
                gr.Dataframe(
                    value=match_table(games,limit=25),
                    headers=["Date","Away team","Away PTS","Home PTS",
                             "Home team","Total","Margin","ESPN box score"],
                    interactive=False,label="25 most recent actual NBA games",
                )
                gr.HTML(_section(
                    "Recent player performances",
                    "Actual per-game points, rebounds, assists, shooting, steals and blocks — each row has an ESPN source."
                ))
                gr.Dataframe(
                    value=player_table(league_rows,limit=80)[1],
                    headers=player_table(league_rows,limit=1)[0],
                    interactive=False,label="80 latest played-game performances",
                )

            with gr.Tab("Forecast Studio"):
                gr.HTML(_section(
                    "Forecast Studio · data-driven next-game estimates",
                    "Modeled players have real historical ESPN logs. "
                    "Forecasts are new model estimates, not actual future results."
                ))
                with gr.Row():
                    modeled=gr.Dropdown(
                        choices=model_names,value=default_model,
                        label="NBA player · chronologically trained ML",
                    )
                    model_stat=gr.Dropdown(
                        choices=list(STATS),value="PTS",
                        label="Prediction metric",
                    )
                    thresh=gr.Number(
                        value=20,minimum=0,label="Threshold (≥)",
                    )
                with gr.Accordion(
                    "Fallback settings · only used if no scheduled ESPN matchup is found",
                    open=False,
                ):
                    fallback_home=gr.Checkbox(
                        label="Assume home court",value=True
                    )
                    fallback_rest=gr.Slider(
                        minimum=1,maximum=7,step=1,value=2,
                        label="Assumed days since last played game",
                    )
                forecast_button=gr.Button(
                    "Run real-data forecast",variant="primary"
                )
                forecast_card=gr.HTML(f_html)
                forecast_plot=gr.Plot(value=f_plot,label="Player trend and forecast")
                gr.HTML(_section(
                    "Historical evidence behind the forecast",
                    "Real game records from the modeled athlete's historical ESPN feed."
                ))
                forecast_table=gr.Dataframe(
                    value=f_logs,
                    headers=player_table(league_rows,limit=1)[0],
                    interactive=False,label="Last 100 real ESPN game logs (where available)",
                )

            with gr.Tab("Player Intelligence"):
                gr.HTML(_section(
                    "Player Intelligence · granular scouting",
                    "Explore every NBA athlete in the retrieved league sample, "
                    "plus longer histories for the 14 featured ML players."
                ))
                with gr.Row():
                    player_selector=gr.Dropdown(
                        choices=all_names,value=default_player,
                        label="Select NBA player",filterable=True,
                    )
                    player_stat=gr.Dropdown(
                        choices=list(METRICS),value="PTS",
                        label="Analyse metric",
                    )
                    player_period=gr.Dropdown(
                        choices=[("Last 5",5),("Last 10",10),
                                 ("Last 20",20),("Last 50",50),
                                 ("All available",500)],
                        value=50,label="Sample window",
                    )
                    player_venue=gr.Dropdown(
                        choices=["All","Home","Away"],
                        value="All",label="Game location",
                    )
                player_btn=gr.Button("Explore real player statistics",variant="primary")
                player_cards=gr.HTML(p_html)
                with gr.Row():
                    with gr.Column(scale=6):
                        player_plot=gr.Plot(value=p_plot)
                    with gr.Column(scale=5):
                        player_dist=gr.Plot(value=p_dist)
                gr.HTML(_section(
                    "Game-by-game detail",
                    "Shot attempts, made threes, defensive counting stats and original ESPN links where available."
                ))
                player_game_table=gr.Dataframe(
                    value=p_logs,headers=player_table(league_rows,limit=1)[0],
                    interactive=False,label="Real player performance log",
                )

            with gr.Tab("Compare Players"):
                gr.HTML(_section(
                    "Head-to-head player benchmarking",
                    "Side-by-side performance and consistency using actual NBA game statistics."
                ))
                with gr.Row():
                    player_a=gr.Dropdown(
                        choices=all_names,value=default_player,
                        label="Player A",filterable=True,
                    )
                    player_b=gr.Dropdown(
                        choices=all_names,value=default_compare,
                        label="Player B",filterable=True,
                    )
                    compare_stat=gr.Dropdown(
                        choices=list(METRICS),value="PTS",
                        label="Statistic",
                    )
                    compare_window=gr.Dropdown(
                        choices=[("Last 5 games",5),("Last 10",10),
                                 ("Last 20",20),("Last 40",40)],
                        value=20,label="Sample",
                    )
                compare_button=gr.Button("Compare verified NBA game logs",variant="primary")
                compare_chart=gr.Plot(value=comp_plot)
                compare_table=gr.Dataframe(
                    value=comp_data,
                    headers=["Player","Games in sample","Average",
                             "Median","Season high in sample",
                             "Standard deviation","Most recent date"],
                    interactive=False,
                    label="Peer comparison · actual historical box scores",
                )
                gr.HTML(
                    '<div class="note-teal">Game index on this chart is each player’s '
                    'own sequence of recent played games, not necessarily matching calendar '
                    'dates or opponents. Different players may have different sample sizes.</div>'
                )

            with gr.Tab("League Leaders"):
                gr.HTML(_section(
                    "League leaderboard · ESPN observed statistics",
                    "Rank 400+ players by actual per-game averages in the recent games "
                    "retrieved for this notebook session. Configure minimum games to "
                    "avoid misleading one-game outliers."
                ))
                with gr.Row():
                    leader_metric=gr.Dropdown(
                        choices=list(METRICS),value="PTS",
                        label="Statistic for player ranking",
                    )
                    leader_min_games=gr.Slider(
                        minimum=1,maximum=10,value=3,step=1,
                        label="Minimum played games",
                    )
                leader_run=gr.Button("Update NBA leaderboard",variant="primary")
                with gr.Row():
                    with gr.Column(scale=6):
                        rank_plot=gr.Plot(
                            value=rank_chart(leader_initial,"PTS",limit=15),
                        )
                    with gr.Column(scale=5):
                        gr.HTML(
                            '<div class="section-card"><div class="section-title">'
                            'Sample-aware league ranking</div>'
                            '<div class="section-note">'
                            'League-wide box scores cover the last 42 calendar '
                            'days at launch, capped to the latest 180 completed '
                            'games. Every bar is an average of played NBA games '
                            'inside that observed sample, not an official season '
                            'leaderboard. Use a higher minimum game count '
                            'for a more stable comparison.</div></div>'
                        )
                        gr.HTML(
                            '<div class="note-teal">The dashboard never generates '
                            'fictional NBA scores or fills missing ESPN values '
                            'with estimates in observed-statistics tables.</div>'
                        )
                rank_table=gr.Dataframe(
                    value=leaderboard_outputs("PTS",3)[1],
                    headers=["Rank","Player","Team","Games",
                             "Average","Total","Single-game high"],
                    interactive=False,label="Top 35 real NBA performers in this sample",
                )

            with gr.Tab("Game Center"):
                gr.HTML(_section(
                    "Game Center · verified NBA results and box scores",
                    "Explore completed NBA matchups, team scores and full "
                    "player box-score statistics. Every game links to ESPN."
                ))
                with gr.Row():
                    game_team=gr.Dropdown(
                        choices=["All teams"]+team_names,value="All teams",
                        label="Filter completed games by team",
                    )
                    game_selector=gr.Dropdown(
                        choices=game_choices("All teams"),
                        value=first_game["game_id"],
                        label="Open official game box score",
                        filterable=True,
                    )
                game_score=gr.HTML(first_box)
                gr.HTML(_section(
                    "Full player box score",
                    "Observed player minutes, starting status, scoring, rebounding, "
                    "assists, field goals, free throws, turnovers and blocks."
                ))
                game_player_table=gr.Dataframe(
                    value=first_players,
                    headers=["Player","Team","Starter","MIN","PTS",
                             "REB","AST","3PM","FGM","FGA","3PA",
                             "FTM","FTA","STL","BLK","TO","PRA"],
                    interactive=False,
                    label="All players with minutes in the selected NBA game",
                )
                gr.HTML(_section(
                    "Complete sampled game results",
                    "Team scores from the ESPN league scoreboard; sample size "
                    "depends on the recent acquisition window."
                ))
                all_match_table=gr.Dataframe(
                    value=match_table(games,limit=250),
                    headers=["Date","Away","Away PTS","Home PTS","Home",
                             "Total PTS","Margin","ESPN box score"],
                    interactive=False,label="Up to 180 actual NBA games",
                )

            with gr.Tab("Team Analytics"):
                gr.HTML(_section(
                    "NBA Team Analytics",
                    "Observed W/L, scoring and player output from the completed "
                    "games collected at launch. Not official full-season standings."
                ))
                with gr.Row():
                    team_selector=gr.Dropdown(
                        choices=team_names,value=team_names[0],
                        label="Select NBA team",filterable=True,
                    )
                team_cards=gr.HTML(initial_teams[0])
                with gr.Row():
                    with gr.Column(scale=6):
                        gr.Plot(
                            value=team_win_chart(team_stats),
                            label="Top NBA teams by win rate in observed games",
                        )
                    with gr.Column(scale=5):
                        gr.Dataframe(
                            value=standings_table(team_stats),
                            headers=["Team","W","L","GP","Win %",
                                     "Point diff / game"],
                            interactive=False,
                            label="All 30 NBA teams in recent ESPN sample",
                        )
                gr.HTML(_section(
                    "Selected team's recent game results",
                    "Scores and opponents from completed ESPN NBA games."
                ))
                team_matches=gr.Dataframe(
                    value=initial_teams[1],
                    headers=["Date","Away","Away PTS","Home PTS","Home",
                             "Total PTS","Margin","ESPN box score"],
                    interactive=False,
                    label="Team game history · observed NBA matchups",
                )
                gr.HTML(_section(
                    "Team player scoring leaders",
                    "Average actual points per played game in this "
                    "recent league-wide box-score sample."
                ))
                team_player_table=gr.Dataframe(
                    value=initial_teams[2],
                    headers=["Player","Games","Avg PTS",
                             "Single-game high","Total PTS"],
                    interactive=False,label="Selected team athlete analysis",
                )

            with gr.Tab("Data Explorer"):
                gr.HTML(_section(
                    "ESPN Data Explorer · browse the underlying evidence",
                    "Search thousands of source-linked NBA game records, "
                    "including the broader recent league sample and longer "
                    "game histories of featured players."
                ))
                with gr.Row():
                    data_search=gr.Textbox(
                        value="",label="Player or opponent",
                        placeholder="e.g. Jokic, Celtics, Knicks ...",
                    )
                    data_team=gr.Dropdown(
                        choices=["All teams"]+team_names,
                        value="All teams",label="Team filter",
                    )
                    data_metric=gr.Dropdown(
                        choices=list(METRICS),value="PTS",
                        label="Metric for sample summary",
                    )
                    data_limit=gr.Dropdown(
                        choices=[("50 rows",50),("100 rows",100),
                                 ("300 rows",300),("1,000 rows",1000),
                                 ("All available rows",10000)],
                        value=100,label="Visible records",
                    )
                data_button=gr.Button("Filter real ESPN database",variant="primary")
                data_cards=gr.HTML(explorer_html)
                data_table=gr.Dataframe(
                    value=explorer_rows,
                    headers=player_table(league_rows,limit=1)[0],
                    interactive=False,label="Real basketball box-score observations",
                )
                gr.HTML(
                    '<div class="note-teal">Source lineage: ESPN scoreboards '
                    'and game-summary box scores, with additional ESPN athlete '
                    'gamelogs for modeled players. Recent games are deduplicated '
                    'by ESPN athlete ID plus game ID. Blank columns mean the '
                    'original feed did not provide that field.</div>'
                )

            with gr.Tab("ML Model Audit"):
                gr.HTML(_section(
                    "Model audit · evidence before forecasts",
                    "Reproducible chronological evaluation. Models can "
                    "underperform a rolling-average baseline; the dashboard "
                    "shows that explicitly."
                ))
                gr.Plot(
                    value=model_accuracy_chart(state["models"]),
                    label="Observed held-out model errors",
                )
                metrics=[
                    [name,m["selected_model"],m["validation_mae"],
                     m["test_mae"],m["baseline_test_mae"],
                     round(m["test_mae"]-m["baseline_test_mae"],3),
                     m["n_train"],m["n_val"],m["n_test"],
                     m["test_from"],m["test_through"]]
                    for name,m in state["models"].items()
                ]
                gr.Dataframe(
                    value=metrics,
                    headers=["Metric","Chosen model","Validation MAE",
                             "Test MAE","Baseline MAE",
                             "Model MAE − baseline","Train N",
                             "Validation N","Test N",
                             "Test begins","Test ends"],
                    interactive=False,
                    label="Actual ESPN games · global chronological train / validation / test split",
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML(
                            '<div class="section-card"><div class="section-title">'
                            '01 · Feature engineering</div>'
                            '<div class="section-note">'
                            'Only previous games enter the forecast: five-, '
                            'ten- and fifteen-game scoring/rebounding/etc. '
                            'means, player minutes, recent trend, venue '
                            'and rest days. The target game is never used '
                            'to construct its own features.'
                            '</div></div>'
                        )
                    with gr.Column(scale=1):
                        gr.HTML(
                            '<div class="section-card"><div class="section-title">'
                            '02 · Strict evaluation</div>'
                            '<div class="section-note">'
                            'NBA game dates split globally into 70% '
                            'training, 15% validation and 15% holdout. '
                            'Model choice occurs on validation only. '
                            'Only observed game statistics are targets.'
                            '</div></div>'
                        )
                    with gr.Column(scale=1):
                        gr.HTML(
                            '<div class="section-card"><div class="section-title">'
                            '03 · Honest uncertainty</div>'
                            '<div class="section-note">'
                            'Threshold percentages derive from held-out '
                            'validation residuals and are NOT formally '
                            'calibrated probabilities. Team fixtures do not '
                            'confirm a player is available.'
                            '</div></div>'
                        )
                gr.HTML(_section(
                    "Data sources and reproducibility",
                    "This version is a public real-data analytics workflow "
                    "separate from the original research-stage V3 ensemble."
                ))
                gr.HTML(
                    '<div class="section-card"><div class="section-note">'
                    f'<strong>League data fetched:</strong> '
                    f'{escape(league["as_of_utc"])} UTC.<br>'
                    f'<strong>Featured-player data fetched:</strong> '
                    f'{escape(state["as_of_utc"])} UTC.<br>'
                    f'<strong>Completed ESPN games:</strong> {len(games):,}; '
                    f'<strong>player observations:</strong> {len(league_rows):,}; '
                    f'<strong>unique athletes:</strong> {unique_players:,}.<br>'
                    f'<strong>Local isolated SQLite:</strong> '
                    f'<code>{escape(state["db_path"])}</code> for featured players. '
                    'No private Google Drive, personal credentials or '
                    'private trained model binaries are included.<br>'
                    '<strong>Source and method:</strong> '
                    '<a href="https://github.com/zignale-mattia/NBA-Intelligence-ML" '
                    'target="_blank" rel="noopener noreferrer">'
                    'Public GitHub repository ↗</a>. '
                    'The full original 14-stage research notebook is '
                    'maintained separately. This public dashboard does '
                    'not claim its entire V3 ensemble and injury-aware '
                    'pipeline have been reproduced.</div></div>'
                )
                gr.HTML(
                    '<div class="note-amber">Research limitations: '
                    'ESPN endpoints are undocumented and may be delayed or '
                    'unavailable; the recent league snapshot is not an '
                    'official full-season standings table; next-game models '
                    'do not verify player availability, injury reports '
                    'or opponent defensive strength. '
                    'No investment or wagering recommendation is produced.</div>'
                )

        # User interactions: every visible control updates the relevant data.
        forecast_button.click(
            forecast_outputs,
            inputs=[modeled,model_stat,thresh,fallback_home,fallback_rest],
            outputs=[forecast_card,forecast_plot,forecast_table],
        )
        player_btn.click(
            profile_outputs,
            inputs=[player_selector,player_stat,player_period,player_venue],
            outputs=[player_cards,player_plot,player_dist,player_game_table],
        )
        compare_button.click(
            compare_outputs,
            inputs=[player_a,player_b,compare_stat,compare_window],
            outputs=[compare_chart,compare_table],
        )
        leader_run.click(
            leaderboard_outputs,
            inputs=[leader_metric,leader_min_games],
            outputs=[rank_plot,rank_table],
        )
        game_team.change(
            filter_games,
            inputs=[game_team],
            outputs=[all_match_table,game_selector,game_score,game_player_table],
        )
        game_selector.change(
            game_scoreboard,
            inputs=[game_selector],
            outputs=[game_score,game_player_table],
        )
        team_selector.change(
            team_outputs,
            inputs=[team_selector],
            outputs=[team_cards,team_matches,team_player_table],
        )
        data_button.click(
            search_rows,
            inputs=[data_search,data_team,data_metric,data_limit],
            outputs=[data_cards,data_table],
        )
    return app
