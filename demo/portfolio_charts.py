"""Interactive Plotly views for the source-linked, real-data NBA portfolio."""

import plotly.graph_objects as go

BG="#0d1627"
GRID="#253450"
TEXT="#dce7f7"
SUB="#92a4bd"
TEAL="#60e0ca"
BLUE="#69aaff"
AMBER="#ffb85b"
PURPLE="#aa9cff"
COLORS=(TEAL,BLUE,AMBER,PURPLE,"#f68eb2","#69d4f7")


def _figure(title="",height=340):
    fig=go.Figure()
    fig.update_layout(
        title=dict(text=title,font=dict(size=17,color=TEXT)),
        paper_bgcolor=BG,plot_bgcolor=BG,
        font=dict(family="Inter, Arial, sans-serif",color=TEXT,size=13),
        height=height,margin=dict(l=54,r=28,t=58,b=55),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a2840",font=dict(color="#f8fafc")),
        xaxis=dict(showgrid=True,gridcolor=GRID,zeroline=False,
                   linecolor=GRID,tickfont=dict(color=SUB)),
        yaxis=dict(showgrid=True,gridcolor=GRID,zeroline=False,
                   linecolor=GRID,tickfont=dict(color=SUB)),
        legend=dict(orientation="h",x=0,y=1.12),
    )
    return fig


def time_series(rows,stat,title=None,model_estimate=None,last_n=35):
    data=sorted((r for r in rows if r.get(stat) is not None),
                key=lambda r:(r["date"],r.get("game_id",r.get("event_id",""))))[-last_n:]
    fig=_figure(title or f"{stat} · Actual NBA game trend",360)
    if not data:return fig
    dates=[r["date"] for r in data]
    vals=[float(r[stat]) for r in data]
    hints=[f'{r.get("opponent","")} · {r.get("source_url","")}' for r in data]
    fig.add_trace(go.Scatter(
        x=dates,y=vals,name=f"{stat} · Actual",mode="lines+markers",
        line=dict(color=TEAL,width=3),marker=dict(color=TEAL,size=7),
        customdata=hints,
        hovertemplate="%{x}<br><b>%{y:.1f} "+stat+"</b><br>%{customdata}<extra></extra>",
    ))
    if len(vals)>=5:
        smooth=[sum(vals[max(0,i-4):i+1]) / len(vals[max(0,i-4):i+1])
                for i in range(len(vals))]
        fig.add_trace(go.Scatter(
            x=dates,y=smooth,name="Rolling 5-game average",
            mode="lines",line=dict(color=PURPLE,width=2,dash="dot"),
            hovertemplate="%{x}<br>Rolling 5: %{y:.1f}<extra></extra>",
        ))
    if model_estimate is not None:
        fig.add_trace(go.Scatter(
            x=["Next game"],y=[float(model_estimate)],
            mode="markers",name="ML next-game estimate",
            marker=dict(color=AMBER,size=15,symbol="diamond"),
            hovertemplate="ML next-game estimate: %{y:.1f}<extra></extra>",
        ))
    fig.update_yaxes(title_text=stat,rangemode="tozero")
    fig.update_xaxes(title_text="Game date")
    return fig


def player_comparison(histories,stat,players,last_n=20):
    fig=_figure("Head-to-head · real NBA box scores",390)
    for i,name in enumerate(players):
        h=sorted(histories[name],key=lambda r:(r["date"],r.get("game_id",r.get("event_id",""))))
        rows=[r for r in h if r.get(stat) is not None][-last_n:]
        if not rows:continue
        y=[float(r[stat]) for r in rows]
        fig.add_trace(go.Scatter(
            x=list(range(1,len(rows)+1)),y=y,mode="lines+markers",name=name,
            line=dict(width=2.7,color=COLORS[i%len(COLORS)]),
            marker=dict(size=6),
            customdata=[r["date"]+" · "+r.get("opponent","") for r in rows],
            hovertemplate="Game %{x}<br><b>%{y:.1f} "+stat+"</b><br>%{customdata}<extra></extra>",
        ))
    fig.update_xaxes(title_text=f"Game index · last {last_n} actual played games",dtick=2)
    fig.update_yaxes(title_text=stat,rangemode="tozero")
    return fig


def rank_chart(leaders,stat,limit=12):
    fig=_figure(f"League leaders · average {stat} per played game",435)
    rows=list(reversed(leaders[:limit]))
    if rows:
        fig.add_trace(go.Bar(
            x=[round(r["mean"],2) for r in rows],y=[r["player"] for r in rows],
            orientation="h",marker=dict(color=TEAL,opacity=0.9),
            customdata=[f'{r["team"]} · {r["games"]} GP' for r in rows],
            text=[f'{r["mean"]:.1f}' for r in rows],textposition="outside",
            hovertemplate="%{y}<br>%{x:.2f} "+stat+"/game<br>%{customdata}<extra></extra>",
        ))
    fig.update_layout(margin=dict(l=158,r=50,t=50,b=44),showlegend=False)
    fig.update_xaxes(title_text=f"{stat} per game",rangemode="tozero")
    fig.update_yaxes(showgrid=False)
    return fig


def distribution_chart(rows,stat,player):
    vals=[float(r[stat]) for r in rows if r.get(stat) is not None]
    fig=_figure(f"{player} · {stat} game distribution",340)
    if vals:
        fig.add_trace(go.Histogram(
            x=vals,nbinsx=min(16,max(5,len(vals)//4)),
            marker=dict(color=BLUE,line=dict(color=BG,width=1)),
            hovertemplate=f"{stat}=%{{x}}<br>Games=%{{y}}<extra></extra>",
        ))
        avg=sum(vals)/len(vals)
        fig.add_vline(x=avg,line=dict(color=AMBER,width=2,dash="dash"),
                      annotation_text=f"Mean {avg:.1f}",
                      annotation_font_color=AMBER)
    fig.update_xaxes(title_text=stat)
    fig.update_yaxes(title_text="Number of played games",dtick=1)
    return fig


def model_accuracy_chart(models):
    names=list(models)
    fig=_figure("Chronological out-of-sample error · lower is better",380)
    fig.add_trace(go.Bar(
        x=names,y=[models[s]["test_mae"] for s in names],
        name="Selected ML model",marker_color=TEAL,
        text=[f'{models[s]["test_mae"]:.2f}' for s in names],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=names,y=[models[s]["baseline_test_mae"] for s in names],
        name="Last-10 mean baseline",marker_color=BLUE,
        text=[f'{models[s]["baseline_test_mae"]:.2f}' for s in names],
        textposition="outside",
    ))
    fig.update_layout(barmode="group",legend=dict(orientation="h",y=1.14))
    fig.update_yaxes(title_text="Mean absolute error (stat units)",rangemode="tozero")
    return fig


def team_win_chart(standings):
    fig=_figure("NBA team win % · games in this collected sample",500)
    rows=list(reversed(standings[:20]))
    if rows:
        fig.add_trace(go.Bar(
            x=[r["win_pct"] for r in rows],
            y=[r["team"] for r in rows],orientation="h",
            marker_color=PURPLE,
            customdata=[f'{r["W"]}W / {r["L"]}L · {r["games"]} games' for r in rows],
            hovertemplate="%{y}<br>%{x:.1f}% W<br>%{customdata}<extra></extra>",
        ))
    fig.update_layout(margin=dict(l=190,r=35,t=50,b=50),showlegend=False)
    fig.update_xaxes(title_text="Win rate in analyzed games (%)",range=[0,103])
    fig.update_yaxes(showgrid=False)
    return fig
