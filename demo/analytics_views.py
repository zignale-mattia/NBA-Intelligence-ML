"""Pure helpers for a recruiter-grade NBA analytics dashboard.

All displayed statistical observations originate from ESPN game logs/boxscores.
This module deliberately has zero Gradio dependencies for robust unit testing.
"""
from collections import defaultdict
from datetime import date
from html import escape
from math import sqrt

METRICS=("PTS","REB","AST","3PM","PRA","MIN","STL","BLK")

def average(rows, stat):
    values=[float(r[stat]) for r in rows if r.get(stat) is not None]
    return sum(values)/len(values) if values else None

def snapshot(rows, matches, players):
    return {
        "performances":len(rows),"games":len(matches),
        "athletes":len({r["athlete_id"] for r in rows}),
        "teams":len({r["team"] for r in rows}),
        "ml_players":len(players),
        "last_game":max((g["date"] for g in matches),default="—"),
    }

def player_rows(rows,player,season=None,venue="All",last_n=0):
    result=[r for r in rows if r["player"]==player]
    if season and season!="All":
        result=[r for r in result if r["date"][:4]==season[:4]]
    if venue=="Home":result=[r for r in result if r["home"]]
    if venue=="Away":result=[r for r in result if not r["home"]]
    result=sorted(result,key=lambda r:(r["date"],r["game_id"]))
    return result[-int(last_n):] if last_n else result

def describe(rows,stat):
    if stat not in METRICS:raise ValueError("Unknown metric")
    vals=[float(r[stat]) for r in rows if r.get(stat) is not None]
    if not vals:return {"games":0,"avg":None,"median":None,"high":None,"low":None,"std":None,"total":None}
    ordered=sorted(vals);n=len(vals)
    med=ordered[n//2] if n%2 else (ordered[n//2-1]+ordered[n//2])/2
    mean=sum(vals)/n
    return {
        "games":n,"avg":round(mean,2),"median":round(med,2),
        "high":round(max(vals),2),"low":round(min(vals),2),
        "std":round(sqrt(sum((v-mean)**2 for v in vals)/n),2),
        "total":round(sum(vals),2),
    }

def grouped_threshold(rows,stat,threshold):
    if stat not in METRICS:raise ValueError("Unknown metric")
    if threshold is None or float(threshold)<0:raise ValueError("Invalid threshold")
    vals=[float(r[stat]) for r in rows if r.get(stat) is not None]
    if not vals:return None
    return round(sum(v>=float(threshold) for v in vals)/len(vals)*100,1)

def comparison(rows,names,stat,limit=20):
    if stat not in METRICS:raise ValueError("Unknown metric")
    result=[]
    for player in names:
        logs=player_rows(rows,player,last_n=limit)
        if logs:
            s=describe(logs,stat)
            result.append({
                "player":player,"games":s["games"],
                "avg":s["avg"],"high":s["high"],
                "median":s["median"],"std":s["std"],
                "last_game":logs[-1]["date"],
            })
    return sorted(result,key=lambda x:(-x["avg"],x["player"]))

def player_table(rows,limit=100):
    headers=["Date","Player","Team","Opponent","H/A","MIN","PTS",
             "REB","AST","3PM","FGM","FGA","3PA","STL","BLK",
             "TOV","PRA","ESPN box score"]
    output=[]
    for r in sorted(rows,key=lambda x:(x["date"],x["game_id"]),reverse=True)[:limit]:
        output.append([
            r["date"],r["player"],r["team"],r["opponent"],
            "Home" if r["home"] else "Away",
            r.get("MIN"),r.get("PTS"),r.get("REB"),r.get("AST"),
            r.get("3PM"),r.get("FGM"),r.get("FGA"),r.get("3PA"),
            r.get("STL"),r.get("BLK"),r.get("TOV"),r.get("PRA"),
            r["source_url"],
        ])
    return headers,output

def match_table(matches,limit=150):
    return [
        [g["date"],g["away"],g["away_score"],g["home_score"],
         g["home"],g["total"],g["margin"],g["source_url"]]
        for g in sorted(matches,key=lambda x:(x["date"],x["game_id"]),reverse=True)[:limit]
    ]

def standings_table(teams):
    return [
        [v["team"],v["W"],v["L"],v["games"],v["win_pct"],v["diff_per_game"]]
        for v in teams
    ]

def _path(points):
    return " ".join(f"{x:.1f},{y:.1f}" for x,y in points)

def trend_svg(rows,stat="PTS",title="Recent performance",forecast=None):
    if stat not in METRICS:raise ValueError("Unknown metric")
    valid=[(r["date"],float(r[stat])) for r in rows if r.get(stat) is not None]
    valid=valid[-35:]
    if not valid:return '<div class="empty-data">No observed NBA data for this view</div>'
    width,height=980,270
    left,right,top,bottom=50,26,28,38
    vals=[v for _,v in valid]+([float(forecast)] if forecast is not None else [])
    low=min(0,min(vals)*0.82);high=max(vals)*1.10+1
    def coords(idx,v):
        x=left+idx*(width-left-right)/max(1,len(valid)-1+(forecast is not None))
        y=height-bottom-(v-low)/(high-low)*(height-top-bottom)
        return x,y
    points=[coords(i,v) for i,(_,v) in enumerate(valid)]
    grid=[]
    for k in range(5):
        val=low+(high-low)*k/4
        y=coords(0,val)[1]
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#24314b" stroke-dasharray="4 7"/><text x="{left-8}" y="{y+4:.1f}" text-anchor="end" fill="#92a3bc" font-size="13">{val:.0f}</text>')
    circles="".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="#6ce5ce"><title>{escape(d)}: {v:g} {stat}</title></circle>' for ((d,v),(x,y)) in zip(valid,points))
    prediction=""
    if forecast is not None:
        x1,y1=points[-1];x2,y2=coords(len(valid),float(forecast))
        prediction=f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#ffb44b" stroke-width="3" stroke-dasharray="7 6"/><circle cx="{x2:.1f}" cy="{y2:.1f}" r="6" fill="#ffb44b"><title>Model estimate {forecast:.1f}</title></circle>'
    labels=f'<text x="{left}" y="{height-11}" fill="#97a9be" font-size="13">{escape(valid[0][0])}</text><text x="{width-right}" y="{height-11}" text-anchor="end" fill="#97a9be" font-size="13">{escape(valid[-1][0])}</text>'
    key = ('<span><i class="dot-amber"></i>ML estimate</span>'
           if forecast is not None else '')
    return (
        f'<div class="chart-shell"><div class="chart-top">'
        f'<strong>{escape(title)}</strong><span>{len(valid)} actual games · ESPN</span>'
        f'</div><svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(stat)} real game trend">'
        + ''.join(grid)
        + f'<polyline points="{_path(points)}" fill="none" '
          f'stroke="#6ce5ce" stroke-width="3.2" stroke-linejoin="round"/>'
        + circles + prediction + labels
        + '</svg><div class="chart-key"><span><i class="dot-teal"></i>'
          'Actual box scores</span>' + key + '</div></div>'
    )


def bar_svg(rows,label="PTS",limit=12):
    if not rows:return '<div class="empty-data">No player records available</div>'
    top=rows[:limit]
    high=max(float(row["mean"]) for row in top) or 1.0
    bars=[]
    for i,row in enumerate(top):
        width=max(1,round(float(row["mean"])/high*100,1))
        bars.append(f'<div class="rank-row"><span class="rank-num">{i+1:02}</span><span class="rank-player">{escape(row["player"])}<small>{escape(row["team"])} · {row["games"]} GP</small></span><span class="rank-track"><span class="rank-fill" style="width:{width}%"></span></span><b>{float(row["mean"]):.1f}</b></div>')
    return '<div class="ranking">'+''.join(bars)+'</div>'

def metric_card(label,value,detail="",accent="teal"):
    return f'<div class="kpi"><div class="kpi-label">{escape(str(label))}</div><div class="kpi-value {escape(accent)}">{escape(str(value))}</div><div class="kpi-detail">{escape(str(detail))}</div></div>'

def kpis_html(items):
    return '<div class="kpi-grid">'+''.join(metric_card(*x) for x in items)+'</div>'
