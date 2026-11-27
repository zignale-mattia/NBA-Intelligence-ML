"""Expanded, observed NBA league explorer based on ESPN scoreboard + box scores.

Independent of the smaller training athlete cohort. All values come from
completed NBA game records; no generated/simulated NBA observations.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from collections import defaultdict
from typing import Optional

from .real_espn import fetch_json


def _numeric(value):
    try:
        if value is None or str(value).strip() in ("", "--", "DNP", "-"):
            return None
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


def _made_attempted(value):
    if not isinstance(value, str) or "-" not in value:
        return None, None
    left, right = value.split("-", 1)
    return _numeric(left), _numeric(right)


def completed_games(days=42, *, through=None):
    """ESPN league scoreboard, real completed fixtures only."""
    if not (1 <= int(days) <= 110):
        raise ValueError("days should be from 1 to 110")
    day = through or datetime.now(timezone.utc).date()
    urls = [
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
        f"scoreboard?dates={(day-timedelta(days=i)).strftime('%Y%m%d')}&limit=100"
        for i in range(int(days))
    ]
    events = {}
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_json, url, 2): url for url in urls}
        for future in as_completed(futures):
            try:
                payload = future.result()
                for event in payload.get("events", []):
                    # ESPN season.type 2 is regular season. Exclude preseason
                    # and postseason from this time-scoped league dashboard.
                    season = event.get("season") or {}
                    if (season.get("type") == 2 and
                        event.get("status", {}).get("type", {}).get("completed")):
                        events[str(event["id"])] = event
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
    return sorted(events.values(), key=lambda e:e.get("date", "")), failures


def parse_game_boxscore(payload, event):
    """Player stats from ESPN 'summary' and official completed-game metadata."""
    if not event.get("status", {}).get("type", {}).get("completed"):
        return [], None
    event_id = str(event.get("id") or "")
    if not event_id.isdigit():
        return [], None
    game_date = str(event.get("date", ""))[:10]
    competition = (event.get("competitions") or [{}])[0]
    competitors = competition.get("competitors", [])
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        return [], None
    home_team, away_team = home.get("team", {}), away.get("team", {})
    home_score, away_score = _numeric(home.get("score")), _numeric(away.get("score"))
    if home_score is None or away_score is None:
        return [], None
    match = {
        "date": game_date, "game_id": event_id,
        "home": home_team.get("displayName", "Unknown"),
        "away": away_team.get("displayName", "Unknown"),
        "home_abbr": home_team.get("abbreviation", ""),
        "away_abbr": away_team.get("abbreviation", ""),
        "home_score": int(home_score), "away_score": int(away_score),
        "margin": abs(int(home_score-away_score)),
        "total": int(home_score+away_score),
        "source_url": f"https://www.espn.com/nba/boxscore/_/gameId/{event_id}",
    }
    players = []
    for team in (payload.get("boxscore") or {}).get("players", []):
        team_meta = team.get("team") or {}
        club = team_meta.get("displayName", "")
        for section in team.get("statistics", []):
            labels = section.get("labels") or []
            for athlete in section.get("athletes", []):
                if athlete.get("didNotPlay"):
                    continue
                raw = athlete.get("stats") or []
                if len(raw) != len(labels):
                    continue
                vals = dict(zip(labels, raw))
                minutes = _numeric(vals.get("MIN"))
                pts, reb, ast = (_numeric(vals.get(s)) for s in ("PTS","REB","AST"))
                three, three_att = _made_attempted(vals.get("3PT"))
                fgm, fga = _made_attempted(vals.get("FG"))
                ftm, fta = _made_attempted(vals.get("FT"))
                if minutes is None or minutes <= 0 or any(
                    x is None for x in (pts, reb, ast, three)
                ):
                    continue
                bio = athlete.get("athlete") or {}
                name = bio.get("displayName", "")
                if not name or not str(bio.get("id","")).isdigit():
                    continue
                games_home = club == match["home"]
                players.append({
                    "date": game_date, "game_id": event_id,
                    "player": name, "athlete_id": str(bio["id"]),
                    "team": club,
                    "opponent": match["away"] if games_home else match["home"],
                    "home": int(games_home),
                    "position": (bio.get("position") or {}).get("abbreviation", ""),
                    "starter": bool(athlete.get("starter")),
                    "MIN": minutes, "PTS": pts, "REB": reb, "AST": ast,
                    "3PM": three, "3PA": three_att,
                    "FGM": fgm, "FGA": fga, "FTM": ftm, "FTA": fta,
                    "STL": _numeric(vals.get("STL")),
                    "BLK": _numeric(vals.get("BLK")),
                    "TOV": _numeric(vals.get("TO")),
                    "PRA": pts+reb+ast,
                    "source_url": match["source_url"],
                })
    return players, match


def _download_summary(event):
    event_id = str(event["id"])
    uri = ("https://site.api.espn.com/apis/site/v2/"
           f"sports/basketball/nba/summary?event={event_id}")
    return parse_game_boxscore(fetch_json(uri,2),event)


def build_league_explorer(days=42, *, max_games=180, min_games=3):
    """Collect bounded recent ESPN league-wide game summaries on launch."""
    events, failures = completed_games(days=days)
    if not events:
        raise RuntimeError("ESPN scoreboard returned no completed NBA events")
    selected = events[-max_games:]
    games_by_id, players_by_key = {}, {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_download_summary, e): str(e["id"])
                   for e in selected}
        for f in as_completed(futures):
            try:
                players, game = f.result()
                if game and players:
                    games_by_id[game["game_id"]] = game
                    for p in players:
                        players_by_key[(p["game_id"],p["athlete_id"])] = p
            except Exception as exc:
                failures.append(f"{futures[f]}: {type(exc).__name__}: {exc}")
    records = sorted(players_by_key.values(),
                     key=lambda p:(p["date"],p["game_id"],p["player"]))
    matches = sorted(games_by_id.values(),key=lambda g:(g["date"],g["game_id"]))
    if len(matches)<12 or len(records)<150:
        raise RuntimeError(
            f"Too little actual ESPN data: {len(matches)} games and "
            f"{len(records)} player performances. Missing: {len(failures)} requests"
        )
    return {
        "rows": records, "matches": matches,
        "warnings": failures,
        "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "players": len(set(r["athlete_id"] for r in records)),
        "teams": len(set(r["team"] for r in records)),
        "requested_games": len(selected),
    }


def leaderboards(rows, stat="PTS", min_games=3):
    if stat not in ("PTS","REB","AST","3PM","PRA","MIN","STL","BLK"):
        raise ValueError("Unknown league statistic")
    grouped = defaultdict(list)
    for r in rows:grouped[(r["athlete_id"],r["player"],r["team"])].append(r)
    table = []
    for (athlete_id,name,team),records in grouped.items():
        if len(records)<min_games: continue
        vals=[float(x[stat]) for x in records if x.get(stat) is not None]
        if len(vals)<min_games:continue
        table.append({
            "player":name,"athlete_id":athlete_id,"team":team,
            "games":len(vals),"mean":sum(vals)/len(vals),
            "total":sum(vals),"high":max(vals),"low":min(vals),
        })
    return sorted(table,key=lambda r:(-r["mean"],-r["games"],r["player"]))


def team_performance(matches):
    teams=defaultdict(lambda:{"W":0,"L":0,"PF":0,"PA":0})
    for g in matches:
        for side,other in (("home","away"),("away","home")):
            name=g[side]
            pf=g[f"{side}_score"];pa=g[f"{other}_score"]
            teams[name]["PF"]+=pf;teams[name]["PA"]+=pa
            teams[name]["W" if pf>pa else "L"]+=1
    result=[]
    for name,v in teams.items():
        games=v["W"]+v["L"]
        result.append({
            "team":name,"W":v["W"],"L":v["L"],
            "win_pct":round(v["W"]/games*100,1),
            "diff_per_game":round((v["PF"]-v["PA"])/games,1),
            "games":games,
        })
    return sorted(result,key=lambda x:(-x["win_pct"],-x["W"],x["team"]))


def persist_league_sqlite(league: dict, directory: str) -> str:
    """Persist observed game scores and athlete box scores, isolated from Drive."""
    import sqlite3
    from pathlib import Path
    destination=Path(directory)/"nba_league_real_espn.sqlite"
    destination.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(destination) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS completed_games (
            game_id TEXT PRIMARY KEY,date TEXT,home TEXT,away TEXT,
            home_score INTEGER,away_score INTEGER,source_url TEXT
        )
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS player_boxscores (
            game_id TEXT, athlete_id TEXT,game_date TEXT,player TEXT,
            team TEXT,opponent TEXT,home INTEGER,starter INTEGER,
            minutes REAL,pts REAL,reb REAL,ast REAL,three_pm REAL,
            three_pa REAL,fgm REAL,fga REAL,ftm REAL,fta REAL,
            steals REAL,blocks REAL,turnovers REAL,pra REAL,
            source_url TEXT,
            PRIMARY KEY(game_id,athlete_id)
        )
        """)
        con.execute("DELETE FROM completed_games")
        con.execute("DELETE FROM player_boxscores")
        con.executemany(
            "INSERT INTO completed_games VALUES (?,?,?,?,?,?,?)",
            [
                (g["game_id"],g["date"],g["home"],g["away"],
                 g["home_score"],g["away_score"],g["source_url"])
                for g in league["matches"]
            ],
        )
        con.executemany(
            "INSERT INTO player_boxscores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (r["game_id"],r["athlete_id"],r["date"],r["player"],
                 r["team"],r["opponent"],r["home"],int(r["starter"]),
                 r["MIN"],r["PTS"],r["REB"],r["AST"],r["3PM"],r["3PA"],
                 r["FGM"],r["FGA"],r["FTM"],r["FTA"],
                 r["STL"],r["BLK"],r["TOV"],r["PRA"],r["source_url"])
                for r in league["rows"]
            ],
        )
    return str(destination)
