"""ESPN NBA player game-log ingestion for the REAL-data Colab portfolio.

No synthetic data, secret, licensed asset bundle, or private Google Drive is used.
Data is fetched on demand from ESPN public JSON endpoints. API is undocumented
and may fail/change; errors are reported rather than replacing real records.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
import time
import urllib.error
import urllib.request

# Athlete identities/IDs verified against ESPN athlete metadata.
ATHLETES = {
    "Nikola Jokic": "3112335",
    "Luka Doncic": "3945274",
    "Shai Gilgeous-Alexander": "4278073",
    "Giannis Antetokounmpo": "3032977",
    "Jalen Brunson": "3934672",
    "Victor Wembanyama": "5104157",
    "Anthony Edwards": "4594268",
    "Tyrese Maxey": "4431678",
    "Jayson Tatum": "4065648",
    "Stephen Curry": "3975",
    "Jaylen Brown": "3917376",
    "De'Aaron Fox": "4066259",
    "LaMelo Ball": "4432816",
    "Deni Avdija": "4683021",
}
SEASONS = (2026, 2027)
STATS = ("PTS", "REB", "AST", "3PM", "PRA")
SOURCE = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes"
USER_AGENT = "NBAIntelligencePortfolio/1.0 (research educational demo)"


def espn_gamelog_url(athlete_id: str, season: int) -> str:
    if athlete_id not in ATHLETES.values() or season not in SEASONS:
        raise ValueError("Player/season not in public demo allowlist")
    return f"{SOURCE}/{athlete_id}/gamelog?season={season}"


def fetch_json(url: str, attempts: int = 3) -> dict:
    """Retry transient public ESPN errors; never use stored/private responses."""
    last_error = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=20) as response:
                result = json.load(response)
            if not isinstance(result, dict):
                raise ValueError("Unexpected ESPN JSON payload")
            return result
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ESPN fetch failed for {url}: {last_error}")


def _float(value):
    try:
        if value is None or str(value).strip() in ("", "-", "--", "DNP", "N/A"):
            return None
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _threes(value):
    """ESPN's '3PT' label stores makes-attempts, e.g. '2-6'."""
    if not isinstance(value, str) or "-" not in value:
        return None
    return _float(value.split("-", 1)[0])


def parse_gamelog(payload: dict, athlete_id: str, name: str, season: int) -> list[dict]:
    """Use ONLY played regular-season games with verifiable ESPN event IDs."""
    labels = payload.get("labels", [])
    events_by_id = payload.get("events", {})
    if not isinstance(events_by_id, dict):
        raise ValueError("ESPN gamelog lacks event metadata")
    for required in ("MIN", "PTS", "REB", "AST", "3PT"):
        if required not in labels:
            raise ValueError(f"ESPN changed gamelog schema: {required} missing")
    now = datetime.now(timezone.utc).date()
    games = {}
    for season_type in payload.get("seasonTypes", []):
        if "Regular Season" not in season_type.get("displayName", ""):
            continue
        for cat in season_type.get("categories", []):
            if cat.get("type") != "event":
                continue
            for item in cat.get("events", []):
                event_id = str(item.get("eventId") or "")
                info = events_by_id.get(event_id, {})
                raw_stats = item.get("stats", [])
                if not event_id.isdigit() or len(raw_stats) != len(labels):
                    continue
                stats = dict(zip(labels, raw_stats))
                minutes = _float(stats["MIN"])
                pts = _float(stats["PTS"])
                reb = _float(stats["REB"])
                ast = _float(stats["AST"])
                made_3 = _threes(stats["3PT"])
                if (minutes is None or minutes <= 0 or
                    any(value is None for value in (pts, reb, ast, made_3))):
                    continue
                raw_date = str(info.get("gameDate") or "")[:10]
                try:
                    game_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if game_date > now:
                    continue
                opponent = info.get("opponent") or {}
                team = info.get("team") or {}
                if not opponent.get("displayName") or not team.get("abbreviation"):
                    continue
                games[event_id] = {
                    "player": name, "athlete_id": athlete_id,
                    "event_id": event_id, "date": game_date.isoformat(),
                    "season": f"{season-1}-{str(season)[-2:]}",
                    "team": str(team["abbreviation"]),
                    "opponent": str(opponent["displayName"]),
                    "home": int(info.get("atVs") == "vs"),
                    "MIN": minutes,
                    "PTS": pts, "REB": reb, "AST": ast, "3PM": made_3,
                    "PRA": pts + reb + ast,
                    "source_url": (
                        f"https://www.espn.com/nba/boxscore/_/gameId/{event_id}"
                    ),
                }
    return sorted(games.values(), key=lambda x: (x["date"], x["event_id"]))


def _fetch_one(player: str, athlete_id: str, season: int) -> list[dict]:
    url = espn_gamelog_url(athlete_id, season)
    return parse_gamelog(fetch_json(url), athlete_id, player, season)


def collect_real_games(
    athletes: dict | None = None, seasons: tuple[int, ...] = SEASONS,
    *, min_players: int = 5, workers: int = 5,
) -> dict:
    """Fetch real ESPN regular-season player logs for the requested athletes."""
    people = dict(ATHLETES if athletes is None else athletes)
    if not people or not set(people.items()).issubset(set(ATHLETES.items())):
        raise ValueError("Selected ESPN athlete IDs are not recognized")
    if not seasons or not set(seasons).issubset(set(SEASONS)):
        raise ValueError("Unsupported season")
    records = {}
    failures = []
    tasks = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for player, athlete_id in people.items():
            for season in seasons:
                future = executor.submit(_fetch_one, player, athlete_id, season)
                tasks[future] = (player, season)
        for future in as_completed(tasks):
            player, season = tasks[future]
            try:
                logs = future.result()
                for game in logs:
                    records[(game["athlete_id"], game["event_id"])] = game
            except Exception as exc:
                failures.append(f"{player} {season}: {type(exc).__name__}: {exc}")
    games = sorted(records.values(), key=lambda r: (
        r["date"], r["player"], r["event_id"]
    ))
    players_with_games = {r["player"] for r in games}
    if len(players_with_games) < min_players or len(games) < 150:
        raise RuntimeError(
            "Insufficient REAL ESPN records. No synthetic fallback. "
            f"Players={len(players_with_games)}, games={len(games)}. "
            f"Failures: {failures[:3]}"
        )
    return {
        "games": games, "failures": failures,
        "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "players": sorted(players_with_games),
        "source": "ESPN athlete regular-season game logs (public JSON endpoint)",
    }


def write_real_sqlite(games: list[dict], root: str | Path) -> Path:
    """Create temporary, local REAL-data database; never overwrite original DB."""
    path = Path(root) / "nba_public_real_espn.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS espn_real_player_games (
                athlete_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                player TEXT NOT NULL,
                game_date TEXT NOT NULL,
                season TEXT NOT NULL,
                team TEXT NOT NULL,
                opponent TEXT NOT NULL,
                home INTEGER NOT NULL,
                minutes REAL NOT NULL,
                pts REAL NOT NULL, reb REAL NOT NULL,
                ast REAL NOT NULL, three_pm REAL NOT NULL,
                pra REAL NOT NULL, source_url TEXT NOT NULL,
                PRIMARY KEY (athlete_id, event_id)
            )
        """)
        conn.execute("DELETE FROM espn_real_player_games")
        conn.executemany(
            "INSERT INTO espn_real_player_games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (g["athlete_id"], g["event_id"], g["player"], g["date"],
                 g["season"], g["team"], g["opponent"], g["home"], g["MIN"],
                 g["PTS"], g["REB"], g["AST"], g["3PM"], g["PRA"],
                 g["source_url"])
                for g in games
            ],
        )
    return path


def prepare_data(root: str | Path = "/content/NBA_Real_Data", athletes=None) -> dict:
    result = collect_real_games(athletes)
    result["db_path"] = str(write_real_sqlite(result["games"], root))
    return result


def upcoming_matchups(games: list[dict], days: int = 9) -> dict:
    """Find next actual scheduled ESPN matchup for each last-known player team.

    Schedule might be unavailable or change; never fabricate a fixture.
    """
    from datetime import timedelta
    teams = {}
    for game in sorted(games, key=lambda g: (g["date"], g["event_id"])):
        teams[game["player"]] = game["team"]
    today = datetime.now(timezone.utc).date()
    urls = [
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
        f"scoreboard?dates={(today + timedelta(days=i)).strftime('%Y%m%d')}&limit=100"
        for i in range(days)
    ]
    scheduled = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_json, url, 2) for url in urls]
        for future in as_completed(futures):
            try:
                payload = future.result()
            except Exception:
                continue
            for event in payload.get("events", []):
                event_id = str(event.get("id") or "")
                if not event_id.isdigit():
                    continue
                if event.get("status", {}).get("type", {}).get("completed"):
                    continue
                try:
                    start = datetime.fromisoformat(
                        event["date"].replace("Z", "+00:00")
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if start <= datetime.now(timezone.utc):
                    continue
                competitions = event.get("competitions", [])
                if not competitions:
                    continue
                competitors = competitions[0].get("competitors", [])
                home = next(
                    (c for c in competitors if c.get("homeAway") == "home"), None
                )
                away = next(
                    (c for c in competitors if c.get("homeAway") == "away"), None
                )
                if home is None or away is None:
                    continue
                ht, at = home.get("team", {}), away.get("team", {})
                for player, team in teams.items():
                    is_home = team == ht.get("abbreviation")
                    is_away = team == at.get("abbreviation")
                    if not (is_home or is_away):
                        continue
                    item = {
                        "date_utc": start.isoformat(timespec="minutes"),
                        "event_id": event_id,
                        "opponent": (
                            at.get("displayName", at.get("abbreviation"))
                            if is_home
                            else ht.get("displayName", ht.get("abbreviation"))
                        ),
                        "team": team, "home": bool(is_home),
                        "url": f"https://www.espn.com/nba/game/_/gameId/{event_id}",
                    }
                    if (
                        player not in scheduled or
                        item["date_utc"] < scheduled[player]["date_utc"]
                    ):
                        scheduled[player] = item
    return scheduled
