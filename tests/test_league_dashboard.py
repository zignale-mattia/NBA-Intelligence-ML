"""Offline verification of expanded REAL-data league pipeline.

Test fixtures mimic ESPN structures. They are never shown as NBA observations.
"""
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from demo.analytics_views import (
    METRICS,describe,grouped_threshold,player_rows,comparison,
    player_table,match_table,trend_svg,bar_svg,snapshot,
)
from demo.league_explorer import (
    completed_games,parse_game_boxscore,build_league_explorer,
    persist_league_sqlite,leaderboards,team_performance,
)


def completed_event(game_id="401909319",regular=True):
    return {
        "id":game_id, "date":"2026-11-26T01:00Z",
        "season":{"type":2 if regular else 1},
        "status":{"type":{"completed":True}},
        "competitions":[{"competitors":[
            {"homeAway":"home",
             "team":{"displayName":"New York Knicks","abbreviation":"NY"},
             "score":"120"},
            {"homeAway":"away",
             "team":{"displayName":"Cleveland Cavaliers","abbreviation":"CLE"},
             "score":"103"},
        ]}],
    }


def summary():
    def athlete(aid,name,points,rebounds,assists):
        return {
            "didNotPlay":False,"starter":True,
            "athlete":{"id":aid,"displayName":name,
                       "position":{"abbreviation":"PG"}},
            "stats":[
                "32",str(points),"8-18","3-8","4-4",
                str(rebounds),str(assists),"2","1","0",
                "1","4","1","+9",
            ],
        }
    labels=["MIN","PTS","FG","3PT","FT","REB","AST","TO","STL",
            "BLK","OREB","DREB","PF","+/-"]
    return {
        "boxscore":{"players":[
            {"team":{"displayName":"New York Knicks"},
             "statistics":[{"labels":labels,
                             "athletes":[athlete("3934672","Jalen Brunson",23,4,8)]}]},
            {"team":{"displayName":"Cleveland Cavaliers"},
             "statistics":[{"labels":labels,
                             "athletes":[athlete("3908809","Donovan Mitchell",28,6,3)]}]},
        ]}
    }


class LeagueParserTests(unittest.TestCase):
    def test_season_2_only_and_completed_games(self):
        def mock_fetch(url,attempts=2):
            return {"events":[completed_event("1"),completed_event("2",False),
                              {"id":"3","status":{"type":{"completed":False}},
                               "season":{"type":2}}]}
        with patch("demo.league_explorer.fetch_json",side_effect=mock_fetch):
            events,errors=completed_games(days=2,through=date(2026,11,27))
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]["id"],"1")
        self.assertFalse(errors)

    def test_full_boxscore_comes_from_provider(self):
        players,game=parse_game_boxscore(summary(),completed_event())
        self.assertEqual(len(players),2)
        self.assertEqual(game["home_score"],120)
        self.assertEqual(game["away_score"],103)
        brunson=next(p for p in players if p["player"]=="Jalen Brunson")
        self.assertEqual(brunson["PTS"],23)
        self.assertEqual(brunson["PRA"],35)
        self.assertEqual(brunson["3PM"],3)
        self.assertEqual(brunson["3PA"],8)
        self.assertEqual(brunson["FGA"],18)
        self.assertTrue(brunson["home"])
        self.assertIn("401909319",brunson["source_url"])
        mitchell=next(p for p in players if p["player"]=="Donovan Mitchell")
        self.assertEqual(mitchell["home"],0)

    def test_incomplete_athletes_are_not_invented(self):
        s=summary()
        s["boxscore"]["players"][0]["statistics"][0]["athletes"][0]["didNotPlay"]=True
        players,_=parse_game_boxscore(s,completed_event())
        self.assertEqual([r["player"] for r in players],["Donovan Mitchell"])
        players,game=parse_game_boxscore(s,completed_event(regular=False))
        # This parser expects upstream regular season filtering.
        self.assertEqual(len(players),1)

    def test_persistent_sqlite_reflects_real_records_only(self):
        players,game=parse_game_boxscore(summary(),completed_event())
        with tempfile.TemporaryDirectory() as temp:
            path=persist_league_sqlite(
                {"rows":players,"matches":[game]},temp
            )
            self.assertTrue(Path(path).exists())
            self.assertNotEqual(Path(path).name,"nba_database.db")
            with sqlite3.connect(path) as con:
                self.assertEqual(con.execute(
                    "SELECT COUNT(*) FROM completed_games"
                ).fetchone()[0],1)
                self.assertEqual(con.execute(
                    "SELECT COUNT(*) FROM player_boxscores"
                ).fetchone()[0],2)
                self.assertEqual(con.execute(
                    "SELECT three_pm,pts FROM player_boxscores "
                    "WHERE player='Jalen Brunson'"
                ).fetchone(),(3.0,23.0))

    def test_leaderboards_and_team_performance_from_actual_rows(self):
        p,g=parse_game_boxscore(summary(),completed_event())
        rankings=leaderboards(p,"PTS",min_games=1)
        self.assertEqual(rankings[0]["player"],"Donovan Mitchell")
        self.assertEqual(rankings[0]["mean"],28.0)
        team=team_performance([g])
        self.assertEqual(team[0]["team"],"New York Knicks")
        self.assertEqual(team[0]["W"],1)
        self.assertEqual(team[0]["win_pct"],100.0)


class ViewsTests(unittest.TestCase):
    def setUp(self):
        self.players,self.match=parse_game_boxscore(summary(),completed_event())

    def test_player_statistics_and_thresholds(self):
        one=[p for p in self.players if p["player"]=="Jalen Brunson"]
        self.assertEqual(describe(one,"PTS")["avg"],23)
        self.assertEqual(grouped_threshold(one,"PTS",20),100)
        self.assertEqual(grouped_threshold(one,"PTS",24),0)
        self.assertEqual(player_rows(self.players,"Jalen Brunson"),one)
        self.assertEqual(comparison(self.players,["Jalen Brunson"],"PTS",
                                    limit=10)[0]["avg"],23.0)

    def test_report_tables_and_data_lineage(self):
        columns,rows=player_table(self.players,limit=1)
        self.assertEqual(len(rows),1)
        self.assertIn("ESPN box score",columns)
        self.assertEqual(match_table([self.match])[0][0],"2026-11-26")
        counts=snapshot(self.players,[self.match],["Jalen Brunson"])
        self.assertEqual(counts["performances"],2)
        self.assertEqual(counts["athletes"],2)

    def test_svg_escapes_untrusted_names_and_uses_real_values(self):
        rows=[dict(self.players[0],opponent="<script>alert(1)</script>")]
        graphic=trend_svg(rows,"PTS",title="ESPN score")
        self.assertIn("23",graphic)
        self.assertNotIn("<script>",graphic)
        graphic=bar_svg([{"player":"<bad>","team":"NY","mean":22,
                         "games":5}])
        self.assertIn("&lt;bad&gt;",graphic)

    def test_unsupported_metric_fails(self):
        with self.assertRaises(ValueError):
            describe(self.players,"INVALID")


if __name__=="__main__":
    unittest.main()
