"""Tests that never touch a real NBA database or Google Drive."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.project_config import resolve_data_dir


class DataDirectoryTests(unittest.TestCase):
    def test_prefers_override(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {"NBA_DATA_DIR": str(Path(temp) / "custom")}):
                self.assertEqual(resolve_data_dir(), Path(temp) / "custom")

    def test_explicit_override_wins_over_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {"NBA_DATA_DIR": "ignored"}):
                self.assertEqual(resolve_data_dir(temp), Path(temp))

    def test_prefers_existing_database_to_empty_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            empty, populated = base / "empty", base / "with_db"
            empty.mkdir()
            populated.mkdir()
            (populated / "nba_database.db").touch()
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(resolve_data_dir(candidates=(empty, populated)), populated)

    def test_can_require_database(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                resolve_data_dir(temp, require_database=True)

    def test_missing_candidates_is_explicit_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_data_dir(candidates=())


if __name__ == "__main__":
    unittest.main()
