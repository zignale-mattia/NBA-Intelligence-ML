"""Structural checks for the independent, publicly shareable Colab notebook."""
import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_NOTEBOOK = ROOT / "notebooks/NBA_Intelligence_Public_Demo.ipynb"


class PublicNotebookTests(unittest.TestCase):
    def test_notebook_is_valid_empty_and_has_distinct_cell_ids(self):
        data = json.loads(DEMO_NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(data["nbformat"], 4)
        self.assertEqual(len(data["cells"]), 4)
        ids = [cell["id"] for cell in data["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in data["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            ast.parse("".join(cell["source"]))

    def test_notebook_is_independent_and_explicitly_simulated(self):
        data = json.loads(DEMO_NOTEBOOK.read_text(encoding="utf-8"))
        text = "\n".join("".join(c["source"]) for c in data["cells"])
        self.assertIn("SYNTHETIC DATA ONLY", text)
        self.assertIn("demo.public_demo", text)
        self.assertIn("demo.app", text)
        self.assertIn("Runtime", text)
        self.assertNotIn("drive.mount", text)
        self.assertNotIn("os.environ[\"OPENAI_API_KEY\"]", text)
        self.assertNotIn("nba_database.db", text)

    def test_badge_and_dashboard_path_configuration_are_discoverable(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        notebook = "notebooks/NBA_Intelligence_Public_Demo.ipynb"
        self.assertIn(notebook, readme)
        self.assertIn("colab-badge.svg", readme)
        self.assertIn("synthetic", readme.lower())
        dashboard = (
            ROOT / "dashboard/STEP_24_Dashboard_V5.py"
        ).read_text(encoding="utf-8")
        self.assertIn('BASE_DIR = os.environ.get("NBA_DATA_DIR")', dashboard)
        self.assertIn("nba_database.db", dashboard)


if __name__ == "__main__":
    unittest.main()
