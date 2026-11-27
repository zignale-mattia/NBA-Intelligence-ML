"""Structure and provenance checks for the real-data public Colab notebook."""
import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/NBA_Intelligence_Public_Demo.ipynb"


class PublicNotebookTests(unittest.TestCase):
    def test_notebook_has_no_outputs_and_valid_code(self):
        data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(data["nbformat"], 4)
        self.assertEqual(len(data["cells"]), 4)
        ids = [c["id"] for c in data["cells"]]
        self.assertEqual(len(set(ids)), len(ids))
        for cell in data["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                ast.parse("".join(cell["source"]))

    def test_notebook_uses_only_real_espn_pipeline(self):
        data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        content = "\n".join("".join(c["source"]) for c in data["cells"])
        self.assertIn("REAL NBA", content)
        self.assertIn("demo.real_espn", content)
        self.assertIn("demo.real_models", content)
        self.assertIn("demo.real_app", content)
        self.assertIn("Runtime", content)
        self.assertNotIn("demo.public_demo", content)
        self.assertNotIn("demo.app import", content)
        self.assertNotIn("drive.mount", content)
        self.assertNotIn("sk-proj-", content)

    def test_readme_badge_and_source_paths(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("notebooks/NBA_Intelligence_Public_Demo.ipynb", readme)
        self.assertIn("colab-badge.svg", readme)
        self.assertIn("real espn", readme.lower())
        self.assertIn("docs/COLAB.md", readme)


if __name__ == "__main__":
    unittest.main()
